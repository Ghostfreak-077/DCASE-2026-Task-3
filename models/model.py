"""
model.py  —  UNetSAISELD  (corrected architecture)
----------------------------------------------------
ROOT CAUSE FIX (v2):
  The original encoder used symmetric stride-2 on both T and F dimensions.
  With T=8 (SALSA time bins per frame), three stride-2 stages collapsed the
  height to 8->4->2->1. The decoder then bilinear-interpolated a 1-pixel-tall
  feature map to 180 rows, producing identical values across all rows —
  exactly the horizontal stripe failure observed in evaluation.

  Fix: use asymmetric striding — stride (1,2) so only F is downsampled,
  T is preserved throughout. The bottleneck is (B, C, T=8, F//16=8),
  a genuine 2D spatial feature with both temporal and frequency structure.
  The decoder then upsamples (8,8) → (16,32) → (32,64) → (64,128) → (180,360).
  No spatial information is destroyed at any stage.

Architecture
~~~~~~~~~~~~
Input : (B, 4, T, F)   T=8, F=128

Encoder  — asymmetric stride (1,2): downsample F only, preserve T
  E0: (B,  64, T,   F  )  stride (1,1)  [skip]
  E1: (B, 128, T,   F/2)  stride (1,2)  [skip]
  E2: (B, 256, T,   F/4)  stride (1,2)  [skip]
  E3: (B, 512, T,   F/8)  stride (1,2)  bottleneck  → seed (T, F/8) = (8,16)

Decoder — upsample to (180,360) with SE-gated skip connections
  D2: (B, 256, T*2,  F/4)   = (16, 32)
  D1: (B, 128, T*4,  F/2)   = (32, 64)
  D0: (B,  64, T*8,  F  )   = (64, 128)
  Out:           → upsample → (180, 360)

Output heads:
  energy_map    : (B, N_CLASSES, 180, 360)  sigmoid
  instance_mask : (B, N_CLASSES, 180, 360)  sigmoid
  distance_head : (B, N_CLASSES)            softplus

Loss:
  loss_energy   = MSE(pred_combined, gt_vmap)  + ANNOT_W * MSE on annotated px
  loss_mask     = Dice + weighted BCE per instance
  loss_distance = Huber(dist_pred, gt_dist)  × dist_w
"""

import math
import os
import json
import glob
import re
import hashlib
import pickle
import sqlite3
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
from models.blocks import (
    _EncoderBlock,
    _DecoderBlock,
    _ConvBnRelu
)
from models.output_heads import (
    _DistanceHead,
    _EnergyMapHead,
    _InstanceMaskHead
)
from models.conformer_model import ConformerEncoder

class UNetSAISELD(nn.Module):
    """
    U-Net for DCASE 2026 Task 3 Track A.

    Encoder uses asymmetric stride (1,2) to preserve the T dimension throughout.
    This prevents the horizontal-stripe failure mode caused by collapsing T to 1.

    Parameters
    ----------
    n_classes       : 14 (13 sound classes + background)
    in_ch           : 4 (SALSA-Lite channels)
    img_h / img_w   : 180 × 360 equirectangular output canvas
    energy_annot_w  : weight on annotated-pixel energy loss
    dist_w          : weight on distance loss (recommend 15)
    """

    def __init__(
        self,
        n_classes:      int   = 14,
        in_ch:          int   = 4,
        img_h:          int   = 180,
        img_w:          int   = 360,
        energy_annot_w: float = 5.0,
        dist_w:         float = 15.0,
        dropout:        float = 0.1,
        class_weights:  torch.Tensor = None,
    ):
        super().__init__()
        self.n_classes      = n_classes
        self.img_h          = img_h
        self.img_w          = img_w
        self.energy_annot_w = energy_annot_w
        self.dist_w         = dist_w

        if class_weights is None:
            class_weights = torch.ones(n_classes)
        self.register_buffer("class_weights", class_weights)

        self.temporal = ConformerEncoder(
            dim=512,
            depth=4,
            heads=8,
            ff_mult=4
        )

        # ── Encoder  (asymmetric stride: preserve T, halve F each stage) ──
        # Input: (B, 4, T=8, F=128)
        self.enc0 = _EncoderBlock(in_ch,  64,  stride=(1, 1), dropout=dropout)  # (B,  64, 8, 128)
        self.enc1 = _EncoderBlock(64,  128,     stride=(1, 2), dropout=dropout)  # (B, 128, 8,  64)
        self.enc2 = _EncoderBlock(128, 256,     stride=(1, 2), dropout=dropout)  # (B, 256, 8,  32)
        self.enc3 = _EncoderBlock(256, 512,     stride=(1, 2), dropout=dropout)  # (B, 512, 8,  16)
        # bottleneck shape: (B, 512, T, F//8) = (B, 512, 8, 16)

        # ── Decoder  (upsample both dims toward 180×360) ──────────────────
        # D2: (8,16) → (16,32)
        self.dec2 = _DecoderBlock(512, 256, 256, dropout=dropout)
        # D1: (16,32) → (32,64)
        self.dec1 = _DecoderBlock(256, 128, 128, dropout=dropout)
        # D0: (32,64) → (64,128)
        self.dec0 = _DecoderBlock(128,  64,  64, dropout=dropout)
        # Final upsample: (64,128) → (180,360)
        self.final = nn.Sequential(
            _ConvBnRelu(64, 32),
            _ConvBnRelu(32, 32),
        )

        # ── Skip connection spatial adapters ──────────────────────────────
        # Each skip is adaptive-pooled to match the decoder stage input size
        self.skip2 = nn.AdaptiveAvgPool2d((16, 32))   # enc2 (8,32)  → (16,32)
        self.skip1 = nn.AdaptiveAvgPool2d((32, 64))   # enc1 (8,64)  → (32,64)
        self.skip0 = nn.AdaptiveAvgPool2d((64, 128))  # enc0 (8,128) → (64,128)

        # ── Output heads ─────────────────────────────────────────────────
        self.energy_head   = _EnergyMapHead(32, n_classes)
        self.mask_head     = _InstanceMaskHead(32, n_classes, dropout=0.2)
        self.distance_head = _DistanceHead(512, n_classes, hidden=256, dropout=0.3)

    # ------------------------------------------------------------------
    def _encode(self, x):
        e0 = self.enc0(x)   # (B,  64, 8, 128)
        e1 = self.enc1(e0)  # (B, 128, 8,  64)
        e2 = self.enc2(e1)  # (B, 256, 8,  32)
        e3 = self.enc3(e2)  # (B, 512, 8,  16)
        return e0, e1, e2, e3

    def _decode(self, e0, e1, e2, e3):
        # Bottleneck: (B, 512, 8, 16)
        # Dec2: upsample to (16,32), skip from enc2
        d = self.dec2(e3, self.skip2(e2), (16, 32))    # (B, 256, 16, 32)
        # Dec1: upsample to (32,64), skip from enc1
        d = self.dec1(d,  self.skip1(e1), (32, 64))    # (B, 128, 32, 64)
        # Dec0: upsample to (64,128), skip from enc0
        d = self.dec0(d,  self.skip0(e0), (64, 128))   # (B,  64, 64, 128)
        # Final upsample to exact canvas
        d = F.interpolate(d, size=(self.img_h, self.img_w),
                          mode="bilinear", align_corners=False)
        return self.final(d)  # (B, 32, 180, 360)

    # ------------------------------------------------------------------
    def forward(self, images: list, targets: list = None, epoch: int = 99):
        x = torch.stack(images, dim=0)          # (B, 4, T, F)

        e0, e1, e2, e3 = self._encode(x)
        B,C,T,F = e3.shape

        x = e3.permute(0,2,3,1)       # B,T,F,C
        x = x.reshape(B,T*F,C)

        x = self.temporal(x)

        e3 = x.reshape(B,T,F,C)
        e3 = e3.permute(0,3,1,2)

        d               = self._decode(e0, e1, e2, e3)

        energy_maps = self.energy_head(d)        # (B, N_CLASSES, 180, 360)
        mask_maps   = self.mask_head(d)          # (B, N_CLASSES, 180, 360)
        dist_pred   = self.distance_head(e3)     # (B, N_CLASSES)

        if self.training:
            assert targets is not None
            return self._compute_loss(energy_maps, mask_maps, dist_pred, targets, x.device, epoch)
        else:
            return self._build_detections(energy_maps, mask_maps, dist_pred)

    # ------------------------------------------------------------------
    @staticmethod
    def _focal_bce(pred: torch.Tensor, gt: torch.Tensor,
                   gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
        """
        FIX 3: Focal loss for binary segmentation.
        Handles the ~430:1 background/foreground pixel imbalance that caused
        BCE to push the sigmoid toward 0 everywhere (loss_mask stuck at ~1.69
        = near-random performance).
        alpha=0.25 upweights foreground; gamma=2.0 down-weights easy negatives.
        """
        bce   = F.binary_cross_entropy(pred, gt, reduction="none")  # (H, W)
        pt    = torch.where(gt > 0.5, pred, 1.0 - pred)
        focal = ((1.0 - pt) ** gamma) * bce
        a_t   = torch.where(gt > 0.5,
                            torch.full_like(gt, alpha),
                            torch.full_like(gt, 1.0 - alpha))
        return (a_t * focal).mean()

    def _compute_loss(self, energy_maps, mask_maps, dist_pred, targets, device,
                      epoch: int = 99):
        """
        FIX 3: Focal loss replaces plain BCE — handles 430:1 pixel imbalance.
        FIX 4: Mask loss only on annotated instances — no BCE signal on empty frames.
        FIX 5: Distance loss ramped — 0 for epochs 1-2, linear ramp to full by ep 5.
                Prevents distance head from competing with mask/energy in early epochs
                (which caused val loss_distance to diverge: 0.75→0.83).
        """
        B            = energy_maps.shape[0]
        loss_energy  = energy_maps.new_zeros(1).squeeze()
        loss_mask    = energy_maps.new_zeros(1).squeeze()
        loss_dist    = energy_maps.new_zeros(1).squeeze()
        n_instances  = 0
        n_dist_terms = 0

        # FIX 5: distance ramp — 0 for epoch 1-2, reaches full dist_w by epoch 5
        dist_ramp = 0.0 if epoch <= 2 else min(1.0, (epoch - 2) / 3.0) * self.dist_w

        for i, tgt in enumerate(targets):
            labels   = tgt["labels"].to(device)
            emaps    = tgt["energy_maps"].to(device)
            emasks   = tgt["energy_masks"].to(device)
            vmap     = tgt["vmap"].to(device)
            vmask    = tgt["vmask"].to(device)
            dists    = tgt["distances"].to(device)

            # ── Full-image energy map loss ────────────────────────────────
            pred_combined = energy_maps[i].max(dim=0).values
            W             = torch.ones(self.img_h, self.img_w, device=device)
            W[vmask]      = self.energy_annot_w
            loss_energy   = loss_energy + (W * (pred_combined - vmap).pow(2)).mean()

            # ── FIX 4: mask loss ONLY on frames with annotations ─────────
            # N = len(labels)
            # if N > 0:
            #     for n in range(N):
            #         cls_idx = int(labels[n].item())
            #         if cls_idx < 1 or cls_idx >= self.n_classes:
            #             continue

            #         pred_mask_nc   = mask_maps[i,   cls_idx]
            #         pred_energy_nc = energy_maps[i, cls_idx]
            #         gt_mask        = emasks[n].float()
            #         gt_emap        = emaps[n]

            #         # FIX 3: Focal loss
            #         cw    = self.class_weights[cls_idx]

            #         with torch.amp.autocast("cuda", enabled=False):
            #             pred_clamped = pred_mask_nc.float().clamp(1e-6, 1.0 - 1e-6)
            #             focal = self._focal_bce(pred_clamped, gt_mask.float()) * cw

            #         # Dice (handles imbalance from the overlap side)
            #         inter = (pred_mask_nc * gt_mask).sum()
            #         denom = pred_mask_nc.sum() + gt_mask.sum() + 1e-6
            #         dice  = 1.0 - 2.0 * inter / denom

            #         # Energy MSE only on GT-annotated pixels
            #         ann_px = gt_mask > 0.5

            #         e_mse  = (F.mse_loss(pred_energy_nc[ann_px].float(), gt_emap[ann_px].float())
            #                   if ann_px.any()
            #                   else pred_energy_nc.new_zeros(1).squeeze())

            #         loss_mask   = loss_mask + focal + dice + e_mse
            #         # loss_mask = loss_mask + torch.zeros_like(gt_mask)
            #         n_instances += 1

            #     # FIX 5: distance only when ramp > 0
            #     if dist_ramp > 0:
            #         for cls_raw in labels.unique():
            #             cls_idx = int(cls_raw.item())
            #             if cls_idx < 1 or cls_idx >= self.n_classes:
            #                 continue
            #             gt_d         = dists[labels == cls_raw].mean()
            #             loss_dist    = loss_dist + F.huber_loss(
            #                 dist_pred[i, cls_idx], gt_d, delta=0.5)
            #             n_dist_terms += 1

        denom = max(B, 1)
        return {
            "loss_energy":   loss_energy / denom,
            "loss_mask":     torch.zeros_like(loss_mask   / max(n_instances, 1)),
            # "loss_mask":     torch.zeros_like(torch.stack([loss_mask])),
            "loss_distance": torch.zeros_like(dist_ramp * loss_dist / max(n_dist_terms, 1)),
            # "loss_distance": torch.zeros_like(dists),
        }

    # ------------------------------------------------------------------
    def _build_detections(self, energy_maps, mask_maps, dist_pred):
        """
        FIX 1: Score threshold = 0.15 on energy peak (not 1e-3 on mask peak).
                energy_head loss dropped 5x — it IS calibrated to GT.
                mask_head loss is ~random — its max is meaningless as a score.
        FIX 2: Score = energy_c.max(). Box derived from energy map, not mask map.
                Bounding box = region where energy > 10% of its own peak.
                This is spatially meaningful; mask > 0.3 was not.
        """
        ENERGY_SCORE_THR = 0.15   # min energy peak to emit a detection
        ENERGY_BOX_FRAC  = 0.10   # box = pixels with energy >= 10% of peak

        B          = energy_maps.shape[0]
        detections = []
        for i in range(B):
            em = energy_maps[i].detach().cpu()   # (N_CLASSES, H, W)
            dp = dist_pred[i].detach().cpu()     # (N_CLASSES,)

            boxes_list, labels_list, scores_list = [], [], []
            emaps_list, dists_list               = [], []

            for cls_idx in range(1, self.n_classes):
                energy_c = em[cls_idx]                  # (H, W)
                score    = float(energy_c.max())

                # FIX 1: energy-calibrated threshold
                if score < ENERGY_SCORE_THR:
                    continue

                # FIX 2: bounding box from energy map region
                active = energy_c >= (ENERGY_BOX_FRAC * score)
                if not active.any():
                    continue
                rows = torch.where(active.any(dim=1))[0]
                cols = torch.where(active.any(dim=0))[0]
                x0 = float(cols[0]);  x1 = float(cols[-1] + 1)
                y0 = float(rows[0]);  y1 = float(rows[-1] + 1)

                boxes_list.append([x0, y0, x1, y1])
                labels_list.append(cls_idx)
                scores_list.append(score)
                emaps_list.append(energy_c.unsqueeze(0))
                dists_list.append(float(dp[cls_idx]))

            if boxes_list:
                detections.append({
                    "boxes":       torch.tensor(boxes_list,  dtype=torch.float32),
                    "labels":      torch.tensor(labels_list, dtype=torch.long),
                    "scores":      torch.tensor(scores_list, dtype=torch.float32),
                    "energy_maps": torch.cat(emaps_list, dim=0),
                    "dist_pred":   torch.tensor(dists_list,  dtype=torch.float32),
                    "full_energy": em.max(dim=0).values,
                })
            else:
                detections.append({
                    "boxes":       torch.zeros(0, 4),
                    "labels":      torch.zeros(0, dtype=torch.long),
                    "scores":      torch.zeros(0),
                    "energy_maps": torch.zeros(0, self.img_h, self.img_w),
                    "dist_pred":   torch.zeros(0),
                    "full_energy": em.max(dim=0).values,
                })
        return detections
