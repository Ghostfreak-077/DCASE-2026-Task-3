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
    def __init__(
        self,
        n_classes:      int   = 14,
        in_ch:          int   = 10,
        img_h:          int   = 180,
        img_w:          int   = 360,
        energy_annot_w: float = 5.0,
        dist_w:         float = 15.0,
        dropout:        float = 0.1,
        class_weights:  torch.Tensor = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.img_h = img_h
        self.img_w = img_w
        self.energy_annot_w = energy_annot_w
        self.dist_w = dist_w

        if class_weights is None:
            class_weights = torch.ones(n_classes)
        self.register_buffer("class_weights", class_weights)

        # ── Encoder (asymmetric stride) ──────────────────────────────
        # Input: (B, 10, 8, 128)
        self.enc0 = _EncoderBlock(in_ch,  64,  stride=(1, 1), dropout=dropout)  # (B, 64, 8, 128)
        self.enc1 = _EncoderBlock(64,  128,     stride=(1, 2), dropout=dropout)  # (B, 128, 8, 64)
        self.enc2 = _EncoderBlock(128, 256,     stride=(1, 2), dropout=dropout)  # (B, 256, 8, 32)
        self.enc3 = _EncoderBlock(256, 512,     stride=(1, 2), dropout=dropout)  # (B, 512, 8, 16) [BOTTLENECK]

        # ── Decoder blocks ───────────────────────────────────────────
        self.dec2 = _DecoderBlock(512, 256, 256, dropout=dropout)  # (512, 8,16) + skip(256, 16,32) → (256, 16, 32)
        self.dec1 = _DecoderBlock(256, 128, 128, dropout=dropout)  # (256, 16,32) + skip(128, 32,64) → (128, 32, 64)
        self.dec0 = _DecoderBlock(128,  64,  64, dropout=dropout)  # (128, 32,64) + skip(64, 64,128) → (64, 64, 128)

        # ── Final upsample to (180, 360) ─────────────────────────────
        self.final = nn.Sequential(
            _ConvBnRelu(64, 32),
            _ConvBnRelu(32, 32),
        )

        # ── Skip connection UP-samplers (NOT poolers!) ────────────────
        # Adapt encoder outputs to match decoder input sizes
        self.skip2_up = nn.Upsample(size=(16, 32), mode='bilinear', align_corners=False)
        self.skip1_up = nn.Upsample(size=(32, 64), mode='bilinear', align_corners=False)
        self.skip0_up = nn.Upsample(size=(64, 128), mode='bilinear', align_corners=False)

        # ── Output heads ─────────────────────────────────────────────
        self.energy_head   = _EnergyMapHead(32, n_classes)
        self.mask_head     = _InstanceMaskHead(32, n_classes, dropout=0.2)
        self.distance_head = _DistanceHead(512, n_classes, hidden=256, dropout=0.3)

    # ------------------------------------------------------------------
    def _encode(self, x):
        """Encoder: (B, 10, 8, 128) → bottleneck (B, 512, 8, 16)"""
        e0 = self.enc0(x)   # (B,  64, 8, 128)
        e1 = self.enc1(e0)  # (B, 128, 8,  64)
        e2 = self.enc2(e1)  # (B, 256, 8,  32)
        e3 = self.enc3(e2)  # (B, 512, 8,  16) [BOTTLENECK]
        return e0, e1, e2, e3

    def _decode(self, e0, e1, e2, e3):
        """Decoder: bottleneck (B, 512, 8, 16) → final (B, 32, 180, 360)"""
        # dec2: (B, 512, 8, 16) + skip(B, 256, 16, 32) → (B, 256, 16, 32)
        skip2 = self.skip2_up(e2)  # (B, 256, 8, 32) → (B, 256, 16, 32)
        d = self.dec2(e3, skip2, (16, 32))

        # dec1: (B, 256, 16, 32) + skip(B, 128, 32, 64) → (B, 128, 32, 64)
        skip1 = self.skip1_up(e1)  # (B, 128, 8, 64) → (B, 128, 32, 64)
        d = self.dec1(d, skip1, (32, 64))

        # dec0: (B, 128, 32, 64) + skip(B, 64, 64, 128) → (B, 64, 64, 128)
        skip0 = self.skip0_up(e0)  # (B, 64, 8, 128) → (B, 64, 64, 128)
        d = self.dec0(d, skip0, (64, 128))

        # Final upsample: (B, 64, 64, 128) → (B, 32, 180, 360)
        d = F.interpolate(d, size=(self.img_h, self.img_w),
                          mode='bilinear', align_corners=False)
        return self.final(d)  # (B, 32, 180, 360)

    # ------------------------------------------------------------------
    def forward(self, images: list, gt_xys: torch.Tensor = None, epoch: int = 99):
        x = torch.stack(images, dim=0)  # (B, C_in=10, T=8, F=128)
        
        # Encode
        e0, e1, e2, e3 = self._encode(x)
        
        # Decode
        d = self._decode(e0, e1, e2, e3)  # (B, 32, 180, 360)
        
        # Heatmap head
        heatmap = torch.sigmoid(self.energy_head(d)[:, 0:1])  # (B, 1, 180, 360)
        
        # # Soft-argmax to XY
        # B, _, H, W = heatmap.shape
        # hmap_flat = heatmap.view(B, -1)
        # hmap_prob = torch.softmax(hmap_flat * 10, dim=1).view(B, H, W)
        
        # gy = torch.linspace(0, 1, H, device=x.device)
        # gx = torch.linspace(0, 1, W, device=x.device)
        
        # pred_y = (hmap_prob.sum(2) * gy).sum(1)
        # pred_x = (hmap_prob.sum(1) * gx).sum(1)
        # pred_xy = torch.stack([pred_x, pred_y], dim=1)
        
        # return pred_xy

        return heatmap