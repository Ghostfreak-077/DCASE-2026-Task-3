"""
model.py
--------
UNetSAISELD — Path B1 replacement for EnergyInstanceModel.

Architecture
~~~~~~~~~~~~
Input : (B, 4, T, F)  T≈8, F=128  (SALSA-Lite features)

Encoder (4 stages, with circular-pad on F axis for azimuth continuity):
  E0: Conv block  → (B,  64, T,   F  )   [stride-1, skip]
  E1: Conv block  → (B, 128, T/2, F/2)   [stride-2, skip]
  E2: Conv block  → (B, 256, T/4, F/4)   [stride-2, skip]
  E3: Conv block  → (B, 512, T/8, F/8)   [stride-2, bottleneck]

Spatial Projection Bridge:
  Bilinear upsample from encoder bottleneck to a fixed (H0, W0) spatial seed,
  then progressive decode to (180, 360) equirectangular canvas.
  Seed shape chosen so 3 doubling stages reach 180×360.

Decoder (with SE-gated skip connections):
  D3: upsample + skip → (B, 256, 22,  45)
  D2: upsample + skip → (B, 128, 45,  90)
  D1: upsample + skip → (B,  64, 90, 180)
  D0: upsample + skip → (B,  32, 180, 360)

Output heads (all on the 180×360 canvas):
  energy_map   : (B, N_CLASSES, 180, 360)  — per-class energy field  [sigmoid]
  instance_mask: (B, N_CLASSES, 180, 360)  — per-class binary mask   [sigmoid]
  distance_head: (B, N_CLASSES)            — per-class distance scalar

Loss
~~~~
  loss_energy  = MSE(energy_map, gt_vmap)  + ANNOT_W * MSE on annotated pixels
  loss_mask    = Dice + BCE  (per instance, accumulated to class-level)
  loss_distance= Huber(dist_pred, gt_dist)

All three losses are returned as a dict in training mode.
In eval mode, a list of per-image dicts is returned (submission-compatible).

Reused from baseline
~~~~~~~~~~~~~~~~~~~~
  EnergySegDataset, collate_fn, worker_init_fn, get_sequence_infos,
  InstanceTracker  (all imported directly — no changes needed)
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from acoustic_features import wav_path_from_seq_dir

# ════════════════════════════════════════════════════════════════════════════
# 1.  BUILDING BLOCKS
# ════════════════════════════════════════════════════════════════════════════

def _circ_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Circular pad on the width (last) dimension, reflect on height."""
    x = F.pad(x, (pad, pad, 0, 0), mode="circular")   # azimuth wrap
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")    # elevation reflect
    return x


class _CircConv2d(nn.Module):
    """Conv2d with circular horizontal + reflect vertical padding."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, bias=False):
        super().__init__()
        self.pad    = kernel // 2
        self.stride = stride
        self.conv   = nn.Conv2d(
            in_ch, out_ch, kernel,
            stride=stride, padding=0, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = _circ_pad(x, self.pad)
        return self.conv(x)


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            _CircConv2d(in_ch, out_ch, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _EncoderBlock(nn.Module):
    """Two conv-bn-relu + optional stride-2 downsampling."""
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            _ConvBnRelu(in_ch,  out_ch, stride=stride),
            _ConvBnRelu(out_ch, out_ch, stride=1),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(1, ch // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.shape[0], -1, 1, 1)
        return x * w


class _DecoderBlock(nn.Module):
    """
    Upsample → concat skip (SE-gated) → two conv-bn-relu.
    Target size is passed at forward time for exact alignment.
    """
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.1):
        super().__init__()
        self.se   = _SEBlock(skip_ch)
        self.conv = nn.Sequential(
            _ConvBnRelu(in_ch + skip_ch, out_ch),
            _ConvBnRelu(out_ch,          out_ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x, skip, target_size):
        x    = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        skip = self.se(skip)
        x    = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ════════════════════════════════════════════════════════════════════════════
# 2.  OUTPUT HEADS
# ════════════════════════════════════════════════════════════════════════════

class _EnergyMapHead(nn.Module):
    """Per-class energy map regression head → (B, N_CLASSES, H, W)."""
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            _ConvBnRelu(in_ch, in_ch),
            _CircConv2d(in_ch, n_classes, kernel=1, bias=True),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(x)


class _InstanceMaskHead(nn.Module):
    """Per-class binary mask head → (B, N_CLASSES, H, W)."""
    def __init__(self, in_ch, n_classes, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(
            _ConvBnRelu(in_ch, in_ch),
            nn.Dropout2d(dropout),
            _CircConv2d(in_ch, n_classes, kernel=1, bias=True),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(x)


class _DistanceHead(nn.Module):
    """
    Per-class distance regression.
    Uses global average pool of deepest encoder feature → MLP.
    Returns (B, N_CLASSES) — one range estimate per class per frame.
    """
    def __init__(self, bottleneck_ch, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp  = nn.Sequential(
            nn.Linear(bottleneck_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        x = self.pool(bottleneck).flatten(1)   # (B, bottleneck_ch)
        return F.softplus(self.mlp(x))          # (B, N_CLASSES)  ≥ 0


# ════════════════════════════════════════════════════════════════════════════
# 3.  MAIN MODEL
# ════════════════════════════════════════════════════════════════════════════

class UNetSAISELD(nn.Module):
    """
    Lightweight U-Net for DCASE 2026 Task 3 Track A.

    Parameters
    ----------
    n_classes       : number of sound classes including background (14)
    in_ch           : SALSA-Lite channels (4)
    img_h / img_w   : output equirectangular canvas size (180 × 360)
    energy_annot_w  : weight on annotated-pixel energy loss
    dist_w          : weight on distance loss
    class_weights   : (n_classes,) tensor for weighted mask BCE
    """

    # Seed spatial resolution for the bridge
    # Three doubling stages: 22→45→90→180 (h), 45→90→180→360 (w)
    _SEED_H = 22
    _SEED_W = 45

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

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc0 = _EncoderBlock(in_ch,   64,  stride=(1,2), dropout=dropout)  # (B,  64, T,   F  )
        self.enc1 = _EncoderBlock(64,      128, stride=(1,2), dropout=dropout)  # (B, 128, T/2, F/2)
        self.enc2 = _EncoderBlock(128,     256, stride=(1,2), dropout=dropout)  # (B, 256, T/4, F/4)
        self.enc3 = _EncoderBlock(256,     512, stride=(1,2), dropout=dropout)  # (B, 512, ?,   ?  )

        # ── Bridge: project bottleneck to spatial seed ────────────────────
        # After enc3 the spatial dims depend on T and F.
        # We use AdaptiveAvgPool to a fixed (SEED_H, SEED_W) then upsample.
        # self.bridge_pool = nn.AdaptiveAvgPool2d((self._SEED_H, self._SEED_W))
        self.bridge_conv = _ConvBnRelu(512, 512)

        # ── Decoder ──────────────────────────────────────────────────────
        # We need skip features at the same spatial size as each decode step.
        # Since encoder and decoder are in different spatial domains
        # (T×F vs H×W), we project skips through adaptive pooling.
        self.skip_pool3 = nn.AdaptiveAvgPool2d((self._SEED_H * 2,     self._SEED_W *2   ))   # for enc2 skip
        self.skip_pool2 = nn.AdaptiveAvgPool2d((self._SEED_H * 4, self._SEED_W * 4))   # for enc1 skip
        self.skip_pool1 = nn.AdaptiveAvgPool2d((self._SEED_H * 8, self._SEED_W * 8))   # for enc0 skip

        self.dec3 = _DecoderBlock(512, 256, 256, dropout=dropout)  # seed → seed*2
        self.dec2 = _DecoderBlock(256, 128, 128, dropout=dropout)  # → seed*4
        self.dec1 = _DecoderBlock(128,  64,  64, dropout=dropout)  # → seed*8
        # Final upsample to full 180×360
        self.dec0 = nn.Sequential(
            _ConvBnRelu(64, 32),
            _ConvBnRelu(32, 32),
        )

        # ── Output heads ─────────────────────────────────────────────────
        self.energy_head   = _EnergyMapHead(32, n_classes)
        self.mask_head     = _InstanceMaskHead(32, n_classes, dropout=0.2)
        self.distance_head = _DistanceHead(512, n_classes, hidden=256, dropout=0.3)

    # ------------------------------------------------------------------
    def _encode(self, x):
        e0 = self.enc0(x)   # (B,  64, T,   F)
        e1 = self.enc1(e0)  # (B, 128, T/2, F/2)
        e2 = self.enc2(e1)  # (B, 256, T/4, F/4)
        e3 = self.enc3(e2)  # (B, 512, ?, ?)
        return e0, e1, e2, e3

    # ------------------------------------------------------------------
    def _decode(self, e0, e1, e2, e3):
        # b = self.bridge_conv(self.bridge_pool(e3))  # (B, 512, SEED_H, SEED_W)

        s2 = self.skip_pool3(e2)  # (B, 256, SEED_H,   SEED_W  )
        s1 = self.skip_pool2(e1)  # (B, 128, SEED_H*2, SEED_W*2)
        s0 = self.skip_pool1(e0)  # (B,  64, SEED_H*4, SEED_W*4)

        d = self.dec3(e3,  s2, (self._SEED_H * 2, self._SEED_W * 2))   # (B, 256, 44, 90)
        d = self.dec2(d,  s1, (self._SEED_H * 4, self._SEED_W * 4))   # (B, 128, 88, 180)
        d = self.dec1(d,  s0, (self._SEED_H * 8, self._SEED_W * 8))   # (B,  64, 176, 360)
        # Final upsample to exact canvas size
        d = F.interpolate(d, size=(self.img_h, self.img_w), mode="bilinear", align_corners=False)
        d = self.dec0(d)  # (B, 32, 180, 360)
        return d

    # ------------------------------------------------------------------
    def forward(self, images: list, targets: list = None):
        """
        images  : list of (4, T, F) tensors
        targets : list of target dicts (same format as baseline EnergySegDataset)
        """
        x = torch.stack(images, dim=0)   # (B, 4, T, F)

        e0, e1, e2, e3 = self._encode(x)
        d = self._decode(e0, e1, e2, e3)

        energy_maps = self.energy_head(d)     # (B, N_CLASSES, 180, 360)
        mask_maps   = self.mask_head(d)       # (B, N_CLASSES, 180, 360)
        dist_pred   = self.distance_head(e3)  # (B, N_CLASSES)

        if self.training:
            assert targets is not None
            return self._compute_loss(energy_maps, mask_maps, dist_pred, targets, x.device)
        else:
            return self._build_detections(energy_maps, mask_maps, dist_pred)

    # ------------------------------------------------------------------
    def _compute_loss(self, energy_maps, mask_maps, dist_pred, targets, device):
        B = energy_maps.shape[0]
        loss_energy = energy_maps.new_zeros(1).squeeze()
        loss_mask   = energy_maps.new_zeros(1).squeeze()
        loss_dist   = energy_maps.new_zeros(1).squeeze()
        n_instances = 0

        for i, tgt in enumerate(targets):
            boxes    = tgt["boxes"].to(device)          # (N, 4)
            labels   = tgt["labels"].to(device)         # (N,) 1-indexed
            emaps    = tgt["energy_maps"].to(device)    # (N, H, W)
            emasks   = tgt["energy_masks"].to(device)   # (N, H, W) bool
            vmap     = tgt["vmap"].to(device)           # (H, W)
            vmask    = tgt["vmask"].to(device)          # (H, W) bool
            dists    = tgt["distances"].to(device)      # (N,)

            # ── Full-image energy map loss (Pearson proxy via MSE) ────────
            pred_energy_i = torch.sigmoid(energy_maps[i])   # (N_CLASSES, H, W)
            # Weighted MSE: annotated pixels get extra weight
            W = torch.ones(self.img_h, self.img_w, device=device)
            W[vmask] = self.energy_annot_w

            # Sum over classes: compare pred max-pool across classes vs vmap
            pred_combined = pred_energy_i.max(dim=0).values  # (H, W)
            loss_energy = loss_energy + (W * (pred_combined - vmap).pow(2)).mean()

            # ── Per-instance mask + energy loss ──────────────────────────
            N = len(labels)
            if N > 0:
                for n in range(N):
                    cls_idx = int(labels[n].item())   # 1-indexed, 0 = background
                    if cls_idx < 1 or cls_idx >= self.n_classes:
                        continue

                    pred_mask_nc  = mask_maps[i, cls_idx]   # (H, W)
                    pred_mask_prob = torch.sigmoid(pred_mask_nc)
                    pred_energy_nc = torch.sigmoid(energy_maps[i, cls_idx]) # (H, W)

                    gt_mask = emasks[n].float()    # (H, W)
                    gt_emap = emaps[n]             # (H, W)

                    # Dice loss
                    inter  = (pred_mask_prob * gt_mask).sum()
                    denom  = pred_mask_prob.sum() + gt_mask.sum() + 1e-6
                    dice   = 1.0 - 2.0 * inter / denom

                    # BCE loss with class weight
                    cw     = self.class_weights[cls_idx]
                    bce    = F.binary_cross_entropy_with_logits(
                        pred_mask_nc, gt_mask, reduction="mean"
                    ) * cw

                    # Energy MSE on annotated pixels
                    ann_px = gt_mask > 0.5
                    if ann_px.any():
                        e_mse = F.mse_loss(pred_energy_nc[ann_px], gt_emap[ann_px])
                    else:
                        e_mse = pred_energy_nc.new_zeros(1).squeeze()

                    loss_mask = loss_mask + dice + bce + e_mse
                    n_instances += 1

                # ── Distance loss ─────────────────────────────────────────
                # Use class-averaged GT distance per class present in this frame
                for cls_idx_raw in labels.unique():
                    cls_idx = int(cls_idx_raw.item())
                    if cls_idx < 1 or cls_idx >= self.n_classes:
                        continue
                    mask_c   = labels == cls_idx_raw
                    gt_d     = dists[mask_c].mean()
                    pr_d     = dist_pred[i, cls_idx]
                    loss_dist = loss_dist + F.huber_loss(pr_d, gt_d, delta=0.5)

        denom = max(B, 1)
        return {
            "loss_energy": loss_energy / denom,
            "loss_mask":   loss_mask   / max(n_instances, 1),
            "loss_distance": self.dist_w * loss_dist / denom,
        }

    # ------------------------------------------------------------------
    def _build_detections(self, energy_maps, mask_maps, dist_pred):
        """
        Returns list of per-image dicts compatible with InstanceTracker and
        the submission JSON format.
        """
        B = energy_maps.shape[0]
        detections = []
        for i in range(B):
            em  = energy_maps[i].detach().cpu()   # (N_CLASSES, H, W)
            mm  = mask_maps[i].detach().cpu()     # (N_CLASSES, H, W)
            dp  = dist_pred[i].detach().cpu()     # (N_CLASSES,)

            boxes_list, labels_list, scores_list = [], [], []
            emaps_list, dists_list = [], []

            for cls_idx in range(1, self.n_classes):  # skip background
                # mask_c   = mm[cls_idx]    # (H, W)
                # energy_c = em[cls_idx]    # (H, W)

                mask_c   = torch.sigmoid(mm[cls_idx])
                energy_c = torch.sigmoid(em[cls_idx])

                score = float(mask_c.max())

                if score < 1e-3:
                    continue

                # Derive bounding box from mask threshold
                active = mask_c > 0.3
                if not active.any():
                    continue
                rows = torch.where(active.any(dim=1))[0]
                cols = torch.where(active.any(dim=0))[0]
                x0, x1 = float(cols[0]),  float(cols[-1] + 1)
                y0, y1 = float(rows[0]),  float(rows[-1] + 1)

                boxes_list.append([x0, y0, x1, y1])
                labels_list.append(cls_idx)
                scores_list.append(score)
                emaps_list.append(energy_c.unsqueeze(0))  # (1, H, W) — use full map
                dists_list.append(float(dp[cls_idx]))

            if boxes_list:
                detections.append({
                    "boxes":       torch.tensor(boxes_list,  dtype=torch.float32),
                    "labels":      torch.tensor(labels_list, dtype=torch.long),
                    "scores":      torch.tensor(scores_list, dtype=torch.float32),
                    "energy_maps": torch.cat(emaps_list, dim=0),   # (N_det, H, W)
                    "dist_pred":   torch.tensor(dists_list, dtype=torch.float32),
                    "full_energy": em.max(dim=0).values,           # (H, W)
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


# ════════════════════════════════════════════════════════════════════════════
# 4.  DATASET UTILITIES  (reused verbatim from baseline — re-exported here)
# ════════════════════════════════════════════════════════════════════════════

import os
import json
import glob
import re
import hashlib
import pickle
import sqlite3
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset


def json_to_seq_name(json_path: str) -> str:
    stem = os.path.splitext(os.path.basename(json_path))[0]
    return re.sub(r"_std$", "", stem, flags=re.IGNORECASE)


def frame_path(seq_dir, seq_name, idx):
    return os.path.join(seq_dir, f"{seq_name}_{idx:04d}.png")


def scan_available_frames(seq_dir, seq_name):
    pattern = os.path.join(seq_dir, f"{seq_name}_*.png")
    out = []
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"_(\d{4})\.png$", os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def get_sequence_infos(split_keyword, labels_base, frames_base):
    infos = []
    if not os.path.isdir(labels_base) or not os.path.isdir(frames_base):
        return infos
    for split_dir in os.listdir(labels_base):
        if split_keyword not in split_dir:
            continue
        split_path = os.path.join(labels_base, split_dir)
        if not os.path.isdir(split_path):
            continue
        for json_file in sorted(glob.glob(os.path.join(split_path, "*.json"))):
            seq_name = json_to_seq_name(json_file)
            seq_dir  = os.path.join(frames_base, split_dir, seq_name)
            infos.append((json_file, seq_dir, seq_name))
    return infos


_DB_SCHEMA_VERSION = "v4_unet"


def _dataset_cache_fingerprint(sequence_infos):
    h = hashlib.md5()
    h.update(_DB_SCHEMA_VERSION.encode())
    for json_path, seq_dir, seq_name in sorted(sequence_infos):
        mtime = str(os.path.getmtime(json_path)) if os.path.exists(json_path) else "missing"
        h.update(f"{json_path}:{mtime}:{seq_dir}:{seq_name}".encode())
    return h.hexdigest()[:16]


class EnergySegDataset(Dataset):
    CACHE_DIR = ".dataset_cache"

    def __init__(
        self,
        sequence_infos,
        frames_base,
        mic_base,
        acoustic_extractor,
        frames_per_epoch=None,
        img_w=360,
        img_h=180,
        dist_norm=500.0,
        cache_max_size=200,
        augmentor=None,
    ):
        self.frames_base        = frames_base
        self.mic_base           = mic_base
        self.acoustic_extractor = acoustic_extractor
        self.frames_per_epoch   = frames_per_epoch
        self.img_w              = img_w
        self.img_h              = img_h
        self.dist_norm          = dist_norm
        self.cache_max_size     = cache_max_size
        self.augmentor          = augmentor

        self.frame_cache    = OrderedDict()
        self.seq_map        = {}
        self.all_samples    = []
        self.current_indices= []
        self.class_to_sample_indices = defaultdict(list)
        self.db_conn        = None

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        fingerprint  = _dataset_cache_fingerprint(sequence_infos)
        self.db_path = os.path.join(self.CACHE_DIR, f"annotations_{fingerprint}.db")

        if os.path.exists(self.db_path):
            print(f"[Dataset] Cache hit  — loading index from {self.db_path}")
            self._load_index_from_db()
        else:
            print(f"[Dataset] Cache miss — building SQLite database from JSONs...")
            self._build_db(sequence_infos)

        print(f"[Dataset] frames_per_epoch={frames_per_epoch} → "
              f"{'subsampling' if frames_per_epoch else 'using all ' + str(len(self.all_samples))}")
        self.reset_epoch()

    def _get_db_conn(self):
        if self.db_conn is None:
            import pathlib
            db_uri       = pathlib.Path(self.db_path).absolute().as_uri()
            self.db_conn = sqlite3.connect(f"{db_uri}?mode=ro", uri=True)
        return self.db_conn

    def _build_db(self, sequence_infos):
        from tqdm import tqdm
        conn = sqlite3.connect(self.db_path)
        c    = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS annots (seq_id INTEGER, frame_idx INTEGER, data BLOB)")
        c.execute("CREATE TABLE IF NOT EXISTS seqs   (seq_id INTEGER, seq_dir TEXT, seq_name TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS frame_classes (sample_idx INTEGER, category_id INTEGER)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_lookup ON annots (seq_id, frame_idx)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_fc     ON frame_classes (category_id)")

        seq_id = 0
        for json_path, seq_dir, seq_name in tqdm(sequence_infos, desc="[Dataset] Building SQLite DB", leave=False):
            c.execute("INSERT INTO seqs VALUES (?, ?, ?)", (seq_id, seq_dir, seq_name))
            self.seq_map[seq_id] = (seq_dir, seq_name)

            with open(json_path) as f:
                data = json.load(f)

            by_frame = defaultdict(list)
            for ann in data.get("annotations", []):
                by_frame[int(ann["metadata_frame_index"])].append(ann)

            all_ids = sorted(set(by_frame.keys()) | set(scan_available_frames(seq_dir, seq_name)))
            for fi in all_ids:
                anns       = by_frame.get(fi, [])
                sample_idx = len(self.all_samples)
                self.all_samples.append((seq_id, fi))
                c.execute("INSERT INTO annots VALUES (?, ?, ?)", (seq_id, fi, pickle.dumps(anns)))
                for ann in anns:
                    cat_id = int(ann["category_id"])
                    c.execute("INSERT INTO frame_classes VALUES (?, ?)", (sample_idx, cat_id))
                    self.class_to_sample_indices[cat_id].append(sample_idx)
            seq_id += 1

        conn.commit()
        conn.close()

    def _load_index_from_db(self):
        from tqdm import tqdm
        conn = sqlite3.connect(self.db_path)
        c    = conn.cursor()
        for row in c.execute("SELECT seq_id, seq_dir, seq_name FROM seqs"):
            self.seq_map[row[0]] = (row[1], row[2])
        c.execute("SELECT COUNT(*) FROM annots")
        total = c.fetchone()[0]
        c.execute("SELECT seq_id, frame_idx FROM annots")
        for row in tqdm(c, total=total, desc="[Dataset] Loading Index", leave=False):
            self.all_samples.append((row[0], row[1]))
        for row in c.execute("SELECT sample_idx, category_id FROM frame_classes"):
            self.class_to_sample_indices[row[1]].append(row[0])
        if self.class_to_sample_indices:
            n_pairs = sum(len(v) for v in self.class_to_sample_indices.values())
            print(f"[Dataset] Class index: {len(self.class_to_sample_indices)} classes, {n_pairs} pairs")
        conn.close()

    def clear_caches(self):
        self.frame_cache.clear()
        if self.acoustic_extractor is not None:
            self.acoustic_extractor.clear_cache()

    def __del__(self):
        if self.db_conn is not None:
            self.db_conn.close()

    def load_frame_tensor(self, seq_dir, seq_name, frame_idx):
        key = (seq_name, frame_idx)
        if key in self.frame_cache:
            t = self.frame_cache.pop(key)
            self.frame_cache[key] = t
            return t.clone()

        wav_path = wav_path_from_seq_dir(seq_dir, self.frames_base, self.mic_base)
        tensor   = self.acoustic_extractor.get_frame_bands(wav_path, frame_idx)

        if self.cache_max_size > 0:
            self.frame_cache[key] = tensor
            if len(self.frame_cache) > self.cache_max_size:
                self.frame_cache.popitem(last=False)
        return tensor.clone()

    def build_annotation_target(self, frame_annots):
        boxes, labels       = [], []
        bin_masks           = []
        energy_maps_list    = []
        energy_masks_list   = []
        distances, iids     = [], []

        combined_energy = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        combined_mask   = np.zeros((self.img_h, self.img_w), dtype=bool)

        for ann in frame_annots:
            cat  = int(ann["category_id"]) + 1
            dist = float(ann["distance"])
            iid  = int(ann["instance_id"])

            energy_map  = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            energy_mask = np.zeros((self.img_h, self.img_w), dtype=bool)
            xs, ys      = [], []

            for sub in ann["segmentation"]:
                for triplet in sub:
                    x, y, v = float(triplet[0]), float(triplet[1]), float(triplet[2])
                    xi, yi  = int(round(x)), int(round(y))
                    if 0 <= xi < self.img_w and 0 <= yi < self.img_h:
                        energy_map[yi, xi]      = float(v)
                        energy_mask[yi, xi]     = True
                        combined_energy[yi, xi] = max(combined_energy[yi, xi], float(v))
                        combined_mask[yi, xi]   = True
                        xs.append(xi); ys.append(yi)

            if not xs:
                continue
            x0, x1 = min(xs), max(xs) + 1
            y0, y1 = min(ys), max(ys) + 1
            boxes.append([float(x0), float(y0), float(x1), float(y1)])
            labels.append(cat)
            bin_masks.append(energy_mask.copy())
            energy_maps_list.append(energy_map.copy())
            energy_masks_list.append(energy_mask.copy())
            distances.append(dist / self.dist_norm)
            iids.append(iid)

        if not boxes:
            N = 0
            return dict(
                boxes        = torch.zeros(N, 4,              dtype=torch.float32),
                labels       = torch.zeros(N,                 dtype=torch.int64),
                masks        = torch.zeros(N, self.img_h, self.img_w, dtype=torch.bool),
                energy_maps  = torch.zeros(N, self.img_h, self.img_w, dtype=torch.float32),
                energy_masks = torch.zeros(N, self.img_h, self.img_w, dtype=torch.bool),
                vmap         = torch.zeros(self.img_h, self.img_w,    dtype=torch.float32),
                vmask        = torch.zeros(self.img_h, self.img_w,    dtype=torch.bool),
                distances    = torch.zeros(N,                 dtype=torch.float32),
                instance_ids = torch.zeros(N,                 dtype=torch.int64),
            )

        return dict(
            boxes        = torch.tensor(boxes,                          dtype=torch.float32),
            labels       = torch.tensor(labels,                         dtype=torch.int64),
            masks        = torch.tensor(np.stack(bin_masks),            dtype=torch.bool),
            energy_maps  = torch.tensor(np.stack(energy_maps_list),     dtype=torch.float32),
            energy_masks = torch.tensor(np.stack(energy_masks_list),    dtype=torch.bool),
            vmap         = torch.tensor(combined_energy,                dtype=torch.float32),
            vmask        = torch.tensor(combined_mask,                  dtype=torch.bool),
            distances    = torch.tensor(distances,                      dtype=torch.float32),
            instance_ids = torch.tensor(iids,                           dtype=torch.int64),
        )

    def reset_epoch(self, balanced=True):
        total = len(self.all_samples)
        n     = min(self.frames_per_epoch, total) if self.frames_per_epoch else total
        use_balanced = balanced and bool(self.class_to_sample_indices)

        if not use_balanced:
            idx = np.arange(total)
            np.random.shuffle(idx)
            self.current_indices = idx[:n]
            return

        classes    = list(self.class_to_sample_indices.keys())
        n_cls      = len(classes)
        min_quota  = max(1, n // n_cls // 2)
        balanced_part = []
        for cls in classes:
            pool  = self.class_to_sample_indices[cls]
            quota = max(int(len(pool) / total * n), min_quota)
            choices = np.random.choice(pool, quota, replace=(len(pool) < quota))
            balanced_part.extend(choices)

        balanced_part  = np.array(balanced_part, dtype=np.int64)
        unique_balanced = np.unique(balanced_part)

        if len(balanced_part) >= n:
            np.random.shuffle(balanced_part)
            self.current_indices = balanced_part[:n]
        else:
            remaining = np.setdiff1d(np.arange(total), unique_balanced)
            n_rem     = n - len(balanced_part)
            if n_rem > 0 and len(remaining) > 0:
                extra = np.random.choice(remaining, n_rem, replace=(len(remaining) < n_rem))
                self.current_indices = np.random.permutation(np.concatenate([balanced_part, extra]))
            else:
                self.current_indices = np.random.permutation(balanced_part)

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        real_idx          = self.current_indices[idx]
        seq_id, fi        = self.all_samples[real_idx]
        seq_dir, seq_name = self.seq_map[seq_id]

        c = self._get_db_conn().cursor()
        c.execute("SELECT data FROM annots WHERE seq_id=? AND frame_idx=?", (seq_id, fi))
        row  = c.fetchone()
        anns = pickle.loads(row[0]) if row and row[0] else []

        image  = self.load_frame_tensor(seq_dir, seq_name, fi)
        target = self.build_annotation_target(anns)

        if self.augmentor is not None:
            image, target = self.augmentor(image, target)

        return image, target


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    if info is not None:
        ds = info.dataset
        if hasattr(ds, "frame_cache"):
            ds.frame_cache.clear()


# ════════════════════════════════════════════════════════════════════════════
# 5.  INSTANCE TRACKER  (reused from baseline, adapted for U-Net output)
# ════════════════════════════════════════════════════════════════════════════

from torchvision.ops import box_iou


class InstanceTracker:
    def __init__(self, iou_thr=0.3, max_age=5, coast_decay=0.9,
                 img_w=360, img_h=180):
        self.iou_thr     = iou_thr
        self.max_age     = max_age
        self.coast_decay = coast_decay
        self.img_w       = img_w
        self.img_h       = img_h
        self._nxt        = 0
        self.tracks: dict = {}

    def reset(self):
        self._nxt   = 0
        self.tracks = {}

    def update(self, det: dict) -> list:
        boxes  = det.get("boxes",       torch.zeros(0, 4))
        labels = det.get("labels",      torch.zeros(0, dtype=torch.long))
        scores = det.get("scores",      torch.zeros(0))
        emaps  = det.get("energy_maps", torch.zeros(0, self.img_h, self.img_w))
        dists  = det.get("dist_pred",   torch.zeros(0))
        full_e = det.get("full_energy", torch.zeros(self.img_h, self.img_w))
        N      = len(boxes)

        def _to_result(tid, coasting):
            trk = self.tracks[tid]
            sc  = trk.get("last_score", 0.0)
            if coasting:
                sc *= self.coast_decay
                trk["last_score"] = sc
            return dict(
                track_id    = tid,
                label       = trk["label"],
                score       = sc,
                box         = trk["box"].tolist(),
                energy_map  = trk["energy_map"],
                dist_pred   = trk["dist_pred"],
                full_energy = trk["full_energy"],
                coasting    = coasting,
            )

        if N == 0:
            results = []
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
                else:
                    results.append(_to_result(tid, coasting=True))
            return results

        track_ids   = list(self.tracks.keys())
        track_boxes = (torch.stack([self.tracks[t]["box"] for t in track_ids])
                       if track_ids else torch.zeros(0, 4))
        track_labels = np.array([self.tracks[t]["label"] for t in track_ids])

        iou_mat = (box_iou(track_boxes.cpu(), boxes.cpu()).numpy()
                   if track_boxes.numel() > 0 else np.zeros((0, N)))

        matched_det, matched_trk = {}, set()
        if iou_mat.size > 0:
            cost = 1.0 - iou_mat
            for r in range(len(track_ids)):
                for c in range(N):
                    if track_labels[r] != int(labels[c]):
                        cost[r, c] = 1000.0
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_thr and cost[r, c] < 1000.0:
                    tid = track_ids[r]
                    self.tracks[tid].update(
                        box         = boxes[c].cpu(),
                        age         = 0,
                        label       = int(labels[c]),
                        energy_map  = emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(self.img_h, self.img_w),
                        dist_pred   = float(dists[c]) if c < len(dists) else 0.0,
                        full_energy = full_e,
                        last_score  = float(scores[c]),
                    )
                    self.tracks[tid]["hits"] += 1
                    matched_det[c] = tid
                    matched_trk.add(r)

        for c in range(N):
            if c not in matched_det:
                tid = self._nxt; self._nxt += 1
                self.tracks[tid] = dict(
                    box        = boxes[c].cpu(), age=0, hits=1,
                    label      = int(labels[c]),
                    energy_map = emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(self.img_h, self.img_w),
                    dist_pred  = float(dists[c]) if c < len(dists) else 0.0,
                    full_energy = full_e,
                    last_score = float(scores[c]),
                )
                matched_det[c] = tid

        for r, tid in enumerate(track_ids):
            if r not in matched_trk:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        results = []
        for c in range(N):
            tid = matched_det[c]
            r   = _to_result(tid, coasting=False)
            r["score"] = float(scores[c])
            results.append(r)
        for r_idx, tid in enumerate(track_ids):
            if r_idx not in matched_trk and tid in self.tracks:
                results.append(_to_result(tid, coasting=True))
        return results