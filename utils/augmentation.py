"""
augmentation.py
---------------
Augmentation pipeline for DCASE 2026 Task 3 Track A (audio-only).

All augmentations operate on:
  image  : Tensor (N_CH, T, F)  — SALSA-Lite features
  target : dict with boxes, labels, masks, energy_maps, energy_masks,
           vmap, vmask, distances, instance_ids

Augmentations
~~~~~~~~~~~~~
1.  Acoustic Channel Swap (ACS)      — negate NIPV channels (left-right mirror)
2.  Azimuth rotation                 — roll vmap/vmask/masks along W axis;
                                       shift box x-coords accordingly
3.  Elevation jitter                 — shift vmap/masks/boxes along H axis
4.  Intensity scale                  — scale all energy values by a random scalar
5.  Acoustic noise                   — additive Gaussian noise on SALSA features
6.  Feature band masking             — zero out random frequency bands
7.  Mixup (feature space)            — blend two frames' features and targets
"""

import random
import math
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roll_target_azimuth(target: dict, shift: int, img_w: int) -> dict:
    """Roll all spatial target tensors along the azimuth (W) axis."""
    t = {}
    for k, v in target.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[-1] == img_w:
            t[k] = torch.roll(v, shift, dims=-1)
        else:
            t[k] = v

    if len(target["boxes"]) > 0:
        boxes = t["boxes"].clone()
        boxes[:, 0] = (boxes[:, 0] + shift) % img_w
        boxes[:, 2] = (boxes[:, 2] + shift) % img_w
        # Clamp to valid range (wrap can invert x0/x1 for boxes crossing boundary)
        # For simplicity, take the union back to [0, img_w]
        boxes[:, 0] = boxes[:, 0].clamp(0, img_w - 1)
        boxes[:, 2] = boxes[:, 2].clamp(1, img_w)
        t["boxes"] = boxes

    return t


def _shift_target_elevation(target: dict, shift: int, img_h: int) -> dict:
    """Roll all spatial target tensors along the elevation (H) axis (reflect, not wrap)."""
    t = {}
    for k, v in target.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[-2] == img_h:
            # Use reflect-style: clamp at boundaries rather than wrap
            t[k] = torch.roll(v, shift, dims=-2)
        else:
            t[k] = v

    if len(target["boxes"]) > 0:
        boxes = t["boxes"].clone()
        boxes[:, 1] = (boxes[:, 1] + shift).clamp(0, img_h - 1)
        boxes[:, 3] = (boxes[:, 3] + shift).clamp(1, img_h)
        # Drop boxes that ended up degenerate
        valid = boxes[:, 3] > boxes[:, 1]
        if not valid.all():
            idx = torch.where(valid)[0]
            boxes = boxes[idx]
            for field in ("labels", "masks", "energy_maps", "energy_masks",
                          "distances", "instance_ids"):
                if field in t and len(t[field]) > 0:
                    t[field] = t[field][idx]
        t["boxes"] = boxes

    return t


# ---------------------------------------------------------------------------
# Main augmentor
# ---------------------------------------------------------------------------

class SeldAugmentor:
    """
    Parameters
    ----------
    img_h, img_w      : canvas size
    n_acoustic        : number of SALSA channels (should be 4)
    azimuth_rotate    : enable random azimuth roll
    hflip_prob        : probability of acoustic channel swap (ACS left-right mirror)
    elev_jitter_px    : ±elevation jitter in pixels (0 = disabled)
    elev_jitter_prob  : probability of applying elevation jitter
    max_bands_masked  : max number of frequency bands to zero out
    intensity_scale_range : (lo, hi) uniform range for energy scaling
    acoustic_noise_std    : std of Gaussian noise added to SALSA features
    mixup_alpha       : Beta distribution alpha for mixup (0 = disabled)
    """

    def __init__(
        self,
        img_h:                  int   = 180,
        img_w:                  int   = 360,
        n_acoustic:             int   = 4,
        azimuth_rotate:         bool  = True,
        hflip_prob:             float = 0.5,
        elev_jitter_px:         int   = 10,
        elev_jitter_prob:       float = 0.5,
        max_bands_masked:       int   = 4,
        intensity_scale_range:  tuple = (0.6, 1.4),
        acoustic_noise_std:     float = 0.03,
        mixup_alpha:            float = 0.0,    # set >0 to enable
    ):
        self.img_h               = img_h
        self.img_w               = img_w
        self.n_acoustic          = n_acoustic
        self.azimuth_rotate      = azimuth_rotate
        self.hflip_prob          = hflip_prob
        self.elev_jitter_px      = elev_jitter_px
        self.elev_jitter_prob    = elev_jitter_prob
        self.max_bands_masked    = max_bands_masked
        self.intensity_scale_range = intensity_scale_range
        self.acoustic_noise_std  = acoustic_noise_std
        self.mixup_alpha         = mixup_alpha

    # ------------------------------------------------------------------
    def __call__(
        self,
        image:   torch.Tensor,
        target:  dict,
        image2:  torch.Tensor = None,
        target2: dict         = None,
    ):
        """
        Apply augmentation pipeline.

        image  : (N_CH, T, F)
        target : annotation dict
        image2, target2 : optional second sample for mixup
        """

        # 1. Acoustic Channel Swap (left-right spatial mirror)
        if random.random() < self.hflip_prob:
            image = self._acs(image)
            target = self._flip_target_lr(target)

        # 2. Azimuth rotation (roll spatial maps)
        if self.azimuth_rotate:
            shift   = random.randint(0, self.img_w - 1)
            target  = _roll_target_azimuth(target, shift, self.img_w)
            # SALSA features are spatial in F-axis only implicitly; azimuth rotation
            # acts on the output canvas, not the input spectrogram.

        # 3. Elevation jitter
        if self.elev_jitter_px > 0 and random.random() < self.elev_jitter_prob:
            shift  = random.randint(-self.elev_jitter_px, self.elev_jitter_px)
            target = _shift_target_elevation(target, shift, self.img_h)

        # 4. Intensity / energy scale
        lo, hi = self.intensity_scale_range
        scale  = random.uniform(lo, hi)
        target = self._scale_energy(target, scale)

        # 5. Acoustic noise on SALSA features
        if self.acoustic_noise_std > 0:
            noise = torch.randn_like(image) * self.acoustic_noise_std
            image = image + noise

        # 6. Frequency band masking on SALSA features
        if self.max_bands_masked > 0:
            image = self._band_mask(image)

        # 7. Mixup
        if self.mixup_alpha > 0 and image2 is not None and target2 is not None:
            lam   = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            image  = lam * image + (1 - lam) * image2
            target = self._mixup_targets(target, target2, lam)

        return image, target

    # ------------------------------------------------------------------
    @staticmethod
    def _acs(image: torch.Tensor) -> torch.Tensor:
        """Acoustic Channel Swap: negate all NIPV channels (ch 1,2,3)."""
        out = image.clone()
        out[1:] = -out[1:]
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _flip_target_lr(target: dict) -> dict:
        """Mirror spatial target tensors left-right (for ACS)."""
        t = {}
        for k, v in target.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                t[k] = v.flip(-1)
            else:
                t[k] = v
        if len(target["boxes"]) > 0:
            img_w  = target["vmap"].shape[-1]
            boxes  = t["boxes"].clone()
            x0_new = img_w - target["boxes"][:, 2]
            x1_new = img_w - target["boxes"][:, 0]
            boxes[:, 0] = x0_new.clamp(0, img_w - 1)
            boxes[:, 2] = x1_new.clamp(1, img_w)
            t["boxes"] = boxes
        return t

    # ------------------------------------------------------------------
    @staticmethod
    def _scale_energy(target: dict, scale: float) -> dict:
        t = dict(target)
        for k in ("energy_maps", "vmap"):
            if k in t:
                t[k] = (t[k] * scale).clamp(0.0, 1.0)
        return t

    # ------------------------------------------------------------------
    def _band_mask(self, image: torch.Tensor) -> torch.Tensor:
        """Zero out up to max_bands_masked random frequency bands."""
        n_bands = random.randint(0, self.max_bands_masked)
        if n_bands == 0:
            return image
        out   = image.clone()
        F_dim = image.shape[-1]
        for _ in range(n_bands):
            bw    = random.randint(1, max(1, F_dim // 16))
            start = random.randint(0, F_dim - bw)
            out[:, :, start : start + bw] = 0.0
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _mixup_targets(t1: dict, t2: dict, lam: float) -> dict:
        """Simple mixup: blend vmaps and keep both instance lists."""
        t = dict(t1)
        # Blend full-image energy maps
        for k in ("vmap",):
            if k in t1 and k in t2:
                t[k] = lam * t1[k] + (1 - lam) * t2[k]
        # Concatenate instance-level targets (model sees both)
        for k in ("boxes", "labels", "masks", "energy_maps",
                  "energy_masks", "distances", "instance_ids"):
            if k in t1 and k in t2 and len(t1[k]) > 0 and len(t2[k]) > 0:
                try:
                    t[k] = torch.cat([t1[k], t2[k]], dim=0)
                except Exception:
                    t[k] = t1[k]
        # Combine vmasks
        if "vmask" in t1 and "vmask" in t2:
            t["vmask"] = t1["vmask"] | t2["vmask"]
        return t