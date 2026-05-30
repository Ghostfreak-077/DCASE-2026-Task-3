# models/unet_saiseld.py
"""Asymmetric-striding U-Net for DCASE 2026 Task 3 Track A."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Padding & Custom 2D Convolution Blocks
# ==============================================================================

def circ_pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Apply circular padding on W (azimuth) and reflection padding on H (elevation)."""
    x = F.pad(x, (pad, pad, 0, 0), mode="circular")
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    return x


class CircConv2d(nn.Module):
    """Convolution wrapper executing joint circular-W + reflect-H padding."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=(1, 1), bias=False):
        super().__init__()
        self.pad = kernel // 2
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=self.stride, padding=0, bias=bias)

    def forward(self, x):
        if self.pad > 0:
            x = circ_pad2d(x, self.pad)
        return self.conv(x)


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            CircConv2d(in_ch, out_ch, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Two-layer convolution block with structural asymmetric striding downsampling."""
    def __init__(self, in_ch, out_ch, stride=(1, 1), dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch, stride=stride),
            ConvBnRelu(out_ch, out_ch, stride=(1, 1)),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel gating attention module."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(1, ch // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.shape[0], -1, 1, 1)


class DecoderBlock(nn.Module):
    """Bilinear upsampling operator merging SE-gated connection tensors."""
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.1):
        super().__init__()
        self.se = SEBlock(skip_ch)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x, skip, target_size):
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        skip = self.se(skip)
        return self.conv(torch.cat([x, skip], dim=1))


# ==============================================================================
# Model Output Heads
# ==============================================================================

class EnergyMapHead(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            ConvBnRelu(in_ch, in_ch),
            CircConv2d(in_ch, n_classes, kernel=1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x): return self.head(x)


class InstanceMaskHead(nn.Module):
    def __init__(self, in_ch, n_classes, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(
            ConvBnRelu(in_ch, in_ch),
            nn.Dropout2d(dropout),
            CircConv2d(in_ch, n_classes, kernel=1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x): return self.head(x)


class DistanceHead(nn.Module):
    def __init__(self, bottleneck_ch, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
    def forward(self, x):
        return F.softplus(self.mlp(self.pool(x).flatten(1)))


# ==============================================================================
# Main UNetSAISELD Neural Network Module
# ==============================================================================

class UNetSAISELD(nn.Module):
    """U-Net mapping multi-channel SALSA spectrogram windows to high-res canvases."""
    def __init__(self, n_classes: int = 14, in_ch: int = 4, img_h: int = 180, img_w: int = 360, energy_annot_w: float = 5.0, dist_w: float = 15.0, dropout: float = 0.1, class_weights=None):
        super().__init__()
        self.n_classes = n_classes
        self.img_h = img_h
        self.img_w = img_w
        self.energy_annot_w = energy_annot_w
        self.dist_w = dist_w

        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.ones(n_classes))

        # Encoder (Asymmetric Striding: preserves T=8, shrinks F)
        self.enc0 = EncoderBlock(in_ch, 64, stride=(1, 1), dropout=dropout)
        self.enc1 = EncoderBlock(64, 128, stride=(1, 2), dropout=dropout)
        self.enc2 = EncoderBlock(128, 256, stride=(1, 2), dropout=dropout)
        self.enc3 = EncoderBlock(256, 512, stride=(1, 2), dropout=dropout)

        # Decoder 
        self.dec2 = DecoderBlock(512, 256, 256, dropout=dropout)
        self.dec1 = DecoderBlock(256, 128, 128, dropout=dropout)
        self.dec0 = DecoderBlock(128, 64, 64, dropout=dropout)
        self.final = nn.Sequential(ConvBnRelu(64, 32), ConvBnRelu(32, 32))

        # Spatial resolution adapter layers for skip connections
        self.skip2 = nn.AdaptiveAvgPool2d((16, 32))
        self.skip1 = nn.AdaptiveAvgPool2d((32, 64))
        self.skip0 = nn.AdaptiveAvgPool2d((64, 128))

        self.energy_head = EnergyMapHead(32, n_classes)
        self.mask_head = InstanceMaskHead(32, n_classes, dropout=0.2)
        self.distance_head = DistanceHead(512, n_classes, hidden=256, dropout=0.3)

    def forward(self, images: list, targets: list = None, epoch: int = 99):
        x = torch.stack(images, dim=0)
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d = self.dec2(e3, self.skip2(e2), (16, 32))
        d = self.dec1(d, self.skip1(e1), (32, 64))
        d = self.dec0(d, self.skip0(e0), (64, 128))
        d = F.interpolate(d, size=(self.img_h, self.img_w), mode="bilinear", align_corners=False)
        d = self.final(d)

        energy_maps = self.energy_head(d)
        mask_maps = self.mask_head(d)
        dist_pred = self.distance_head(e3)

        if self.training:
            return self._compute_loss(energy_maps, mask_maps, dist_pred, targets, x.device, epoch)
        return self._build_detections(energy_maps, mask_maps, dist_pred)

    @staticmethod
    def _focal_bce(pred, gt, gamma=2.0, alpha=0.25):
        bce = F.binary_cross_entropy(pred, gt, reduction="none")
        pt = torch.where(gt > 0.5, pred, 1.0 - pred)
        focal = ((1.0 - pt) ** gamma) * bce
        a_t = torch.where(gt > 0.5, torch.full_like(gt, alpha), torch.full_like(gt, 1.0 - alpha))
        return (a_t * focal).mean()

    def _compute_loss(self, energy_maps, mask_maps, dist_pred, targets, device, epoch):
        B = energy_maps.shape[0]
        loss_energy = energy_maps.new_zeros(1).squeeze()
        loss_mask = energy_maps.new_zeros(1).squeeze()
        loss_dist = energy_maps.new_zeros(1).squeeze()
        n_instances = n_dist_terms = 0

        dist_ramp = 0.0 if epoch <= 2 else min(1.0, (epoch - 2) / 3.0) * self.dist_w

        for i, tgt in enumerate(targets):
            labels = tgt["labels"].to(device)
            emaps, emasks = tgt["energy_maps"].to(device), tgt["energy_masks"].to(device)
            vmap, vmask, dists = tgt["vmap"].to(device), tgt["vmask"].to(device), tgt["distances"].to(device)

            pred_combined = energy_maps[i].max(dim=0).values
            W = torch.ones(self.img_h, self.img_w, device=device)
            W[vmask] = self.energy_annot_w
            loss_energy = loss_energy + (W * (pred_combined - vmap).pow(2)).mean()

            if len(labels) > 0:
                for n in range(len(labels)):
                    cls_idx = int(labels[n].item())
                    if cls_idx < 1 or cls_idx >= self.n_classes: continue
                    
                    focal = self._focal_bce(mask_maps[i, cls_idx], emasks[n].float()) * self.class_weights[cls_idx]
                    inter = (mask_maps[i, cls_idx] * emasks[n]).sum()
                    dice = 1.0 - 2.0 * inter / (mask_maps[i, cls_idx].sum() + emasks[n].sum() + 1e-6)
                    
                    ann_px = emasks[n] > 0.5
                    e_mse = F.mse_loss(energy_maps[i, cls_idx][ann_px], emaps[n][ann_px]) if ann_px.any() else energy_maps.new_zeros(1).squeeze()
                    
                    loss_mask = loss_mask + focal + dice + e_mse
                    n_instances += 1

                if dist_ramp > 0:
                    for cls_raw in labels.unique():
                        cls_idx = int(cls_raw.item())
                        if cls_idx < 1 or cls_idx >= self.n_classes: continue
                        loss_dist = loss_dist + F.huber_loss(dist_pred[i, cls_idx], dists[labels == cls_raw].mean(), delta=0.5)
                        n_dist_terms += 1

        return {"loss_energy": loss_energy / max(B, 1), "loss_mask": loss_mask / max(n_instances, 1), "loss_distance": dist_ramp * loss_dist / max(n_dist_terms, 1)}

    def _build_detections(self, energy_maps, mask_maps, dist_pred):
        B = energy_maps.shape[0]
        detections = []
        for i in range(B):
            em, dp = energy_maps[i].detach().cpu(), dist_pred[i].detach().cpu()
            boxes, labels, scores, emaps, dists = [], [], [], [], []

            for cls_idx in range(1, self.n_classes):
                energy_c = em[cls_idx]
                score = float(energy_c.max())
                if score < 0.15: continue

                active = energy_c >= (0.10 * score)
                if not active.any(): continue
                rows, cols = torch.where(active.any(dim=1))[0], torch.where(active.any(dim=0))[0]
                
                boxes.append([float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)])
                labels.append(cls_idx)
                scores.append(score)
                emaps.append(energy_c.unsqueeze(0))
                dists.append(float(dp[cls_idx]))

            if boxes:
                detections.append({"boxes": torch.tensor(boxes), "labels": torch.tensor(labels), "scores": torch.tensor(scores), "energy_maps": torch.cat(emaps, dim=0), "dist_pred": torch.tensor(dists), "full_energy": em.max(dim=0).values})
            else:
                detections.append({"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0), "energy_maps": torch.zeros(0, self.img_h, self.img_w), "dist_pred": torch.zeros(0), "full_energy": em.max(dim=0).values})
        return detections