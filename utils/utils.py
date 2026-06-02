from contextlib import contextmanager
import math
import os
import random
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.model import UNetSAISELD, box_iou, linear_sum_assignment


# Epoch at which the full encoder is unfrozen
ENCODER_UNFREEZE_EPOCH = 3

class InstanceTracker:
    def __init__(self, iou_thr=0.3, max_age=5, coast_decay=0.9, img_w=360, img_h=180):
        self.iou_thr     = iou_thr
        self.max_age     = max_age
        self.coast_decay = coast_decay
        self.img_w       = img_w
        self.img_h       = img_h
        self._nxt        = 0
        self.tracks: dict = {}

    def reset(self):
        self._nxt = 0; self.tracks = {}

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
            return dict(track_id=tid, label=trk["label"], score=sc,
                        box=trk["box"].tolist(), energy_map=trk["energy_map"],
                        dist_pred=trk["dist_pred"], full_energy=trk["full_energy"],
                        coasting=coasting)

        if N == 0:
            results = []
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
                else:
                    results.append(_to_result(tid, coasting=True))
            return results

        track_ids    = list(self.tracks.keys())
        track_boxes  = (torch.stack([self.tracks[t]["box"] for t in track_ids])
                        if track_ids else torch.zeros(0, 4))
        track_labels = np.array([self.tracks[t]["label"] for t in track_ids])
        iou_mat      = (box_iou(track_boxes.cpu(), boxes.cpu()).numpy()
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
                        box        = boxes[c].cpu(), age=0, label=int(labels[c]),
                        energy_map = emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(self.img_h, self.img_w),
                        dist_pred  = float(dists[c]) if c < len(dists) else 0.0,
                        full_energy = full_e, last_score=float(scores[c]),
                    )
                    self.tracks[tid]["hits"] += 1
                    matched_det[c] = tid; matched_trk.add(r)

        for c in range(N):
            if c not in matched_det:
                tid = self._nxt; self._nxt += 1
                self.tracks[tid] = dict(
                    box=boxes[c].cpu(), age=0, hits=1, label=int(labels[c]),
                    energy_map=emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(self.img_h, self.img_w),
                    dist_pred=float(dists[c]) if c < len(dists) else 0.0,
                    full_energy=full_e, last_score=float(scores[c]),
                )
                matched_det[c] = tid

        for r, tid in enumerate(track_ids):
            if r not in matched_trk:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        results = []
        for c in range(N):
            tid = matched_det[c]; r = _to_result(tid, coasting=False)
            r["score"] = float(scores[c]); results.append(r)
        for r_idx, tid in enumerate(track_ids):
            if r_idx not in matched_trk and tid in self.tracks:
                results.append(_to_result(tid, coasting=True))
        return results


class Logger:
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log      = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ════════════════════════════════════════════════════════════════════════════
# 1.  AUTO-DETECT RESOURCES
# ════════════════════════════════════════════════════════════════════════════

def detect_resources():
    import psutil
    total_ram_gb = psutil.virtual_memory().total / 1e9
    cpu_count    = os.cpu_count() or 1
    gpu_vram_gb  = 0.0

    if torch.cuda.is_available():
        props       = torch.cuda.get_device_properties(0)
        gpu_vram_gb = props.total_memory / 1e9

    num_workers  = min(max(cpu_count - 1, 0), 8)
    cache_per_ds = min(512, max(100, int(total_ram_gb * 6)))

    # UNetSAISELD is lighter than Mask R-CNN; can fit larger batches
    if   gpu_vram_gb >= 40: batch_size = 48
    elif gpu_vram_gb >= 24: batch_size = 24
    elif gpu_vram_gb >= 16: batch_size = 16
    else:                   batch_size = 32

    pin = torch.cuda.is_available()

    print(f"\n[TUNING] BATCH_SIZE  = {batch_size}")
    print(f"[TUNING] NUM_WORKERS = {num_workers}")
    print(f"[TUNING] CACHE_MAX   = {cache_per_ds} frames per dataset")
    print(f"[TUNING] RAM         = {total_ram_gb:.1f} GB")
    print(f"[TUNING] pin_memory  = {pin}\n")

    return batch_size, num_workers, cache_per_ds, pin


# ════════════════════════════════════════════════════════════════════════════
# 3.  PROGRESSIVE ENCODER FREEZE
# ════════════════════════════════════════════════════════════════════════════

def apply_encoder_freeze(model: UNetSAISELD, epoch: int) -> str:
    """
    Epoch 1-2 : enc3 + bridge + decoder + heads  (encoder enc0-2 frozen)
    Epoch 3+  : full model trainable
    """
    if epoch < ENCODER_UNFREEZE_EPOCH:
        # Freeze enc0, enc1, enc2
        for module in (model.enc0, model.enc1, model.enc2):
            for p in module.parameters():
                p.requires_grad = False
        for module in (model.enc3,
                    #     model.bridge_pool, model.bridge_conv,
                    #    model.skip_pool3, model.skip_pool2, model.skip_pool1,
                    #    model.dec3, model.dec2, model.dec1, model.dec0,
                       model.energy_head, model.mask_head, model.distance_head):
            for p in module.parameters():
                p.requires_grad = True
        phase = "enc3+bridge+decoder+heads"
    else:
        for p in model.parameters():
            p.requires_grad = True
        phase = "full_model"

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return phase, n_train, n_total


# ════════════════════════════════════════════════════════════════════════════
# 4.  PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_losses(history: dict, save_path: str):
    base_keys = [k for k in history if not k.startswith("val_") and k != "total"]
    all_keys  = ["total"] + sorted(base_keys)
    epochs    = range(1, len(history.get("total", [])) + 1)
    n_plots   = len(all_keys)
    cols, rows = 4, math.ceil(n_plots / 4)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes      = np.array(axes).flatten()
    fig.suptitle("UNetSAISELD — Training & Validation Loss", fontsize=14, y=1.01)

    def smooth(v, k=7):
        if len(v) < k:
            return v, 0
        k = max(3, min(k, (len(v) // 3) * 2 + 1) | 1)
        return np.convolve(v, np.ones(k) / k, "valid"), k // 2

    for i, key in enumerate(all_keys):
        ax    = axes[i]
        vals  = history.get(key, [])
        title = "Total" if key == "total" else key.replace("loss_", "").replace("_", " ").title()
        ax.plot(list(epochs), vals, lw=1, color="steelblue", alpha=0.4, label="train")
        if len(vals) >= 5:
            s, pad = smooth(vals)
            ax.plot(list(range(pad + 1, len(vals) - pad + 1)), s, lw=2, color="orangered", label="train (sm)")
        val_key = "val_total" if key == "total" else f"val_{key}"
        if val_key in history and len(history[val_key]) == len(epochs):
            ax.plot(list(epochs), history[val_key], lw=2, color="green", label="val")
        ax.legend(fontsize=7); ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Loss curves → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# 5.  EVAL CONTEXT MANAGER
# ════════════════════════════════════════════════════════════════════════════

@contextmanager
def eval_behavior_for_loss(model):
    target_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout, nn.Dropout2d)
    modules      = [m for m in model.modules() if isinstance(m, target_types)]
    orig         = {m: m.training for m in modules}
    for m in modules:
        m.eval()
    try:
        yield
    finally:
        for m, s in orig.items():
            m.train(mode=s)
