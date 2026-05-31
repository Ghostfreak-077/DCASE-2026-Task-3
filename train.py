"""
train.py
--------
Training script for UNetSAISELD — DCASE 2026 Task 3 Track A.

Usage
~~~~~
    python train.py --exp_name my_unet_run

Key differences from baseline train.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Model      : UNetSAISELD  (replaces EnergyInstanceModel)
* Features   : SalsaFeatureExtractor  (replaces AcousticFeatureExtractor)
* Augmentor  : SeldAugmentor with ACS + elevation jitter  (replaces SeldAugmentor)
* Optimizer  : Two-group schedule — encoder (lower LR) + heads (higher LR)
* Unfreezing : NOT backbone-layer-based; instead progressive encoder thaw at Ep 3
* Loss keys  : loss_energy, loss_mask, loss_distance  (no RPN losses)
"""

import os
import sys
import time
import math
import random
import argparse
import shutil
import warnings
from collections import defaultdict
from contextlib import contextmanager

import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acoustic_features import SalsaFeatureExtractor
from augmentation import SeldAugmentor
from model import (
    get_sequence_infos,
    EnergySegDataset,
    collate_fn,
    worker_init_fn,
    UNetSAISELD,
)

# ════════════════════════════════════════════════════════════════════════════
# 0.  UTILITIES
# ════════════════════════════════════════════════════════════════════════════

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
    else:                   batch_size = 8

    pin = torch.cuda.is_available()

    print(f"\n[TUNING] BATCH_SIZE  = {batch_size}")
    print(f"[TUNING] NUM_WORKERS = {num_workers}")
    print(f"[TUNING] CACHE_MAX   = {cache_per_ds} frames per dataset")
    print(f"[TUNING] RAM         = {total_ram_gb:.1f} GB")
    print(f"[TUNING] pin_memory  = {pin}\n")

    return batch_size, num_workers, cache_per_ds, pin


# ════════════════════════════════════════════════════════════════════════════
# 2.  CONFIG
# ════════════════════════════════════════════════════════════════════════════

FRAMES_BASE = "/teamspace/studios/this_studio/data/"
LABELS_BASE = "/teamspace/studios/this_studio/data/labels_dev"
MIC_BASE    = "/teamspace/studios/this_studio/data/foa_dev"

IMG_W, IMG_H = 360, 180
NUM_CLASSES  = 14      # 13 sound classes + background
NUM_EPOCHS   = 100

TRAIN_FRAMES_PER_EPOCH = None
VAL_FRAMES_PER_EPOCH   = 15

DIST_NORM        = 500.0
ENERGY_ANNOT_W   = 5.0
DIST_W           = 15.0    # elevated from baseline (was ~1)
PATIENCE         = 100
SCHEDULER_PATIENCE = 3

# Learning rates for the two-group schedule
LR_ENCODER = 5e-5   # lower — encoder starts partially frozen
LR_HEADS   = 5e-4   # higher — heads train aggressively

# Epoch at which the full encoder is unfrozen
ENCODER_UNFREEZE_EPOCH = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] Device: {DEVICE}  [{torch.cuda.get_device_name(0)}]")
else:
    print(f"[INFO] Device: {DEVICE}")


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


# ════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train(train_infos, val_infos, exp_dir):
    best_model_path = os.path.join(exp_dir, "unet_saiseld_best.pth")

    batch_size, num_workers, cache_max, pin_memory = detect_resources()
    print(f"[TRAIN] NUM_CLASSES = {NUM_CLASSES}")

    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[TRAIN] Mixed precision (AMP) ENABLED")

    # ── Acoustic feature extractor ───────────────────────────────────────
    acoustic_extractor = SalsaFeatureExtractor(
        fs              = 24000,
        n_fft           = 512,
        hop_length      = 300,
        fmin_doa        = 40.0,
        fmax_doa        = 6000.0,
        fmax_spec       = 6000.0,
        ref_mic         = 0,
        context_frames  = 4,         # ±2 frame window for temporal context
        audio_cache_size = 32,
        frame_cache_size = cache_max,
    )
    print(f"[SALSA] Feature shape: {acoustic_extractor.feature_shape}  "
          f"(channels × time × freq)")

    # ── Augmentor ────────────────────────────────────────────────────────
    train_augmentor = SeldAugmentor(
        img_h                = IMG_H,
        img_w                = IMG_W,
        n_acoustic           = acoustic_extractor.n_channels,
        azimuth_rotate       = True,
        hflip_prob           = 0.5,       # ACS probability
        elev_jitter_px       = 10,
        elev_jitter_prob     = 0.5,
        max_bands_masked     = 4,
        intensity_scale_range= (0.6, 1.4),
        acoustic_noise_std   = 0.03,
        mixup_alpha          = 0.0,       # disabled; enable with 0.4 if needed
    )
    print(f"[TRAIN] Augmentation: ACS(p=0.5) | azimuth_rotate | "
          f"elev_jitter(±10px,p=0.5) | band_mask(max=4) | noise(σ=0.03)")

    # ── Datasets ─────────────────────────────────────────────────────────
    train_dataset = EnergySegDataset(
        sequence_infos     = train_infos,
        frames_base        = FRAMES_BASE,
        mic_base           = MIC_BASE,
        acoustic_extractor = acoustic_extractor,
        frames_per_epoch   = TRAIN_FRAMES_PER_EPOCH,
        img_w=IMG_W, img_h=IMG_H,
        dist_norm          = DIST_NORM,
        cache_max_size     = cache_max,
        augmentor          = train_augmentor,
    )

    # One-sample overfit test
    train_dataset.current_indices = np.array([0])

    # Class weights (same smoothed-sqrt approach as baseline)
    print("[TRAIN] Calculating class weights…")
    ann_counts = {k: len(v) for k, v in train_dataset.class_to_sample_indices.items()}
    total_anns = max(sum(ann_counts.values()), 1)
    n_fg       = NUM_CLASSES - 1
    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=DEVICE)
    for cat_id, count in ann_counts.items():
        mid = cat_id + 1
        if count > 0:
            raw = total_anns / (n_fg * count)
            class_weights[mid] = min(float(math.sqrt(raw)), 10.0)
    class_weights[0] = 1.0
    print(f"[TRAIN] Class Weights: {class_weights.cpu().numpy().round(3)}")

    val_dataset = EnergySegDataset(
        sequence_infos     = val_infos,
        frames_base        = FRAMES_BASE,
        mic_base           = MIC_BASE,
        acoustic_extractor = acoustic_extractor,
        frames_per_epoch   = VAL_FRAMES_PER_EPOCH,
        img_w=IMG_W, img_h=IMG_H,
        dist_norm          = DIST_NORM,
        cache_max_size     = max(1, cache_max // 4),
        augmentor          = None,
    )
    # Fix val indices to be deterministic
    val_dataset.current_indices = np.arange(
        min(VAL_FRAMES_PER_EPOCH, len(val_dataset.all_samples))
    )

    loader_kw = dict(
        collate_fn         = collate_fn,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        prefetch_factor    = 3 if num_workers > 0 else None,
        worker_init_fn     = worker_init_fn if num_workers > 0 else None,
        persistent_workers = False,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size*2,   shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size*2, shuffle=False, **loader_kw)

    # ── Model ────────────────────────────────────────────────────────────
    model = UNetSAISELD(
        n_classes      = NUM_CLASSES,
        in_ch          = acoustic_extractor.n_channels,
        img_h          = IMG_H,
        img_w          = IMG_W,
        energy_annot_w = ENERGY_ANNOT_W,
        dist_w         = DIST_W,
        class_weights  = class_weights,
    ).to(DEVICE)

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[MODEL] UNetSAISELD — {n_total:.1f}M parameters")

    # ── Optimizer — two-group: encoder vs decoder+heads ──────────────────
    encoder_params = (
        list(model.enc0.parameters()) +
        list(model.enc1.parameters()) +
        list(model.enc2.parameters()) +
        list(model.enc3.parameters())
    )
    head_params = (
        # list(model.bridge_conv.parameters()) +
        # list(model.dec3.parameters()) +
        list(model.dec2.parameters()) +
        list(model.dec1.parameters()) +
        list(model.dec0.parameters()) +
        list(model.energy_head.parameters()) +
        list(model.mask_head.parameters()) +
        list(model.distance_head.parameters())
    )

    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": LR_ENCODER, "weight_decay": 1e-4, "name": "encoder"},
        {"params": head_params,    "lr": LR_HEADS,   "weight_decay": 1e-3, "name": "heads"},
    ])
    initial_lrs = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}

    print(f"\n[TRAIN] Optimizer Groups:")
    for pg in optimizer.param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  - {pg['name']:<10}: lr={pg['lr']:.1e} | wd={pg['weight_decay']:.1e} | params={n_p:,}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=SCHEDULER_PATIENCE, min_lr=1e-7,
    )

    # ── Training state ────────────────────────────────────────────────────
    history           = defaultdict(list)
    t0                = time.time()
    best_loss         = float("inf")
    phase_best_loss   = float("inf")
    epochs_no_improve = 0
    prev_phase, _, _  = apply_encoder_freeze(model, 1)

    def _to_float(v):
        return v.item() if isinstance(v, torch.Tensor) else float(v or 0)

    print(f"\n[TRAIN] Max {NUM_EPOCHS} epochs × {TRAIN_FRAMES_PER_EPOCH} frames | "
          f"batch={batch_size} | workers={num_workers} | device={DEVICE}")

    for epoch in range(1, NUM_EPOCHS + 1):

        # train_dataset.reset_epoch(balanced=True)
        # One-sample overfit test
        train_dataset.current_indices = np.array([0])

        # ── Freeze schedule ──────────────────────────────────────────────
        phase, n_train, n_total_p = apply_encoder_freeze(model, epoch)

        if phase != prev_phase:
            print(f"\n[FREEZE] Phase transition: '{prev_phase}' → '{phase}'")
            print(f"         ({n_train/1e6:.1f}M / {n_total_p/1e6:.1f}M params active)")

            ckpt_path = os.path.join(exp_dir, f"phase_boundary_ep{epoch-1}.pth")
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, ckpt_path)
            else:
                torch.save(model.state_dict(), ckpt_path)
            print(f"[FREEZE] Boundary checkpoint → {ckpt_path}")

            # Hard reset patience + LRs
            epochs_no_improve  = 0
            phase_best_loss    = float("inf")
            scheduler.num_bad_epochs = 0
            scheduler.best     = float("inf")
            for pg in optimizer.param_groups:
                pg["lr"] = initial_lrs[pg["name"]]
            print(f"[FREEZE] Hard reset: LRs restored, patience cleared.")
            prev_phase = phase

        # ── TRAIN ─────────────────────────────────────────────────────────
        model.train()
        ep_losses = defaultdict(float)
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Train]",
                    leave=False, file=sys.__stdout__, dynamic_ncols=True)

        for images, targets in pbar:
            images  = [img.to(DEVICE, non_blocking=pin_memory).float() for img in images]
            targets = [
                {k: v.to(DEVICE, non_blocking=pin_memory) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in targets
            ]

            optimizer.zero_grad(set_to_none=True)


            # print(images[0].shape)
            # print(images[0].min())
            # print(images[0].max())
            # print(images[0].mean())


            # feat = images[0]

            # for c in range(feat.shape[0]):
            #     print(
            #         c,
            #         feat[c].min().item(),
            #         feat[c].max().item(),
            #         feat[c].mean().item()
            #     )

            # for c in range(targets[0]['energy_maps'].shape[0]):
            #     print(
            #         targets[0]['labels'],
            #         targets[0]['energy_maps'].max().item(),
            #         targets[0]['energy_maps'].sum().item()
            #     )

            # state = torch.load('./experiments/debug_20260531_180952/unet_saiseld_best.pth', map_location=DEVICE, weights_only=True)
            # if isinstance(state, dict):
            #     if   "model_state"      in state: state = state["model_state"]
            #     elif "model_state_dict" in state: state = state["model_state_dict"]
            # missing, unexpected = model.load_state_dict(state, strict=True)
            # if missing:    print(f"[WARN] Missing keys    : {missing}")
            # if unexpected: print(f"[WARN] Unexpected keys : {unexpected}")

            # plt.imshow(targets[0]['energy_maps'][0].cpu())
            # plt.savefig('img.png')

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = model(images, targets, epoch=epoch)
                total     = sum(loss_dict.values())

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()

            ep_losses["total"] += total.item()
            for k, v in loss_dict.items():
                ep_losses[k] += _to_float(v)
            n_batches += 1

            pbar.set_postfix({
                "Tot":  f"{ep_losses['total']/n_batches:.3f}",
                "E":    f"{ep_losses.get('loss_energy',0)/n_batches:.3f}",
                "M":    f"{ep_losses.get('loss_mask',0)/n_batches:.3f}",
                "D":    f"{ep_losses.get('loss_distance',0)/n_batches:.3f}",
            })

        for k, v in ep_losses.items():
            history[k].append(v / max(n_batches, 1))

        # ── VALIDATION ────────────────────────────────────────────────────
        val_ep_losses = defaultdict(float)
        val_n         = 0

        if len(val_loader) > 0:
            pbar_v = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Val]",
                          leave=False, file=sys.__stdout__, dynamic_ncols=True)
            with torch.no_grad(), eval_behavior_for_loss(model):
                for images, targets in pbar_v:
                    images  = [img.to(DEVICE, non_blocking=pin_memory).float() for img in images]
                    targets = [
                        {k: v.to(DEVICE, non_blocking=pin_memory) if isinstance(v, torch.Tensor) else v
                         for k, v in t.items()}
                        for t in targets
                    ]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        loss_dict = model(images, targets, epoch=epoch)
                        val_tot   = sum(loss_dict.values())
                    val_ep_losses["total"] += val_tot.item()
                    for k, v in loss_dict.items():
                        val_ep_losses[k] += _to_float(v)
                    val_n += 1

            for k, v in val_ep_losses.items():
                history[f"val_{k}"].append(v / max(val_n, 1))

        # ── Epoch summary ─────────────────────────────────────────────────
        t_elapsed = time.time() - t0
        print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch}/{NUM_EPOCHS} "
              f"({t_elapsed:.1f}s) | phase={phase}")

        train_tot = history["total"][-1]
        print(f"  [Train] Total={train_tot:.4f}  |  ", end="")
        for k in sorted(ep_losses):
            if k != "total":
                print(f"{k}={history[k][-1]:.4f}  ", end="")

        val_avg = float("inf")
        if val_n > 0:
            val_avg = history["val_total"][-1]
            print(f"  [Val]   Total={val_avg:.4f}  |  ", end="")
            for k in sorted(val_ep_losses):
                if k != "total":
                    print(f"{k}={history[f'val_{k}'][-1]:.4f}  ", end="")
            print()

        enc_lr  = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]
        print(f"  [LR] encoder={enc_lr:.2e} | heads={head_lr:.2e} | "
              f"scheduler_bad={scheduler.num_bad_epochs}")

        # ── Scheduling + early stopping ───────────────────────────────────
        track = val_avg if val_n > 0 else train_tot
        scheduler.step(track)

        if track < best_loss:
            best_loss = track
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] New best: {best_loss:.4f} → {best_model_path}")

        if track < phase_best_loss:
            phase_best_loss   = track
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  [!] No phase improvement {epochs_no_improve}/{PATIENCE} "
                  f"(phase best={phase_best_loss:.4f})")
            if epochs_no_improve >= PATIENCE:
                print(f"\n[!] Early stopping after epoch {epoch}.")
                break

    print(f"\n[TRAIN] Finished in {time.time()-t0:.1f}s")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    return model, dict(history)


# ════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Train UNetSAISELD — Track A")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name (creates experiments/<exp_name>_<timestamp>/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir   = os.path.join(os.getcwd(), "experiments", f"{args.exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(exp_dir, "train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(exp_dir, "error.log"), sys.stderr)

    print(f"[MAIN] Starting experiment : {args.exp_name}")
    print(f"[MAIN] Output directory    : {exp_dir}")

    set_seed(args.seed)

    train_infos = get_sequence_infos("train", LABELS_BASE, FRAMES_BASE)
    val_infos   = get_sequence_infos("test",  LABELS_BASE, FRAMES_BASE)

    if not train_infos:
        raise FileNotFoundError(f"No train sequences found in {LABELS_BASE}")
    if not val_infos:
        print(f"[WARN] No val sequences found in {LABELS_BASE}")

    print(f"\n[MAIN] {len(train_infos)} training sequences | {len(val_infos)} val sequences")

    model, history = train(train_infos, val_infos, exp_dir)
    plot_losses(history, os.path.join(exp_dir, "training_loss.png"))

    print("\n[DONE]")
    print(f"  All artifacts saved to: {exp_dir}")