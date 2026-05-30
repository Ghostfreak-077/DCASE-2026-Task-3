import argparse
import math
import os
import random
import shutil
import sys
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import SeldAugmentor
from data.energy_seg_dataset import EnergySegDataset, collate_fn, get_sequence_infos, worker_init_fn
from data.pipeline import SalsaFeatureExtractor
from models.unet_saiseld import UNetSAISELD

FRAMES_BASE = "/teamspace/studios/this_studio/data/"
LABELS_BASE = "/teamspace/studios/this_studio/data/labels_dev"
MIC_BASE = "/teamspace/studios/this_studio/data/foa_dev"

IMG_W, IMG_H = 360, 180
NUM_CLASSES = 14      
NUM_EPOCHS = 10
TRAIN_FRAMES_PER_EPOCH = 15
VAL_FRAMES_PER_EPOCH = 15
DIST_NORM = 500.0
ENERGY_ANNOT_W = 5.0
DIST_W = 15.0    
PATIENCE = 15
SCHEDULER_PATIENCE = 3
LR_ENCODER = 5e-5   
LR_HEADS = 5e-4   
ENCODER_UNFREEZE_EPOCH = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger:
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def detect_resources():
    import psutil
    total_ram_gb = psutil.virtual_memory().total / 1e9
    cpu_count = os.cpu_count() or 1
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    num_workers = min(max(cpu_count - 1, 0), 8)
    cache_per_ds = min(512, max(100, int(total_ram_gb * 6)))
    if gpu_vram_gb >= 40: batch_size = 48
    elif gpu_vram_gb >= 24: batch_size = 24
    elif gpu_vram_gb >= 16: batch_size = 16
    else: batch_size = 8
    return batch_size, num_workers, cache_per_ds, torch.cuda.is_available()

def apply_encoder_freeze(model, epoch):
    if epoch < ENCODER_UNFREEZE_EPOCH:
        for module in (model.enc0, model.enc1, model.enc2):
            for p in module.parameters(): p.requires_grad = False
        for module in (model.enc3, model.energy_head, model.mask_head, model.distance_head):
            for p in module.parameters(): p.requires_grad = True
        phase = "enc3+bridge+decoder+heads"
    else:
        for p in model.parameters(): p.requires_grad = True
        phase = "full_model"
    return phase, sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

@contextmanager
def eval_behavior_for_loss(model):
    target_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout, nn.Dropout2d)
    modules = [m for m in model.modules() if isinstance(m, target_types)]
    orig = {m: m.training for m in modules}
    for m in modules: m.eval()
    try: yield
    finally:
        for m, s in orig.items(): m.train(mode=s)

def plot_losses(history, save_path):
    base_keys = [k for k in history if not k.startswith("val_") and k != "total"]
    all_keys = ["total"] + sorted(base_keys)
    epochs = range(1, len(history.get("total", [])) + 1)
    cols, rows = 4, math.ceil(len(all_keys) / 4)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for i, key in enumerate(all_keys):
        ax = axes[i]
        vals = history.get(key, [])
        title = "Total" if key == "total" else key.replace("loss_", "").replace("_", " ").title()
        ax.plot(list(epochs), vals, lw=1, color="steelblue", alpha=0.4, label="train")
        if len(vals) >= 5:
            smooth = np.convolve(vals, np.ones(3)/3, "valid")
            ax.plot(list(range(2, len(vals))), smooth, lw=2, color="orangered", label="train (sm)")
        val_key = "val_total" if key == "total" else f"val_{key}"
        if val_key in history and len(history[val_key]) == len(epochs):
            ax.plot(list(epochs), history[val_key], lw=2, color="green", label="val")
        ax.legend(fontsize=7); ax.set_title(title, fontsize=10, fontweight="bold"); ax.grid(alpha=0.3)
    for j in range(i + 1, len(axes)): axes[j].axis("off")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def train_pipeline(train_infos, val_infos, exp_dir):
    best_model_path = os.path.join(exp_dir, "unet_saiseld_best.pth")
    batch_size, num_workers, cache_max, pin_memory = detect_resources()
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    acoustic_extractor = SalsaFeatureExtractor(fs=24000, n_fft=512, hop_length=300, audio_cache_size=32, frame_cache_size=cache_max)
    train_augmentor = SeldAugmentor(img_h=IMG_H, img_w=IMG_W, n_acoustic=acoustic_extractor.n_channels)
    train_dataset = EnergySegDataset(train_infos, FRAMES_BASE, MIC_BASE, acoustic_extractor, frames_per_epoch=TRAIN_FRAMES_PER_EPOCH, augmentor=train_augmentor)

    ann_counts = {k: len(v) for k, v in train_dataset.class_to_sample_indices.items()}
    total_anns = max(sum(ann_counts.values()), 1)
    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=DEVICE)
    for cat_id, count in ann_counts.items():
        if count > 0: class_weights[cat_id + 1] = min(float(math.sqrt(total_anns / ((NUM_CLASSES - 1) * count))), 10.0)

    val_dataset = EnergySegDataset(val_infos, FRAMES_BASE, MIC_BASE, acoustic_extractor, frames_per_epoch=VAL_FRAMES_PER_EPOCH, augmentor=None)
    val_dataset.current_indices = np.arange(min(VAL_FRAMES_PER_EPOCH, len(val_dataset.all_samples)))

    loader_kw = dict(collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=3 if num_workers > 0 else None, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, **loader_kw)

    model = UNetSAISELD(n_classes=NUM_CLASSES, in_ch=acoustic_extractor.n_channels, img_h=IMG_H, img_w=IMG_W, energy_annot_w=ENERGY_ANNOT_W, dist_w=DIST_W, class_weights=class_weights).to(DEVICE)
    encoder_params = list(model.enc0.parameters()) + list(model.enc1.parameters()) + list(model.enc2.parameters()) + list(model.enc3.parameters())
    head_params = list(model.dec2.parameters()) + list(model.dec1.parameters()) + list(model.dec0.parameters()) + list(model.energy_head.parameters()) + list(model.mask_head.parameters()) + list(model.distance_head.parameters())

    optimizer = optim.AdamW([{"params": encoder_params, "lr": LR_ENCODER, "weight_decay": 1e-4, "name": "encoder"}, {"params": head_params, "lr": LR_HEADS, "weight_decay": 1e-3, "name": "heads"}])
    initial_lrs = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE)

    history = defaultdict(list)
    best_loss = phase_best_loss = float("inf"); epochs_no_improve = 0; prev_phase, _, _ = apply_encoder_freeze(model, 1)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_dataset.reset_epoch(balanced=True)
        phase, n_train, n_total_p = apply_encoder_freeze(model, epoch)
        if phase != prev_phase:
            shutil.copy2(best_model_path, os.path.join(exp_dir, f"phase_boundary_ep{epoch-1}.pth")) if os.path.exists(best_model_path) else None
            epochs_no_improve = 0; phase_best_loss = float("inf"); scheduler.num_bad_epochs = 0; scheduler.best = float("inf")
            for pg in optimizer.param_groups: pg["lr"] = initial_lrs[pg["name"]]
            prev_phase = phase

        model.train(); ep_losses = defaultdict(float); n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Train]", file=sys.__stdout__)
        for images, targets in pbar:
            images = [img.to(DEVICE, non_blocking=pin_memory).float() for img in images]
            targets = [{k: v.to(DEVICE, non_blocking=pin_memory) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = model(images, targets, epoch=epoch)
                total = sum(loss_dict.values())
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); scaler.step(optimizer); scaler.update()
            ep_losses["total"] += total.item()
            for k, v in loss_dict.items(): ep_losses[k] += v.item() if torch.is_tensor(v) else float(v)
            n_batches += 1
        for k, v in ep_losses.items(): history[k].append(v / max(n_batches, 1))

        val_ep_losses = defaultdict(float); val_n = 0
        if len(val_loader) > 0:
            with torch.no_grad(), eval_behavior_for_loss(model):
                for images, targets in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Val]", file=sys.__stdout__):
                    images = [img.to(DEVICE, non_blocking=pin_memory).float() for img in images]
                    targets = [{k: v.to(DEVICE, non_blocking=pin_memory) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        loss_dict = model(images, targets, epoch=epoch)
                        val_ep_losses["total"] += sum(loss_dict.values()).item()
                    for k, v in loss_dict.items(): val_ep_losses[k] += v.item() if torch.is_tensor(v) else float(v)
                    val_n += 1
            for k, v in val_ep_losses.items(): history[f"val_{k}"].append(v / max(val_n, 1))

        track = history["val_total"][-1] if val_n > 0 else history["total"][-1]
        scheduler.step(track)
        if track < best_loss:
            best_loss = track; torch.save(model.state_dict(), best_model_path)
            print(f"   [*] Registered top validation checkpoint: {best_loss:.4f}")
        if track < phase_best_loss: phase_best_loss = track; epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE: break
    if os.path.exists(best_model_path): model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    return model, dict(history)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    exp_dir = os.path.join(os.getcwd(), "experiments", f"{args.exp_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(exp_dir, exist_ok=True); sys.stdout = Logger(os.path.join(exp_dir, "train.log"), sys.stdout); set_seed(args.seed)
    train_infos = get_sequence_infos("train", LABELS_BASE, FRAMES_BASE)
    val_infos = get_sequence_infos("test", LABELS_BASE, FRAMES_BASE)
    model, history = train_pipeline(train_infos, val_infos, exp_dir)
    plot_losses(history, os.path.join(exp_dir, "training_loss.png"))