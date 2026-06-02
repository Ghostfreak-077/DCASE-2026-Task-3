# ============================================================
#  ENERGY-FIELD INSTANCE SEGMENTATION  —  INFERENCE SCRIPT
#  run_inference.py  (UNetSAISELD / Path B1 edition)
#
#  Merge notes
#  -----------
#  Taken from mentor's run_inference__3_.py:
#    - TemporalTracker (dataclass Track, min_age/confirmed pattern, no coast_decay)
#    - numpy box_iou + nms_per_class  (no torchvision dependency)
#    - Two-pass inference with adaptive per-class score gating
#      (collects all sequence scores → stretches to [0.05,0.95] → thresholds
#       on SCALED score, submits RAW score to preserve mAP ranking)
#    - Spatial fragment merging (clusters same-class tracks within 50px,
#       emits one merged annotation per cluster)
#    - Per-class detection funnel printout
#
#  Kept from our version:
#    - SalsaFeatureExtractor  (no UpLAM, no RGB frames)
#    - UNetSAISELD model loading
#    - (4, T, F) audio-only InferenceDataset
#    - sparsify_energy_map for full (H, W) energy maps (not 28×28)
#    - exp_dir structure, Logger, JSON-based sequence discovery
#    - score_thr default 0.15 (energy-calibrated, not sigmoid-raw)
#
#  Conflicts resolved:
#    - extract_peaks (28×28 EVAL space) → replaced by sparsify_energy_map
#    - tracker empty-update placeholder shape (0,28,28) → (0,H,W)
#    - InferenceDataset stays SALSA-only; worker init uses SalsaFeatureExtractor
# ============================================================

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Resolve script directory ──────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from acoustic_features import SalsaFeatureExtractor, wav_path_from_seq_dir
from model import (
    UNetSAISELD,
    get_sequence_infos,
    scan_available_frames,
)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

FRAMES_BASE = "/teamspace/studios/this_studio/data/"
LABELS_BASE = "/teamspace/studios/this_studio/gaussian_dataset/labels_dev"
MIC_BASE    = "/teamspace/studios/this_studio/data/foa_dev"

IMG_W, IMG_H = 360, 180
N_CHANNELS   = 4        # SALSA-Lite: 1 log-mel + 3 NIPV
NUM_CLASSES  = 14       # 0=background, 1-13=foreground
DIST_NORM    = 500.0
INFERENCE_HZ = 10

# Model label (1-indexed) → submission category_id (0-indexed)
MODEL_TO_CAT = {k: k - 1 for k in range(1, NUM_CLASSES)}

# SALSA hyperparams — must match train.py
SALSA_FS             = 24000
SALSA_N_FFT          = 512
SALSA_HOP            = 300
SALSA_FMIN_DOA       = 40.0
SALSA_FMAX_DOA       = 6000.0
SALSA_FMAX_SPEC      = 6000.0
SALSA_CONTEXT_FRAMES = 4

if torch.cuda.is_available():
    DEVICE    = torch.device("cuda")
    _gpu_name = torch.cuda.get_device_name(0)
    torch.backends.cudnn.benchmark = True
else:
    DEVICE    = torch.device("cpu")
    _gpu_name = "CPU"

_CPU_COUNT  = os.cpu_count() or 1
NUM_WORKERS = min(max(_CPU_COUNT - 1, 0), 16)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE DATASET  (audio-only, SALSA-Lite features)
# ══════════════════════════════════════════════════════════════════════════════

class InferenceDataset(Dataset):
    def __init__(self, seq_dir: str, seq_name: str, frame_indices: list):
        self.seq_dir       = seq_dir
        self.seq_name      = seq_name
        self.frame_indices = frame_indices
        self.wav_path      = wav_path_from_seq_dir(seq_dir, FRAMES_BASE, MIC_BASE)
        self._extractor    = None   # lazy-init per worker

    def _ensure_extractor(self):
        if self._extractor is None:
            self._extractor = SalsaFeatureExtractor(
                fs               = SALSA_FS,
                n_fft            = SALSA_N_FFT,
                hop_length       = SALSA_HOP,
                fmin_doa         = SALSA_FMIN_DOA,
                fmax_doa         = SALSA_FMAX_DOA,
                fmax_spec        = SALSA_FMAX_SPEC,
                ref_mic          = 0,
                context_frames   = SALSA_CONTEXT_FRAMES,
                audio_cache_size = 4,
                frame_cache_size = 128,
            )

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        self._ensure_extractor()
        fi     = self.frame_indices[idx]
        tensor = self._extractor.get_frame_bands(self.wav_path, fi)
        # tensor: (4, T, F) float32
        return fi, tensor


def _worker_init(worker_id):
    ds = torch.utils.data.get_worker_info().dataset
    ds._extractor = SalsaFeatureExtractor(
        fs=SALSA_FS, n_fft=SALSA_N_FFT, hop_length=SALSA_HOP,
        fmin_doa=SALSA_FMIN_DOA, fmax_doa=SALSA_FMAX_DOA, fmax_spec=SALSA_FMAX_SPEC,
        ref_mic=0, context_frames=SALSA_CONTEXT_FRAMES,
        audio_cache_size=4, frame_cache_size=128,
    )


def _collate(b):
    return [x[0] for x in b], [x[1] for x in b]


def build_loader(seq_dir, seq_name, frame_indices, batch_size, num_workers):
    ds = InferenceDataset(seq_dir=seq_dir, seq_name=seq_name,
                          frame_indices=frame_indices)
    return DataLoader(
        ds,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = torch.cuda.is_available(),
        prefetch_factor    = 2 if num_workers > 0 else None,
        worker_init_fn     = _worker_init if num_workers > 0 else None,
        collate_fn         = _collate,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, num_classes: int) -> UNetSAISELD:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    model = UNetSAISELD(
        n_classes = num_classes,
        in_ch     = N_CHANNELS,
        img_h     = IMG_H,
        img_w     = IMG_W,
    )
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict):
        if   "model_state"      in state: state = state["model_state"]
        elif "model_state_dict" in state: state = state["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:    print(f"[WARN] Missing keys    : {missing}")
    if unexpected: print(f"[WARN] Unexpected keys : {unexpected}")

    model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] UNetSAISELD ready — {n_params:.1f}M params, eval() mode.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  NMS  (mentor's self-contained numpy implementation — no torchvision needed)
# ══════════════════════════════════════════════════════════════════════════════

def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    a1    = (box[2] - box[0]) * (box[3] - box[1])
    a2    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a1 + a2 - inter + 1e-6
    return inter / union


def nms_per_class(
    boxes:   np.ndarray,
    labels:  np.ndarray,
    scores:  np.ndarray,
    iou_thr: float = 0.45,
) -> np.ndarray:
    keep = []
    for cls in np.unique(labels):
        idx   = np.where(labels == cls)[0]
        s, b  = scores[idx], boxes[idx]
        order = np.argsort(-s)
        alive = np.ones(len(order), dtype=bool)
        for i, oi in enumerate(order):
            if not alive[i]:
                continue
            keep.append(idx[oi])
            ious = box_iou(b[oi], b[order[i + 1:]])
            for j, iou in enumerate(ious):
                if iou > iou_thr:
                    alive[i + 1 + j] = False
    return np.array(keep, dtype=int)


# ══════════════════════════════════════════════════════════════════════════════
#  TRACKER  (mentor's TemporalTracker — cleaner min_age/confirmed pattern)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Track:
    track_id:  int
    label:     int
    score:     float
    box:       np.ndarray
    emap:      np.ndarray    # (H, W) full-res energy map
    dist_pred: Optional[float]
    age:       int  = 1      # frames seen (increments on match)
    missed:    int  = 0      # consecutive unmatched frames
    confirmed: bool = False  # True once age >= min_age


class TemporalTracker:
    """
    Mentor's tracker: confirms tracks after min_age hits, prunes after
    max_missed consecutive misses. No coast_decay — cleaner and removes
    the source of permanent-track drift in our earlier version.
    """
    def __init__(
        self,
        iou_thr:    float = 0.30,
        min_age:    int   = 2,
        max_missed: int   = 2,
    ):
        self.iou_thr    = iou_thr
        self.min_age    = min_age
        self.max_missed = max_missed
        self._tracks: List[Track] = []
        self._next_id = 0

    def reset(self):
        self._tracks  = []
        self._next_id = 0

    def update(
        self,
        boxes:      np.ndarray,          # (N, 4)
        labels:     np.ndarray,          # (N,)
        scores:     np.ndarray,          # (N,)
        emaps:      np.ndarray,          # (N, H, W)
        dist_preds: Optional[np.ndarray],# (N,) or None
    ) -> List[Track]:
        n_det           = len(labels)
        unmatched_dets  = list(range(n_det))
        matched_track_i = set()

        if self._tracks and n_det > 0:
            track_boxes = np.stack([t.box for t in self._tracks])
            cost = np.zeros((n_det, len(self._tracks)))
            for di in range(n_det):
                ious = box_iou(boxes[di], track_boxes)
                for ti, t in enumerate(self._tracks):
                    cost[di, ti] = ious[ti] if t.label == labels[di] else 0.0

            flat = np.argsort(-cost.ravel())
            for f in flat:
                di, ti = divmod(int(f), len(self._tracks))
                if cost[di, ti] < self.iou_thr:
                    break
                if di not in unmatched_dets or ti in matched_track_i:
                    continue
                t           = self._tracks[ti]
                t.box       = boxes[di]
                t.score     = scores[di]
                t.emap      = emaps[di]
                t.dist_pred = float(dist_preds[di]) if dist_preds is not None else None
                t.age      += 1
                t.missed    = 0
                if t.age >= self.min_age:
                    t.confirmed = True
                unmatched_dets.remove(di)
                matched_track_i.add(ti)

        # Increment missed counter for unmatched existing tracks
        for ti, t in enumerate(self._tracks):
            if ti not in matched_track_i:
                t.missed += 1

        # Spawn new tracks for unmatched detections
        for di in unmatched_dets:
            self._tracks.append(Track(
                track_id  = self._next_id,
                label     = int(labels[di]),
                score     = float(scores[di]),
                box       = boxes[di].copy(),
                emap      = emaps[di].copy(),
                dist_pred = float(dist_preds[di]) if dist_preds is not None else None,
            ))
            self._next_id += 1

        # Prune dead tracks
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]
        # Return only confirmed tracks
        return [t for t in self._tracks if t.confirmed]


# ══════════════════════════════════════════════════════════════════════════════
#  PEAK EXTRACTION  (our full-res version, adapted from mentor's interface)
#
#  Mentor's extract_peaks worked on 28×28 RoI crops in EVAL (100×50) space.
#  UNetSAISELD energy maps are full (H, W) = (180, 360) in canvas pixel space.
#  We crop to the bounding box region, find top-K peaks, and return
#  [x, y, intensity] triplets in the [0,359]×[0,179] submission coordinate space.
# ══════════════════════════════════════════════════════════════════════════════

def extract_peaks(
    emap_raw:  np.ndarray,   # (H, W) full equirectangular energy map
    box_xyxy:  np.ndarray,   # [x0, y0, x1, y1] in canvas pixel coords
    n_peaks:   int = 20,
) -> List[List[float]]:
    x0, y0, x1, y1 = box_xyxy
    xi0 = max(0, int(math.floor(x0)))
    yi0 = max(0, int(math.floor(y0)))
    xi1 = min(IMG_W, int(math.ceil(x1)))
    yi1 = min(IMG_H, int(math.ceil(y1)))

    if xi1 <= xi0 or yi1 <= yi0:
        # Degenerate box → fall back to global peak
        pk      = int(emap_raw.argmax())
        py, px  = divmod(pk, emap_raw.shape[1])
        return [[float(px), float(py), float(emap_raw.flat[pk])]]

    region = emap_raw[yi0:yi1, xi0:xi1]   # (roi_h, roi_w)

    # Normalise within region so intensity is relative to local peak
    emin, emax = region.min(), region.max()
    if emax - emin < 1e-8:
        norm = np.ones_like(region)
    else:
        norm = (region - emin) / (emax - emin)

    flat    = norm.ravel()
    n_peaks = min(n_peaks, flat.size)
    top_idx = np.argpartition(flat, -n_peaks)[-n_peaks:]
    top_idx = top_idx[np.argsort(-flat[top_idx])]

    roi_w = xi1 - xi0
    triplets = []
    for j, i in enumerate(top_idx):
        ry, rx = divmod(int(i), roi_w)
        ox = float(xi0 + rx)
        oy = float(yi0 + ry)
        # Clamp to valid submission range
        ox = min(max(ox, 0.0), float(IMG_W - 1))
        oy = min(max(oy, 0.0), float(IMG_H - 1))
        # triplets.append([round(ox, 2), round(oy, 2), round(float(flat[i]), 4)])
        raw_vals = region.ravel()
        # ... inside the loop, change to:
        triplets.append([round(ox, 2), round(oy, 2), round(float(raw_vals[top_idx[j]]), 4)])

    return triplets


# ══════════════════════════════════════════════════════════════════════════════
#  PER-SEQUENCE INFERENCE  (mentor's two-pass approach with adaptive gating)
#
#  Pass 1: run model on all frames, accumulate (labels, scores, boxes, emaps).
#          Also collect all per-class scores to compute global min/max.
#  Pass 2: adaptive gating — stretch per-class scores to [0.05, 0.95],
#          threshold on SCALED score, submit RAW score (preserves mAP ranking).
#          Then NMS → TemporalTracker → fragment merging → emit annotations.
# ══════════════════════════════════════════════════════════════════════════════

def infer_sequence(
    model,
    seq_dir:      str,
    seq_name:     str,
    frame_indices: List[int],   # passed from main() = annotated_ids | disk_ids
    batch_size:   int,
    num_workers:  int,
    score_thr:    float,
    nms_iou:      float,
    track_iou:    float,
    min_age:      int,
    max_missed:   int,
    n_peaks:      int,
    dist_scale:   float,
) -> List[dict]:

    # frame_indices is provided by main() as the union of annotated frame
    # indices (from the GT JSON) and on-disk PNG indices.  This ensures
    # inference runs even when no video frames are on disk (audio-only mode).
    if not frame_indices:
        return []

    loader  = build_loader(seq_dir, seq_name, frame_indices, batch_size, num_workers)
    tracker = TemporalTracker(iou_thr=track_iou, min_age=min_age, max_missed=max_missed)
    use_amp = torch.cuda.is_available()
    annotations: List[dict] = []

    # ── Pass 1: accumulate all frame predictions ──────────────────────────
    frame_data_accum = []
    class_scores     = {c: [] for c in range(1, NUM_CLASSES)}

    with torch.no_grad():
        for bf, bt in tqdm(loader, desc=f"  {seq_name[:40]} [Infer]", unit="batch"):
            imgs = [t.to(DEVICE, non_blocking=True).float() for t in bt]
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds_batch = model(imgs, None)

            for fi, dr in zip(bf, preds_batch):
                fi = int(fi)

                def _get(keys):
                    for k in keys:
                        if k in dr:
                            v = dr[k]
                            return v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    return None

                labels_np = _get(["labels", "pred_classes", "pred_labels"])
                scores_np = _get(["scores", "pred_scores",  "confidences"])
                boxes_np  = _get(["boxes",  "pred_boxes",   "bboxes"])
                emaps_np  = _get(["energy_maps", "energy",  "heatmaps"])
                dist_np   = _get(["dist_pred"])

                if labels_np is not None and scores_np is not None and boxes_np is not None:
                    labels_np = labels_np.astype(int)
                    scores_np = scores_np.astype(np.float32)
                    boxes_np  = boxes_np.astype(np.float32)
                    if emaps_np is not None:
                        emaps_np = emaps_np.astype(np.float32)
                    for l, s in zip(labels_np, scores_np):
                        if l > 0:
                            class_scores[l].append(float(s))
                else:
                    labels_np = scores_np = boxes_np = None

                frame_data_accum.append((fi, labels_np, scores_np, boxes_np, emaps_np, dist_np))

    # ── Compute per-class global min/max for adaptive score stretching ────
    class_min_max = {}
    for c, s_list in class_scores.items():
        if s_list:
            class_min_max[c] = (float(np.min(s_list)), float(np.max(s_list)))
        else:
            class_min_max[c] = (0.0, 1.0)

    stats_raw     = {c: 0 for c in range(1, NUM_CLASSES)}
    stats_thresh  = {c: 0 for c in range(1, NUM_CLASSES)}
    stats_nms     = {c: 0 for c in range(1, NUM_CLASSES)}
    stats_emitted = {c: 0 for c in range(1, NUM_CLASSES)}

    # ── Pass 2: adaptive gating → NMS → tracker → fragment merging ───────
    empty_emaps = np.zeros((0, IMG_H, IMG_W), dtype=np.float32)

    for fi, labels_np, scores_np, boxes_np, emaps_np, dist_np in frame_data_accum:

        if labels_np is None:
            tracker.update(np.zeros((0, 4)), np.zeros(0, int),
                           np.zeros(0), empty_emaps, None)
            continue

        for l in labels_np:
            if l > 0:
                stats_raw[l] += 1

        # Adaptive gating: scale scores per-class, threshold on scaled,
        # but keep raw scores for submission (preserves mAP ranking).
        keep_mask = np.zeros(len(scores_np), dtype=bool)
        for i in range(len(scores_np)):
            c     = labels_np[i]
            raw_s = scores_np[i]
            if c > 0 and c in class_min_max:
                cmin, cmax = class_min_max[c]
                if cmax >= 0.08 and cmax > cmin:
                    scaled_s = 0.05 + 0.90 * ((raw_s - cmin) / (cmax - cmin))
                else:
                    scaled_s = raw_s
                if scaled_s >= score_thr:
                    keep_mask[i] = True

        keep = keep_mask & (labels_np > 0)
        for l in labels_np[keep]:
            stats_thresh[l] += 1

        if keep.sum() == 0:
            tracker.update(np.zeros((0, 4)), np.zeros(0, int),
                           np.zeros(0), empty_emaps, None)
            continue

        labels_f = labels_np[keep]
        scores_f = scores_np[keep]   # RAW scores — submitted as-is
        boxes_f  = boxes_np[keep]
        emaps_f  = emaps_np[keep] if emaps_np is not None else np.zeros((keep.sum(), IMG_H, IMG_W))
        dist_f   = dist_np[keep]  if dist_np  is not None else None

        nms_idx          = nms_per_class(boxes_f, labels_f, scores_f, iou_thr=nms_iou)
        labels_f         = labels_f[nms_idx]
        scores_f         = scores_f[nms_idx]
        boxes_f          = boxes_f[nms_idx]
        emaps_f          = emaps_f[nms_idx]
        dist_f           = dist_f[nms_idx] if dist_f is not None else None

        for l in labels_f:
            stats_nms[l] += 1

        confirmed = tracker.update(boxes_f, labels_f, scores_f, emaps_f, dist_f)

        # Fragment merging: group confirmed tracks by category,
        # cluster spatially adjacent ones (centres within 50px),
        # emit one merged annotation per cluster.
        emitted_by_cat = defaultdict(list)
        for track in confirmed:
            cat_id = MODEL_TO_CAT.get(track.label)
            if cat_id is None:
                continue
            triplets = extract_peaks(track.emap, track.box, n_peaks=n_peaks)
            if not triplets:
                continue
            emitted_by_cat[cat_id].append({
                "box":      track.box,
                "score":    float(track.score),
                "triplets": triplets,
                "dist":     float(track.dist_pred) * dist_scale
                            if track.dist_pred is not None else None,
            })

        for cat_id, fragments in emitted_by_cat.items():
            clusters = []
            for frag in fragments:
                cx = (frag["box"][0] + frag["box"][2]) / 2.0
                cy = (frag["box"][1] + frag["box"][3]) / 2.0
                placed = False
                for clus in clusters:
                    for c_frag in clus:
                        ccx = (c_frag["box"][0] + c_frag["box"][2]) / 2.0
                        ccy = (c_frag["box"][1] + c_frag["box"][3]) / 2.0
                        if np.hypot(cx - ccx, cy - ccy) < 50.0:
                            clus.append(frag)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    clusters.append([frag])

            for clus in clusters:
                all_triplets = []
                max_score    = -1.0
                dists        = []
                for frag in clus:
                    all_triplets.extend(frag["triplets"])
                    if frag["score"] > max_score:
                        max_score = frag["score"]
                    if frag["dist"] is not None:
                        dists.append(frag["dist"])

                all_triplets.sort(key=lambda x: x[2], reverse=True)

                original_cls = next(k for k, v in MODEL_TO_CAT.items() if v == cat_id)
                stats_emitted[original_cls] += 1

                entry: dict = {
                    "metadata_frame_index": fi,
                    "category_id":          cat_id,
                    "score":                round(max_score, 5),
                    "segmentation":         [all_triplets[:n_peaks * 2]],
                }
                if dists:
                    entry["distance"] = round(sum(dists) / len(dists), 1)
                annotations.append(entry)

    # ── Per-class detection funnel (mentor's diagnostic printout) ─────────
    print("\n  → Per-Class Detection Funnel:")
    print("    Class |   Raw | >Thr |  NMS | Emitted | Score Range")
    print("    " + "-" * 57)
    for c in range(1, NUM_CLASSES):
        if stats_raw[c] > 0 or stats_emitted[c] > 0:
            cmin, cmax = class_min_max[c]
            print(f"    {c:5d} | {stats_raw[c]:5d} | {stats_thresh[c]:4d} | "
                  f"{stats_nms[c]:4d} | {stats_emitted[c]:7d} | "
                  f"({cmin:.4f}, {cmax:.4f})")
    print()

    return annotations


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submission-format inference — UNetSAISELD (Track A)")

    p.add_argument("--exp_dir",    type=str, required=True,
                   help="Experiment directory (e.g. experiments/unet_run_...)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override checkpoint. Defaults to unet_saiseld_best.pth in exp_dir.")
    p.add_argument("--split",      default="test")
    p.add_argument("--n_seqs",     type=int, default=-1,
                   help="Number of sequences to process (-1 = all).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write JSON files. Defaults to exp_dir/inference_outputs.")
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--num_workers",type=int, default=NUM_WORKERS)
    p.add_argument("--num_classes",type=int, default=NUM_CLASSES)
    p.add_argument("--seed",       type=int, default=42)

    p.add_argument("--score_thr",  type=float, default=0.35,
                   help="Threshold on SCALED (per-class normalised) score.")
    p.add_argument("--nms_iou",    type=float, default=0.45)
    p.add_argument("--track_iou",  type=float, default=0.30)
    p.add_argument("--min_age",    type=int,   default=2)
    p.add_argument("--max_missed", type=int,   default=2)
    p.add_argument("--n_peaks",    type=int,   default=20)
    p.add_argument("--dist_scale", type=float, default=DIST_NORM,
                   help="Multiply model distance output by this to get centimetres.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.exp_dir, "unet_saiseld_best.pth")
    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_dir, "inference_outputs")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = Logger(os.path.join(args.exp_dir, "inference.log"),       sys.stdout)
    sys.stderr = Logger(os.path.join(args.exp_dir, "inference_error.log"), sys.stderr)

    print(f"\n[INFO] Device              : {DEVICE}  [{_gpu_name}]")
    print(f"[INFO] Model               : UNetSAISELD  ({N_CHANNELS}ch SALSA-Lite)")
    print(f"[INFO] Checkpoint          : {args.checkpoint}")
    print(f"[INFO] Output directory    : {out_dir}")
    print(f"[INFO] batch_size          : {args.batch_size}")
    print(f"[INFO] num_workers         : {args.num_workers}")
    print(f"\n[INFO] ── Filtering pipeline ──────────────────────────────────")
    print(f"[INFO]   score_thr  (scaled) = {args.score_thr}")
    print(f"[INFO]   nms_iou             = {args.nms_iou}")
    print(f"[INFO]   track_iou           = {args.track_iou}")
    print(f"[INFO]   min_age             = {args.min_age}")
    print(f"[INFO]   max_missed          = {args.max_missed}")
    print(f"[INFO]   n_peaks             = {args.n_peaks}")
    print(f"[INFO]   dist_scale          = {args.dist_scale}")
    print(f"[INFO] ────────────────────────────────────────────────────────\n")

    model      = load_model(args.checkpoint, args.num_classes)
    test_infos = get_sequence_infos(args.split, LABELS_BASE, FRAMES_BASE)

    if not test_infos:
        print(f"[ERROR] No sequences found for split='{args.split}'")
        sys.exit(1)

    if 0 < args.n_seqs < len(test_infos):
        indices   = np.linspace(0, len(test_infos) - 1, args.n_seqs).astype(int)
        test_infos = [test_infos[i] for i in indices]

    print(f"[INFO] {len(test_infos)} sequences to process.\n")

    total_annots = 0
    skipped      = []

    for seq_idx, (json_path, seq_dir, seq_name) in enumerate(test_infos, 1):
        print(f"\n{'='*60}")
        print(f"[{seq_idx:3d}/{len(test_infos)}]  {seq_name}")

        # Combine on-disk frames with annotated frame indices
        try:
            import json as _json
            with open(json_path) as jf:
                ann_data = _json.load(jf)
            annotated_ids = sorted(
                {int(a["metadata_frame_index"])
                 for a in ann_data.get("annotations", [])})
        except Exception as exc:
            print(f"  [WARN] JSON unreadable ({exc}) — using disk frames only.")
            annotated_ids = []

        disk_ids = set(scan_available_frames(seq_dir, seq_name))
        all_ids  = sorted(set(annotated_ids) | disk_ids)

        if not all_ids:
            print(f"  [WARN] No frames found — skipping.")
            skipped.append(seq_name)
            continue

        print(f"  Frames: {len(all_ids)}  "
              f"(annotated={len(annotated_ids)}, on-disk={len(disk_ids)})")

        annotations = infer_sequence(
            model          = model,
            seq_dir        = seq_dir,
            seq_name       = seq_name,
            frame_indices  = all_ids,   # union of GT-annotated + on-disk frames
            batch_size     = args.batch_size,
            num_workers    = args.num_workers,
            score_thr      = args.score_thr,
            nms_iou        = args.nms_iou,
            track_iou      = args.track_iou,
            min_age        = args.min_age,
            max_missed     = args.max_missed,
            n_peaks        = args.n_peaks,
            dist_scale     = args.dist_scale,
        )

        json_out = out_dir / f"{seq_name}.json"
        with open(json_out, "w") as f:
            json.dump({"annotations": annotations}, f, separators=(",", ":"))

        n        = len(annotations)
        total_annots += n
        size_kb  = json_out.stat().st_size / 1024
        print(f"  → {n:5d} annotations  |  {size_kb:>7.1f} KB  →  {json_out.name}")

    print(f"\n{'='*60}")
    print(f"[DONE] {len(test_infos) - len(skipped)}/{len(test_infos)} sequences written to {out_dir}/")
    if skipped:
        print(f"  Skipped : {', '.join(skipped)}")
    print(f"  Total annotations : {total_annots:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()