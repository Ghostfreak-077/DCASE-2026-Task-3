# ============================================================
#  ENERGY-FIELD INSTANCE SEGMENTATION  —  INFERENCE SCRIPT
#  run_inference.py  (UNetSAISELD / Path B1 edition)
#
#  Usage:
#      python run_inference.py --exp_dir experiments/unet_run_...
#      python run_inference.py --exp_dir path --score_thr 0.25
#      python run_inference.py --exp_dir path --split test --num_seqs 5
#
#  Filtering pipeline (in order, identical to baseline):
#    1. Score threshold       — drop low-confidence detections
#    2. Per-class NMS         — remove spatially overlapping boxes
#    3. Per-frame class cap   — at most N detections per class per frame
#    4. Track confirmation    — only export tracks seen >= min_hits frames
#    5. Energy sparsification — export top-K energy points per detection
#
#  Changes vs baseline run_inference.py:
#    - AcousticFeatureExtractor  →  SalsaFeatureExtractor  (no UpLAM)
#    - EnergyInstanceModel       →  UNetSAISELD
#    - InferenceDataset          →  audio-only, outputs (4, T, F) tensors
#    - energy_maps are (H, W) full-res instead of 28×28 RoI crops
#    - sparsify_energy_map works on full-res maps
#    - checkpoint default: unet_saiseld_best.pth
#    - All output format, filtering, JSON structure: UNCHANGED
# ============================================================

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings
import random
from collections import defaultdict, OrderedDict

warnings.filterwarnings("ignore")

import torch._dynamo
torch._dynamo.disable()

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Resolve script directory so imports always work ──────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── Import from our new modules ───────────────────────────────────────────────
from acoustic_features import SalsaFeatureExtractor, wav_path_from_seq_dir
from model import (
    UNetSAISELD,
    InstanceTracker,
    get_sequence_infos,
    scan_available_frames,
    frame_path,
)


# ══════════════════════════════════════════════════════════════════════════════
#  MLOps UTILITIES  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")

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
LABELS_BASE = "/teamspace/studios/this_studio/data/labels_dev"
MIC_BASE    = "/teamspace/studios/this_studio/data/foa_dev"

IMG_W        = 360
IMG_H        = 180
N_CHANNELS   = 4          # SALSA-Lite: 1 log-mel + 3 NIPV (no RGB, no UpLAM)
NUM_CLASSES  = 14
DIST_NORM    = 500.0
INFERENCE_HZ = 10
SCORE_THR    = 0.05
ENERGY_EXPORT_THR = 0.10
COAST_DECAY  = 0.9

# SALSA-Lite extractor hyperparams (must match train.py)
SALSA_FS           = 24000
SALSA_N_FFT        = 512
SALSA_HOP          = 300
SALSA_FMIN_DOA     = 40.0
SALSA_FMAX_DOA     = 6000.0
SALSA_FMAX_SPEC    = 6000.0
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
#  INFERENCE DATASET
#  Key change: audio-only, outputs (4, T, F) SALSA-Lite tensors per frame.
#  No RGB loading. No UpLAM dependency.
# ══════════════════════════════════════════════════════════════════════════════

class InferenceDataset(Dataset):
    def __init__(
        self,
        seq_dir:       str,
        seq_name:      str,
        frame_indices: list,
        img_w:         int  = IMG_W,
        img_h:         int  = IMG_H,
    ):
        self.seq_dir       = seq_dir
        self.seq_name      = seq_name
        self.frame_indices = frame_indices
        self.img_w         = img_w
        self.img_h         = img_h
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
                audio_cache_size = 4,    # small per-worker cache
                frame_cache_size = 128,
            )

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        self._ensure_extractor()
        fi     = self.frame_indices[idx]
        tensor = self._extractor.get_frame_bands(self.wav_path, fi)
        # tensor: (4, T, F)  float32 on CPU
        return fi, tensor


def _worker_init(worker_id):
    """Re-initialise the SALSA extractor in each worker process."""
    ds = torch.utils.data.get_worker_info().dataset
    ds._extractor = SalsaFeatureExtractor(
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


def _collate(batch):
    fis     = [x[0] for x in batch]
    tensors = [x[1] for x in batch]
    return fis, tensors


def build_loader(seq_dir, seq_name, frame_indices, batch_size, num_workers):
    ds = InferenceDataset(
        seq_dir       = seq_dir,
        seq_name      = seq_name,
        frame_indices = frame_indices,
    )
    return DataLoader(
        ds,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = torch.cuda.is_available(),
        prefetch_factor    = 4 if num_workers > 0 else None,
        worker_init_fn     = _worker_init if num_workers > 0 else None,
        persistent_workers = num_workers > 0,
        drop_last          = False,
        collate_fn         = _collate,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING  (unchanged interface; removed --audio_only since Track A
#  is always audio-only; updated default checkpoint name)
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run trained UNetSAISELD on test sequences (Track A).")

    p.add_argument("--exp_dir",    type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override checkpoint path. Defaults to unet_saiseld_best.pth in exp_dir.")
    p.add_argument("--split",      default="test")
    p.add_argument("--num_seqs",   type=int, default=None)
    p.add_argument("--seed",       type=int, default=42)

    p.add_argument("--score_thr",          type=float, default=0.15,
                   help="Energy peak threshold — matches ENERGY_SCORE_THR in _build_detections")
    p.add_argument("--nms_iou_thr",        type=float, default=0.30)
    p.add_argument("--max_dets_per_class", type=int,   default=3)
    p.add_argument("--min_hits",           type=int,   default=2)
    p.add_argument("--energy_top_k",       type=int,   default=20)
    p.add_argument("--energy_export_thr",  type=float, default=ENERGY_EXPORT_THR)

    p.add_argument("--iou_thr",     type=float, default=0.3)
    p.add_argument("--max_age",     type=int,   default=5)
    p.add_argument("--coast_decay", type=float, default=COAST_DECAY)

    p.add_argument("--batch_size",  type=int, default=48,
                   help="Frames per forward pass (UNet is lighter; default 48)")
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
#  Instantiates UNetSAISELD and loads the checkpoint saved by train.py.
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, num_classes: int) -> UNetSAISELD:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint : {checkpoint_path}")
    model = UNetSAISELD(
        n_classes = num_classes,
        in_ch     = N_CHANNELS,   # 4 SALSA-Lite channels
        img_h     = IMG_H,
        img_w     = IMG_W,
    )
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:    print(f"[WARN] Missing keys    : {missing}")
    if unexpected: print(f"[WARN] Unexpected keys : {unexpected}")

    model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] UNetSAISELD ready — {n_params:.1f}M params, eval() mode.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-STAGE FILTERING  (logic identical; adapted for full-res energy maps)
# ══════════════════════════════════════════════════════════════════════════════

def filter_detections(det, score_thr, nms_iou_thr, max_dets_per_class):
    boxes       = det.get("boxes",       torch.zeros(0, 4))
    labels      = det.get("labels",      torch.zeros(0, dtype=torch.long))
    scores      = det.get("scores",      torch.zeros(0))
    # energy_maps: (N, H, W) full-resolution from UNetSAISELD
    energy_maps = det.get("energy_maps", torch.zeros(0, IMG_H, IMG_W))
    dist_pred   = det.get("dist_pred",   torch.zeros(0))
    N           = scores.shape[0]

    _empty = {
        **det,
        "boxes":       torch.zeros(0, 4),
        "labels":      torch.zeros(0, dtype=torch.long),
        "scores":      torch.zeros(0),
        "energy_maps": torch.zeros(0, IMG_H, IMG_W),
        "dist_pred":   torch.zeros(0),
    }

    if N == 0:
        return _empty

    # Stage 1: score threshold
    keep = scores >= score_thr
    if not keep.any():
        return _empty

    def _sel(t, mask):
        return t[mask] if isinstance(t, torch.Tensor) and t.shape[0] == N else t

    boxes       = _sel(boxes, keep)
    labels      = _sel(labels, keep)
    scores      = _sel(scores, keep)
    energy_maps = _sel(energy_maps, keep)
    dist_pred   = _sel(dist_pred, keep)
    N = scores.shape[0]

    # Stage 2: per-class NMS
    keep_nms = []
    for cls in labels.unique():
        m   = labels == cls
        idx = m.nonzero(as_tuple=True)[0]
        keep_nms.append(idx[nms(boxes[idx], scores[idx], nms_iou_thr)])
    if not keep_nms:
        return _empty
    ki = torch.cat(keep_nms)
    ki = ki[scores[ki].argsort(descending=True)]

    def _idx_sel(t):
        return t[ki] if isinstance(t, torch.Tensor) and t.shape[0] >= ki.max() + 1 else t

    boxes       = _idx_sel(boxes)
    labels      = _idx_sel(labels)
    scores      = _idx_sel(scores)
    energy_maps = _idx_sel(energy_maps)
    dist_pred   = _idx_sel(dist_pred)

    # Stage 3: per-frame class cap
    keep_cap = []
    for cls in labels.unique():
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        keep_cap.append(idx[:max_dets_per_class])
    if not keep_cap:
        return _empty
    ki2 = torch.cat(keep_cap)

    def _idx_sel2(t):
        return t[ki2] if isinstance(t, torch.Tensor) and t.shape[0] >= ki2.max() + 1 else t

    return {
        **det,
        "boxes":       _idx_sel2(boxes),
        "labels":      _idx_sel2(labels),
        "scores":      _idx_sel2(scores),
        "energy_maps": _idx_sel2(energy_maps),
        "dist_pred":   _idx_sel2(dist_pred),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENERGY SPARSIFICATION
#  Adapted for full-resolution (H, W) energy maps from UNetSAISELD.
#  The baseline used 28×28 RoI crops and interpolated back to box coordinates.
#  We now work directly in equirectangular pixel space — no interpolation needed.
# ══════════════════════════════════════════════════════════════════════════════

def sparsify_energy_map(emap: np.ndarray, box: list, top_k: int, thr: float) -> list:
    """
    Extract top-K energy peaks from a full-resolution (H, W) energy map,
    restricted to the bounding box region of the detection.

    Parameters
    ----------
    emap : (H, W) float32 numpy array — full equirectangular energy map
    box  : [x0, y0, x1, y1] in pixel coords
    top_k: max number of peaks to export
    thr  : minimum energy threshold (fraction of max in region)

    Returns
    -------
    List of [x, y, intensity] triplets, sorted by descending intensity.
    Format matches the DCASE 2026 submission spec exactly.
    """
    x0, y0, x1, y1 = box
    # Clamp box to canvas bounds
    xi0 = max(0, int(math.floor(x0)))
    yi0 = max(0, int(math.floor(y0)))
    xi1 = min(emap.shape[1], int(math.ceil(x1)))
    yi1 = min(emap.shape[0], int(math.ceil(y1)))

    if xi1 <= xi0 or yi1 <= yi0:
        # Degenerate box — fall back to global peak
        pk  = int(emap.argmax())
        py, px = divmod(pk, emap.shape[1])
        return [[float(px), float(py), float(emap.flat[pk])]]

    region = emap[yi0:yi1, xi0:xi1]    # (roi_h, roi_w)
    flat   = region.flatten()
    n_pts  = flat.shape[0]

    # Top-K selection
    if 0 < top_k < n_pts:
        top_idx = np.argpartition(flat, -top_k)[-top_k:]
    else:
        top_idx = np.arange(n_pts)

    # Apply threshold relative to region peak
    region_max = flat.max()
    abs_thr    = thr * region_max if region_max > 1e-9 else thr
    top_idx    = top_idx[flat[top_idx] >= abs_thr]

    triplets = []
    roi_h = yi1 - yi0
    roi_w = xi1 - xi0
    for i in top_idx:
        ry, rx = divmod(int(i), roi_w)
        # Convert back to full-canvas coordinates
        x_coord = float(xi0 + rx)
        y_coord = float(yi0 + ry)
        # Clamp to valid submission range
        x_coord = max(0.0, min(359.0, x_coord))
        y_coord = max(0.0, min(179.0, y_coord))
        triplets.append([
            round(x_coord, 4),
            round(y_coord, 4),
            round(float(flat[i]), 6),
        ])

    # Guarantee at least one triplet
    if not triplets:
        pk  = int(flat.argmax())
        ry, rx = divmod(pk, roi_w)
        x_coord = max(0.0, min(359.0, float(xi0 + rx)))
        y_coord = max(0.0, min(179.0, float(yi0 + ry)))
        triplets.append([round(x_coord, 4), round(y_coord, 4), round(float(flat[pk]), 6)])

    triplets.sort(key=lambda t: -t[2])
    return triplets


# ══════════════════════════════════════════════════════════════════════════════
#  PER-SEQUENCE INFERENCE  (unchanged logic; removed audio_only arg)
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_on_sequence(
    model, seq_dir, seq_name, frame_indices,
    score_thr, nms_iou_thr, max_dets_per_class,
    min_hits, iou_thr, max_age, batch_size, num_workers,
    coast_decay=COAST_DECAY,
):
    assert not model.training
    loader  = build_loader(seq_dir, seq_name, frame_indices, batch_size, num_workers)
    tracker = InstanceTracker(iou_thr=iou_thr, max_age=max_age, coast_decay=coast_decay,
                              img_w=IMG_W, img_h=IMG_H)
    results = []
    track_score_memory: dict = {}
    use_amp = torch.cuda.is_available()
    n_raw = n_kept = 0
    seq_t0 = time.time()

    with torch.no_grad():
        pbar = tqdm(
            loader,
            total        = math.ceil(len(frame_indices) / batch_size),
            desc         = f"  {seq_name[:35]}",
            unit         = "batch",
            dynamic_ncols= True,
            leave        = True,
        )

        for batch_fi, batch_tensors in pbar:
            # batch_tensors: list of (4, T, F) tensors
            images_gpu = [t.to(DEVICE, non_blocking=True).float() for t in batch_tensors]

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds_batch = model(images_gpu, None)

            for fi, det_raw in zip(batch_fi, preds_batch):
                det = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                       for k, v in det_raw.items()}

                n_raw  += int(det.get("scores", torch.zeros(0)).shape[0])
                det     = filter_detections(det, score_thr, nms_iou_thr, max_dets_per_class)
                n_kept += int(det.get("scores", torch.zeros(0)).shape[0])

                tracked = tracker.update(det)

                # Score memory with coast decay
                for obj in tracked:
                    tid = obj["track_id"]
                    if not obj.get("coasting", False):
                        track_score_memory[tid] = float(obj.get("score", 0.0))
                    else:
                        prev    = track_score_memory.get(tid, 0.0)
                        decayed = prev * coast_decay
                        track_score_memory[tid] = decayed
                        obj["score"] = decayed
                    obj["hits"] = (tracker.tracks[tid]["hits"]
                                   if tid in tracker.tracks else obj.get("hits", 1))

                full_e = (det["full_energy"].numpy()
                          if "full_energy" in det
                          else np.zeros((IMG_H, IMG_W), dtype=np.float32))

                results.append(dict(
                    frame       = int(fi),
                    time_s      = int(fi) / INFERENCE_HZ,
                    objects     = tracked,
                    full_energy = full_e,
                ))

            elapsed = max(time.time() - seq_t0, 1e-6)
            pbar.set_postfix({
                "fps":    f"{len(results)/elapsed:.1f}",
                "raw":    n_raw,
                "kept":   n_kept,
                "tracks": len(tracker.tracks),
            }, refresh=True)

    elapsed = time.time() - seq_t0
    n = len(frame_indices)
    print(f"  [INFO] {n} frames in {elapsed:.1f}s  ({n/max(elapsed,1e-6):.1f} fps) | "
          f"raw={n_raw}  after_filter={n_kept}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  JSON SERIALISATION  (format identical to baseline — evaluator unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def dump_inference_json(results, save_path, min_hits, energy_top_k, energy_export_thr):
    # First pass: find max track hits for confirmation filter
    max_hits: dict = {}
    for frame_result in results:
        for obj in frame_result["objects"]:
            tid = obj["track_id"]
            max_hits[tid] = max(max_hits.get(tid, 0), obj.get("hits", 1))

    annotations = []
    for frame_result in results:
        fi = frame_result["frame"]
        for obj in frame_result["objects"]:
            tid = obj["track_id"]
            if max_hits.get(tid, 0) < min_hits:
                continue

            emap = obj["energy_map"]
            if isinstance(emap, torch.Tensor):
                emap = emap.numpy()

            # energy_map from UNetSAISELD is full (H, W); box is in pixel coords
            triplets = sparsify_energy_map(
                emap, obj["box"], energy_top_k, energy_export_thr
            )

            annotations.append({
                "metadata_frame_index": int(fi),
                "instance_id"         : int(obj["track_id"]),
                "category_id"         : int(obj["label"]) - 1,   # 0-indexed for submission
                "score"               : round(float(obj.get("score", 0.0)), 6),
                "distance"            : round(float(obj["dist_pred"]) * DIST_NORM, 4),
                "segmentation"        : [triplets],
            })

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"annotations": annotations}, f, indent=2)

    n_frames_out = len({a["metadata_frame_index"] for a in annotations})
    print(f"  [JSON] {len(annotations):5d} annotations across "
          f"{n_frames_out:4d} frames  →  {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results, seq_name, min_hits):
    max_hits_summary: dict = {}
    for r in results:
        for o in r["objects"]:
            tid = o["track_id"]
            max_hits_summary[tid] = max(max_hits_summary.get(tid, 0), o.get("hits", 1))

    confirmed  = [o for r in results for o in r["objects"]
                  if max_hits_summary.get(o["track_id"], 0) >= min_hits]
    n_frames   = len(results)
    n_active   = sum(1 for r in results
                     if any(max_hits_summary.get(o["track_id"], 0) >= min_hits
                            for o in r["objects"]))
    cat_counts: dict = defaultdict(int)
    for o in confirmed:
        cat_counts[int(o["label"]) - 1] += 1
    duration_s = max((r["time_s"] for r in results), default=0.0)

    print(f"\n  +-- {seq_name}")
    print(f"  |  Frames          : {n_frames:>6d}  ({duration_s:.1f}s)")
    print(f"  |  Active frames   : {n_active:>6d}  ({100*n_active/max(n_frames,1):.1f}%)")
    print(f"  |  Confirmed objs  : {len(confirmed):>6d}")
    print(f"  |  Unique tracks   : {len({o['track_id'] for o in confirmed}):>6d}")
    if cat_counts:
        print(f"  |  Category counts : "
              + "  ".join(f"C{c}:{n}" for c, n in sorted(cat_counts.items())))
    print(f"  +{'--'*25}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN  (identical structure to baseline)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.exp_dir, "unet_saiseld_best.pth")

    args.out_dir = os.path.join(args.exp_dir, "inference_outputs")
    os.makedirs(args.out_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(args.exp_dir, "inference.log"),       sys.stdout)
    sys.stderr = Logger(os.path.join(args.exp_dir, "inference_error.log"), sys.stderr)

    print(f"\n[INFO] Device              : {DEVICE}  [{_gpu_name}]")
    print(f"[INFO] Model               : UNetSAISELD  ({N_CHANNELS}ch SALSA-Lite input)")
    print(f"[INFO] Experiment Dir      : {args.exp_dir}")
    print(f"[INFO] Output directory    : {args.out_dir}")
    print(f"[INFO] batch_size          : {args.batch_size}")
    print(f"[INFO] num_workers         : {args.num_workers}")
    print(f"\n[INFO] ── Filtering pipeline ──────────────────────────────────")
    print(f"[INFO]   Stage 1  score_thr          = {args.score_thr}")
    print(f"[INFO]   Stage 2  nms_iou_thr        = {args.nms_iou_thr}")
    print(f"[INFO]   Stage 3  max_dets_per_class = {args.max_dets_per_class}")
    print(f"[INFO]   Stage 4  min_hits           = {args.min_hits}")
    print(f"[INFO]   Stage 5  energy_top_k       = {args.energy_top_k}")
    print(f"[INFO]            energy_export_thr  = {args.energy_export_thr}")
    print(f"[INFO] ── Tracker ─────────────────────────────────────────────")
    print(f"[INFO]   iou_thr     = {args.iou_thr}")
    print(f"[INFO]   max_age     = {args.max_age}")
    print(f"[INFO]   coast_decay = {args.coast_decay}")
    print(f"[INFO] ────────────────────────────────────────────────────────\n")

    model      = load_model(args.checkpoint, args.num_classes)
    test_infos = get_sequence_infos(args.split, LABELS_BASE, FRAMES_BASE)

    if not test_infos:
        print(f"[ERROR] No sequences found — split='{args.split}'")
        sys.exit(1)

    if args.num_seqs is not None:
        if args.num_seqs >= len(test_infos):
            print(f"[INFO] Requested num_seqs ({args.num_seqs}) >= available "
                  f"({len(test_infos)}). Using all.")
        else:
            print(f"[INFO] Randomly sampling {args.num_seqs} / {len(test_infos)} sequences")
            test_infos = random.sample(test_infos, args.num_seqs)

    print(f"[INFO] {len(test_infos)} sequences to process.\n")

    wall_t0      = time.time()
    output_files = []
    skipped      = []

    for seq_idx, (json_path, seq_dir, seq_name) in enumerate(test_infos, 1):
        print(f"\n{'='*60}")
        print(f"[{seq_idx:3d}/{len(test_infos)}]  {seq_name}")

        try:
            with open(json_path) as jf:
                ann_data = json.load(jf)
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

        results = run_inference_on_sequence(
            model              = model,
            seq_dir            = seq_dir,
            seq_name           = seq_name,
            frame_indices      = all_ids,
            score_thr          = args.score_thr,
            nms_iou_thr        = args.nms_iou_thr,
            max_dets_per_class = args.max_dets_per_class,
            min_hits           = args.min_hits,
            iou_thr            = args.iou_thr,
            max_age            = args.max_age,
            batch_size         = args.batch_size,
            num_workers        = args.num_workers,
            coast_decay        = args.coast_decay,
        )

        print_summary(results, seq_name, args.min_hits)

        out_json = os.path.join(args.out_dir, f"{seq_name}_inference.json")
        dump_inference_json(
            results           = results,
            save_path         = out_json,
            min_hits          = args.min_hits,
            energy_top_k      = args.energy_top_k,
            energy_export_thr = args.energy_export_thr,
        )
        output_files.append(out_json)

    wall_elapsed = time.time() - wall_t0
    print(f"\n{'='*60}")
    print(f"  DONE — {len(output_files)}/{len(test_infos)} sequences")
    if skipped:
        print(f"  Skipped  : {', '.join(skipped)}")
    print(f"  Wall time: {wall_elapsed:.1f}s  ({wall_elapsed/60:.1f} min)")
    print(f"  Output   : {os.path.abspath(args.out_dir)}")
    for fpath in output_files:
        print(f"    {os.path.basename(fpath):<50s}  "
              f"{os.path.getsize(fpath)/1024:.1f} KB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()