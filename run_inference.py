import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore")
import torch._dynamo
torch._dynamo.disable()

import numpy as np
import torch
from tqdm import tqdm

from model import UNetSAISELD, get_sequence_infos, scan_available_frames
from data.inference_dataset import build_audio_inference_loader
from utils.metrics import nms_per_class, extract_peaks

FRAMES_BASE = "/teamspace/studios/this_studio/data/"
LABELS_BASE = "/teamspace/studios/this_studio/data/labels_dev"
MIC_BASE    = "/teamspace/studios/this_studio/data/foa_dev"

IMG_W, IMG_H = 360, 180
N_CHANNELS = 4        
NUM_CLASSES = 14       
DIST_NORM = 500.0
MODEL_TO_CAT = {k: k - 1 for k in range(1, NUM_CLASSES)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger:
    def __init__(self, filename, stream=sys.stdout):
        self.terminal, self.log = stream, open(filename, "a", encoding="utf-8")
    def write(self, message): self.terminal.write(message); self.log.write(message)
    def flush(self): self.terminal.flush(); self.log.flush()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def infer_sequence(model, seq_dir: str, seq_name: str, frame_indices: List[int], args) -> List[dict]:
    if not frame_indices: return []
    loader = build_audio_inference_loader(seq_dir, seq_name, frame_indices, args.batch_size, args.num_workers, FRAMES_BASE, MIC_BASE)
    tracker = TemporalTracker(iou_thr=args.track_iou, min_age=args.min_age, max_missed=args.max_missed)
    annotations = []

    frame_data_accum = []; class_scores = {c: [] for c in range(1, NUM_CLASSES)}
    with torch.no_grad():
        for bf, bt in tqdm(loader, desc=f"  {seq_name[:40]} [Infer]", unit="batch"):
            imgs = [t.to(DEVICE, non_blocking=True).float() for t in bt]
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()): preds_batch = model(imgs, None)
            for fi, dr in zip(bf, preds_batch):
                def _get(keys):
                    for k in keys:
                        if k in dr: return dr[k].cpu().numpy() if isinstance(dr[k], torch.Tensor) else dr[k]
                    return None
                labels = _get(["labels", "pred_classes", "pred_labels"])
                scores = _get(["scores", "pred_scores", "confidences"])
                boxes = _get(["boxes", "pred_boxes", "bboxes"])
                emaps = _get(["energy_maps", "energy", "heatmaps"])
                dists = _get(["dist_pred"])
                if labels is not None and scores is not None and boxes is not None:
                    labels, scores, boxes = labels.astype(int), scores.astype(np.float32), boxes.astype(np.float32)
                    if emaps is not None: emaps = emaps.astype(np.float32)
                    for l, s in zip(labels, scores):
                        if l > 0: class_scores[l].append(float(s))
                else: labels = scores = boxes = None
                frame_data_accum.append((int(fi), labels, scores, boxes, emaps, dists))

    class_min_max = {c: (float(np.min(s)), float(np.max(s))) if s else (0.0, 1.0) for c, s in class_scores.items()}
    stats_raw, stats_thresh, stats_nms, stats_emitted = [defaultdict(int) for _ in range(4)]
    empty_emaps = np.zeros((0, IMG_H, IMG_W), dtype=np.float32)

    for fi, labels_np, scores_np, boxes_np, emaps_np, dist_np in frame_data_accum:
        if labels_np is None:
            tracker.update(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), empty_emaps, None); continue
        for l in labels_np:
            if l > 0: stats_raw[l] += 1

        keep_mask = np.zeros(len(scores_np), dtype=bool)
        for i in range(len(scores_np)):
            c, raw_s = labels_np[i], scores_np[i]
            if c > 0 and c in class_min_max:
                cmin, cmax = class_min_max[c]
                scaled_s = 0.05 + 0.90 * ((raw_s - cmin) / (cmax - cmin)) if (cmax >= 0.08 and cmax > cmin) else raw_s
                if scaled_s >= args.score_thr: keep_mask[i] = True

        keep = keep_mask & (labels_np > 0)
        for l in labels_np[keep]: stats_thresh[l] += 1
        if keep.sum() == 0:
            tracker.update(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), empty_emaps, None); continue

        labels_f, scores_f, boxes_f = labels_np[keep], scores_np[keep], boxes_np[keep]
        emaps_f = emaps_np[keep] if emaps_np is not None else np.zeros((keep.sum(), IMG_H, IMG_W))
        dist_f = dist_np[keep] if dist_np is not None else None

        nms_idx = nms_per_class(boxes_f, labels_f, scores_f, iou_thr=args.nms_iou)
        labels_f, scores_f, boxes_f, emaps_f = labels_f[nms_idx], scores_f[nms_idx], boxes_f[nms_idx], emaps_f[nms_idx]
        dist_f = dist_f[nms_idx] if dist_f is not None else None

        for l in labels_f: stats_nms[l] += 1
        confirmed = tracker.update(boxes_f, labels_f, scores_f, emaps_f, dist_f)

        emitted_by_cat = defaultdict(list)
        for track in confirmed:
            cat_id = MODEL_TO_CAT.get(track.label)
            if cat_id is None: continue
            triplets = extract_peaks(track.emap, track.box, n_peaks=args.n_peaks)
            if not triplets: continue
            emitted_by_cat[cat_id].append({"box": track.box, "score": float(track.score), "triplets": triplets, "dist": float(track.dist_pred) * args.dist_scale if track.dist_pred is not None else None})

        for cat_id, fragments in emitted_by_cat.items():
            clusters = []
            for frag in fragments:
                cx, cy = (frag["box"][0] + frag["box"][2]) / 2.0, (frag["box"][1] + frag["box"][3]) / 2.0
                placed = False
                for clus in clusters:
                    for c_frag in clus:
                        ccx, ccy = (c_frag["box"][0] + c_frag["box"][2]) / 2.0, (c_frag["box"][1] + c_frag["box"][3]) / 2.0
                        if np.hypot(cx - ccx, cy - ccy) < 50.0: clus.append(frag); placed = True; break
                    if placed: break
                if not placed: clusters.append([frag])

            for clus in clusters:
                all_triplets, dists, max_score = [], [], -1.0
                for frag in clus:
                    all_triplets.extend(frag["triplets"])
                    if frag["score"] > max_score: max_score = frag["score"]
                    if frag["dist"] is not None: dists.append(frag["dist"])
                all_triplets.sort(key=lambda x: x[2], reverse=True)
                original_cls = next(k for k, v in MODEL_TO_CAT.items() if v == cat_id)
                stats_emitted[original_cls] += 1
                entry = {"metadata_frame_index": fi, "category_id": cat_id, "score": round(max_score, 5), "segmentation": [all_triplets[:args.n_peaks * 2]]}
                if dists: entry["distance"] = round(sum(dists) / len(dists), 1)
                annotations.append(entry)
    return annotations

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--score_thr", type=float, default=0.35)
    p.add_argument("--nms_iou", type=float, default=0.45)
    p.add_argument("--track_iou", type=float, default=0.30)
    p.add_argument("--min_age", type=int, default=2)
    p.add_argument("--max_missed", type=int, default=2)
    p.add_argument("--n_peaks", type=int, default=20)
    p.add_argument("--dist_scale", type=float, default=DIST_NORM)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args(); set_seed(args.seed)
    if args.checkpoint is None: args.checkpoint = os.path.join(args.exp_dir, "unet_saiseld_best.pth")
    out_dir = Path(args.exp_dir) / "inference_outputs"; out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.exp_dir, "inference.log"), sys.stdout)
    model = UNetSAISELD(n_classes=NUM_CLASSES, in_ch=N_CHANNELS, img_h=IMG_H, img_w=IMG_W)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in state: state = state["model_state_dict"]
    model.load_state_dict(state, strict=True).to(DEVICE).eval()
    test_infos = get_sequence_infos(args.split, LABELS_BASE, FRAMES_BASE)
    for seq_idx, (_, seq_dir, seq_name) in enumerate(test_infos, 1):
        all_ids = sorted(scan_available_frames(seq_dir, seq_name))
        if not all_ids: continue
        res = infer_sequence(model, seq_dir, seq_name, all_ids, args)
        with open(out_dir / f"{seq_name}.json", "w") as f: json.dump({"annotations": res}, f, separators=(",", ":"))