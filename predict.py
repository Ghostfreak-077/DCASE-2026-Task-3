import argparse
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.unet_saiseld import UNetSAISELD
from data.energy_seg_dataset import scan_available_frames
from data.inference_dataset import build_audio_inference_loader
from utils.metrics import nms_per_class, extract_peaks
from utils.tracking import TemporalTracker

FRAMES_BASE = "/gpfs/scratch/eez086/STARSS23/frames_dev"
MIC_BASE = "/gpfs/scratch/eez086/STARSS23/mic_dev"
IMG_W, IMG_H = 360, 180
NUM_CLASSES = 14
MODEL_TO_CAT = {k: k - 1 for k in range(1, NUM_CLASSES)} 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_sequence(model, sd: str, sn: str, args) -> list:
    fis = sorted(scan_available_frames(sd, sn))
    loader = build_audio_inference_loader(sd, sn, fis, args.batch_size, args.num_workers, FRAMES_BASE, MIC_BASE)
    tracker = TemporalTracker(iou_thr=args.track_iou, min_age=args.min_age, max_missed=args.max_missed)
    annotations = []

    frame_data_accum = []; class_scores = {c: [] for c in range(1, NUM_CLASSES)}
    with torch.no_grad():
        for bf, bt in tqdm(loader, desc=f"  {sn[:40]} [Infer]"):
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
    for fi, labels_np, scores_np, boxes_np, emaps_np, dist_np in frame_data_accum:
        if labels_np is None:
            tracker.update(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), np.zeros((0, 180, 360)), None); continue
        keep_mask = np.zeros(len(scores_np), dtype=bool)
        for i in range(len(scores_np)):
            c, raw_s = labels_np[i], scores_np[i]
            if c > 0 and c in class_min_max:
                cmin, cmax = class_min_max[c]
                scaled_s = 0.05 + 0.90 * ((raw_s - cmin) / (cmax - cmin)) if (cmax >= 0.08 and cmax > cmin) else raw_s
                if scaled_s >= args.score_thr: keep_mask[i] = True

        keep = keep_mask & (labels_np > 0)
        if keep.sum() == 0:
            tracker.update(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), np.zeros((0, 180, 360)), None); continue

        labels_f, scores_f, boxes_f = labels_np[keep], scores_np[keep], boxes_np[keep]
        emaps_f = emaps_np[keep] if emaps_np is not None else np.zeros((keep.sum(), 180, 360))
        dist_f = dist_np[keep] if dist_np is not None else None

        nms_idx = nms_per_class(boxes_f, labels_f, scores_f, iou_thr=args.nms_iou)
        confirmed = tracker.update(boxes_f[nms_idx], labels_f[nms_idx], scores_f[nms_idx], emaps_f[nms_idx], dist_f[nms_idx] if dist_f is not None else None)

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
                cx, cy = (frag['box'][0] + frag['box'][2]) / 2.0, (frag['box'][1] + frag['box'][3]) / 2.0
                placed = False
                for clus in clusters:
                    for c_frag in clus:
                        ccx, ccy = (c_frag['box'][0] + c_frag['box'][2]) / 2.0, (c_frag['box'][1] + c_frag['box'][3]) / 2.0
                        if np.hypot(cx - ccx, cy - ccy) < 50.0: clus.append(frag); placed = True; break
                    if placed: break
                if not placed: clusters.append([frag])

            for clus in clusters:
                all_triplets, dists, max_score = [], [], -1.0
                for frag in clus:
                    all_triplets.extend(frag['triplets'])
                    if frag['score'] > max_score: max_score = frag['score']
                    if frag['dist'] is not None: dists.append(frag['dist'])
                all_triplets.sort(key=lambda x: x[2], reverse=True)
                entry = {"metadata_frame_index": fi, "category_id": cat_id, "score": round(max_score, 5), "segmentation": [all_triplets[:args.n_peaks * 2]]}
                if dists: entry["distance"] = round(sum(dists) / len(dists), 1)
                annotations.append(entry)
    return annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="submission_output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--score_thr", type=float, default=0.65)
    parser.add_argument("--nms_iou", type=float, default=0.45)
    parser.add_argument("--track_iou", type=float, default=0.30)
    parser.add_argument("--min_age", type=int, default=2)
    parser.add_argument("--max_missed", type=int, default=2)
    parser.add_argument("--n_peaks", type=int, default=20)
    parser.add_argument("--dist_scale", type=float, default=1000.0)
    args = parser.parse_args(); out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = UNetSAISELD(n_classes=NUM_CLASSES, in_ch=4, img_w=360, img_h=180).to(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in state: state = state["model_state_dict"]
    model.load_state_dict(state, strict=True); model.eval()
    base_folders = [d for d in os.listdir(FRAMES_BASE) if os.path.isdir(os.path.join(FRAMES_BASE, d))]
    for sn in base_folders:
        res = infer_sequence(model, os.path.join(FRAMES_BASE, sn), sn, args)
        with open(out_dir / f"{sn}.json", "w") as f: json.dump({"annotations": res}, f, separators=(",", ":"))