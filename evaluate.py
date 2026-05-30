import argparse
import json
import os
from pathlib import Path
import numpy as np

from utils.metrics import mask_soft_iou, mask_pearson
from utils.viz import plot_ap_curves, CLASS_NAMES, N_CLASSES, CANVAS_W, CANVAS_H

class EvalAccumulator:
    def __init__(self):
        self.cls_records = defaultdict(list)
        self.cls_n_gt = defaultdict(int)
        self.micro_records = []
        self.micro_n_gt = 0
        self.frame_iou = defaultdict(lambda: defaultdict(dict))

def render_poly(triplets) -> np.ndarray:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    for x, y, v in triplets:
        xi, yi = int(round(x * (CANVAS_W/360))), int(round(y * (CANVAS_H/180)))
        if 0 <= xi < CANVAS_W and 0 <= yi < CANVAS_H: canvas[yi, xi] = max(canvas[yi, xi], v)
    return canvas

def render_annotation(annot) -> np.ndarray:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    for poly in annot.get("segmentation", []):
        if poly: canvas = np.maximum(canvas, render_poly(poly))
    if canvas.max() > 1e-8: canvas /= canvas.max()
    return canvas

def match_frame(gt_annots, pr_annots, iou_thresholds) -> list:
    records = []
    for pr in pr_annots:
        pm = render_annotation(pr)
        best_iou = 0.0; best_gt = None
        for gt in gt_annots:
            gm = render_annotation(gt)
            iou = mask_soft_iou(pm, gm)
            if iou > best_iou: best_iou = iou; best_gt = gm
        records.append({"score": pr["score"], "tp": best_iou >= iou_thresholds, "fn": False, "iou": best_iou, "pearson": mask_pearson(best_gt, pm) if best_gt is not None else 0.0})
    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output_dir", default="eval_output")
    args = parser.parse_args(); out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    acc = EvalAccumulator(); thrs = np.array([0.25, 0.5, 0.75])
    for pf in sorted(Path(args.pred_dir).glob("*.json")):
        with open(pf) as f: p_ann = json.load(f).get("annotations", [])
        with open(Path(args.gt_dir) / pf.name) as f: g_ann = json.load(f).get("annotations", [])
        for fi in set([a["metadata_frame_index"] for a in p_ann + g_ann]):
            p_f = [a for a in p_ann if a["metadata_frame_index"] == fi]
            g_f = [a for a in g_ann if a["metadata_frame_index"] == fi]
            acc.micro_records.extend(match_frame(g_f, p_f, thrs))
    print("[DONE] Evaluation execution metrics compiled.")