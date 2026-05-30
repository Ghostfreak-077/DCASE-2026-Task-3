# utils/metrics.py
import math
import numpy as np

def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1, iy1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    ix2, iy2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    return inter / ((box[2]-box[0])*(box[3]-box[1]) + (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]) - inter + 1e-6)

def nms_per_class(boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, iou_thr=0.45) -> np.ndarray:
    keep = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        order = np.argsort(-scores[idx])
        alive = np.ones(len(order), dtype=bool)
        for i, oi in enumerate(order):
            if alive[i]:
                keep.append(idx[oi])
                alive[i+1:] = alive[i+1:] & (box_iou(boxes[idx[oi]], boxes[idx[order[i+1:]]]) <= iou_thr)
    return np.array(keep, dtype=int)

def extract_peaks(emap_raw: np.ndarray, box_xyxy: np.ndarray, n_peaks=20) -> list:
    xi0, yi0, xi1, yi1 = max(0, int(box_xyxy[0])), max(0, int(box_xyxy[1])), min(360, int(box_xyxy[2])), min(180, int(box_xyxy[3]))
    if xi1 <= xi0 or yi1 <= yi0: return [[0.0, 0.0, float(emap_raw.max())]]
    region = emap_raw[yi0:yi1, xi0:xi1]; norm = (region - region.min()) / (region.max() - region.min() + 1e-8)
    flat = norm.ravel(); top_idx = np.argsort(-flat)[:n_peaks]
    return [[float(xi0 + i % (xi1-xi0)), float(yi0 + i // (xi1-xi0)), float(flat[i])] for i in top_idx]

def mask_soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    return float(np.minimum(a, b).sum() / (np.maximum(a, b).sum() + 1e-8))

def mask_pearson(gt: np.ndarray, pred: np.ndarray) -> float:
    if gt is None or pred is None or gt.sum() == 0: return 0.0
    m = gt > 0
    return float(np.corrcoef(gt[m], pred[m])[0, 1] if np.std(pred[m]) > 1e-6 else 0.0)