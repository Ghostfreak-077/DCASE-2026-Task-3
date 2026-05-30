# utils/tracking.py
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from utils.metrics import box_iou

@dataclass
class Track:
    track_id: int; label: int; score: float; box: np.ndarray; emap: np.ndarray; dist_pred: Optional[float]
    age: int = 1; missed: int = 0; confirmed: bool = False

class TemporalTracker:
    def __init__(self, iou_thr=0.30, min_age=2, max_missed=2):
        self.iou_thr, self.min_age, self.max_missed = iou_thr, min_age, max_missed
        self._tracks: List[Track] = []; self._next_id = 0
    def reset(self): self._tracks = []; self._next_id = 0
    def update(self, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, emaps: np.ndarray, dist_preds: Optional[np.ndarray]) -> List[Track]:
        n_det = len(labels); unmatched_dets = list(range(n_det)); matched_track_i = set()
        if self._tracks and n_det > 0:
            for di in range(n_det):
                ious = box_iou(boxes[di], np.stack([t.box for t in self._tracks]))
                for ti, t in enumerate(self._tracks):
                    if ious[ti] >= self.iou_thr and t.label == labels[di] and ti not in matched_track_i:
                        t.box, t.score, t.emap = boxes[di], scores[di], emaps[di]
                        t.age += 1; t.missed = 0
                        if t.age >= self.min_age: t.confirmed = True
                        unmatched_dets.remove(di); matched_track_i.add(ti); break
        for ti, t in enumerate(self._tracks):
            if ti not in matched_track_i: t.missed += 1
        for di in unmatched_dets:
            self._tracks.append(Track(self._next_id, int(labels[di]), float(scores[di]), boxes[di].copy(), emaps[di].copy(), float(dist_preds[di]) if dist_preds is not None else None))
            self._next_id += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]
        return [t for t in self._tracks if t.confirmed]