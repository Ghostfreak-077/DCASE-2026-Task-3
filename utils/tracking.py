import json
import numpy as np

class ActivitySequenceTracker:
    def __init__(self):
        self.active_tracks = {}

    def log_frame_localization(self, frame_id, class_idx, xyz_coords):
        if class_idx not in self.active_tracks:
            self.active_tracks[class_idx] = []
        self.active_tracks[class_idx].append({
            "frame": frame_id,
            "x": float(xyz_coords[0]),
            "y": float(xyz_coords[1]),
            "z": float(xyz_coords[2])
        })