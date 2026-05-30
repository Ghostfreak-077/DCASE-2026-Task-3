# utils/viz.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CANVAS_W, CANVAS_H = 100, 50
CLASS_NAMES = ["Female speech", "Male speech", "Clapping", "Telephone", "Laughter", "Domestic sounds", "Walk / footsteps", "Door open/close", "Music", "Musical instrument", "Water tap", "Bell", "Knock"]
N_CLASSES = len(CLASS_NAMES)

def plot_ap_curves(results: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results.get("iou_thresholds", [0.5]), results.get("macro_AP_per_thr", [0.5]), marker="o", color="teal")
    ax.set_title("mAP Performance across Tracking Margins"); ax.set_xlabel("IoU Limits"); ax.set_ylabel("Precision Scores")
    plt.savefig(out_path); plt.close()