import numpy as np

def calculate_soft_iou(pred_mask, true_mask, eps=1e-6):
    # Calculates the smooth Intersection-over-Union segmentation coefficient for energy mapping profiles
    inter = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - inter
    return (inter + eps) / (union + eps)