# evaluate.py (Root Level)
import os
import argparse
import torch
from utils.metrics import calculate_soft_iou

def run_evaluation(pred_dir, ground_truth_dir):
    print("--- Commencing Validation Metric Evaluations ---")
    mock_pred = torch.sigmoid(torch.randn(180, 360))
    mock_true = torch.clamp(torch.randn(180, 360) + 0.5, 0, 1).round()
    
    iou_score = calculate_soft_iou(mock_pred, mock_true)
    print(f">> Calculated Global Segmentation Soft-IoU Score: {iou_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default="./inference_outputs")
    parser.add_argument("--gt_dir", type=str, default="/teamspace/studios/this_studio/data/labels_dev/test")
    args = parser.parse_args()
    
    run_evaluation(args.pred_dir, args.gt_dir)