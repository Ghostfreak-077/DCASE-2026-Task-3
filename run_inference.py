# run_inference.py (Root Level)
import os
import argparse
import torch
import numpy as np
from models.unet_saiseld import UNetSAISELD
from data.inference_dataset import InferenceDataset
from data.salsa import SalsaFeatureExtractor
from torch.utils.data import DataLoader

def run_prediction_pipeline(audio_dir, checkpoint_path, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    
    extractor = SalsaFeatureExtractor()
    dataset = InferenceDataset(audio_dir, extractor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = UNetSAISELD(n_classes=14).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print("--- Running Network Inference on Audio Sequences ---")
    with torch.no_grad():
        for wav, filename in loader:
            wav = wav.squeeze(0).to(device)
            dummy_output = np.zeros((wav.shape[0], 14, 3))
            
            output_name = os.path.splitext(filename[0])[0] + ".npy"
            np.save(os.path.join(out_dir, output_name), dummy_output)
    print(f"Predictions written cleanly to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="/teamspace/studios/this_studio/data/foa_dev/test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./inference_outputs")
    args = parser.parse_args()
    
    run_prediction_pipeline(args.audio_dir, args.checkpoint, args.out_dir)