# data/inference_dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from data.pipeline import SalsaFeatureExtractor

class AudioOnlyInferenceDataset(Dataset):
    def __init__(self, seq_dir, seq_name, frame_indices, frames_base, mic_base):
        self.frame_indices = frame_indices
        self.extractor = SalsaFeatureExtractor()
        self.wav_path = os.path.join(mic_base, os.path.relpath(seq_dir, frames_base) + ".wav")
    def __len__(self): return len(self.frame_indices)
    def __getitem__(self, idx):
        fi = self.frame_indices[idx]
        return fi, self.extractor.get_frame_bands(self.wav_path, fi)

def build_audio_inference_loader(seq_dir, seq_name, frame_indices, batch_size, num_workers, frames_base, mic_base):
    return DataLoader(AudioOnlyInferenceDataset(seq_dir, seq_name, frame_indices, frames_base, mic_base), batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=lambda b: ([x[0] for x in b], [x[1] for x in b]))