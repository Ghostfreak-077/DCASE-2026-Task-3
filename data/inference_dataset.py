import os
from torch.utils.data import Dataset
import numpy as np
from .pipeline import load_spatial_audio

class InferenceDataset(Dataset):
    def __init__(self, audio_dir, extractor):
        self.audio_dir = audio_dir
        self.extractor = extractor
        self.files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = os.path.join(self.audio_dir, self.files[idx])
        wav = load_spatial_audio(fp)
        return wav, self.files[idx]