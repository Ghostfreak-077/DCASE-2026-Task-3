import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

def scan_available_frames(seq_dir, seq_name):
    return [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in sorted(glob.glob(os.path.join(seq_dir, "*.png")))]

def wav_path_from_seq_dir(seq_dir, frames_base, mic_base):
    rel_path = os.path.relpath(seq_dir, frames_base)
    return os.path.join(mic_base, rel_path + ".wav")

def get_sequence_infos(split, labels_base, frames_base):
    infos = []
    labels_dir = os.path.join(labels_base, split)
    if not os.path.exists(labels_dir):
        return infos
    for jp in sorted(glob.glob(os.path.join(labels_dir, "*.json"))):
        sn = os.path.splitext(os.path.basename(jp))[0]
        sd = os.path.join(frames_base, split, sn)
        if os.path.exists(sd):
            infos.append((sd, sn, jp))
    return infos

class EnergySegDataset(Dataset):
    def __init__(self, sequence_infos, frames_base, mic_base, acoustic_extractor, frames_per_epoch=15, img_w=360, img_h=180, dist_norm=500.0, augmentor=None):
        self.frames_base = frames_base
        self.mic_base = mic_base
        self.extractor = acoustic_extractor
        self.frames_per_epoch = frames_per_epoch
        self.img_w, self.img_h = img_w, img_h
        self.dist_norm = dist_norm
        self.augmentor = augmentor
        
        self.all_samples = []
        self.class_to_sample_indices = defaultdict(list)
        
        for idx, (sd, sn, jp) in enumerate(sequence_infos):
            for fi in scan_available_frames(sd, sn)[:frames_per_epoch]:
                self.all_samples.append((sd, sn, fi, jp))
                
        self.current_indices = np.arange(len(self.all_samples))

    def reset_epoch(self, balanced=True):
        np.random.shuffle(self.current_indices)

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        sd, sn, fi, jp = self.all_samples[self.current_indices[idx]]
        wp = wav_path_from_seq_dir(sd, self.frames_base, self.mic_base)
        
        # Load and extract spatial multichannel audio features
        from .pipeline import load_spatial_audio
        wav = load_spatial_audio(wp)
        img = self.extractor.get_frame_bands(wav, fi)
        
        if self.augmentor:
            img = self.augmentor.apply_channel_swapping(img)
            
        target = {"vmap": torch.zeros(self.img_h, self.img_w, dtype=torch.float32)}
        return img, target

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = {k: torch.stack([b[1][k] for b in batch], dim=0) for k in batch[0][1].keys()}
    return images, targets

def worker_init_fn(wid):
    np.random.seed(torch.initial_seed() % 2**32 + wid)