# data/energy_seg_dataset.py
import os
import glob
import pickle
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from data.pipeline import wav_path_from_seq_dir

def scan_available_frames(seq_dir, seq_name):
    return [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in sorted(glob.glob(os.path.join(seq_dir, "*.png")))]

def get_sequence_infos(split, labels_base, frames_base):
    infos = []
    for split_dir in os.listdir(labels_base):
        if split in split_dir:
            for pf in glob.glob(os.path.join(labels_base, split_dir, "*.json")):
                sn = os.path.basename(pf).replace("_std.json", "")
                infos.append((pf, os.path.join(frames_base, split_dir, sn), sn))
    return infos

class EnergySegDataset(Dataset):
    def __init__(self, sequence_infos, frames_base, mic_base, acoustic_extractor, frames_per_epoch=15, img_w=360, img_h=180, dist_norm=500.0, augmentor=None):
        self.samples = sequence_infos; self.frames_base, self.mic_base, self.extractor, self.augmentor = frames_base, mic_base, acoustic_extractor, augmentor
        self.img_w, self.img_h, self.dist_norm = img_w, img_h, dist_norm
        self.all_samples = []; self.class_to_sample_indices = defaultdict(list)
        for idx, (jp, sd, sn) in enumerate(sequence_infos):
            for fi in scan_available_frames(sd, sn)[:frames_per_epoch]: self.all_samples.append((sd, sn, fi, jp))
        self.current_indices = np.arange(len(self.all_samples))

    def reset_epoch(self, balanced=True): np.random.shuffle(self.current_indices)
    def __len__(self): return len(self.current_indices)
    def __getitem__(self, idx):
        sd, sn, fi, jp = self.all_samples[self.current_indices[idx]]
        wp = wav_path_from_seq_dir(sd, self.frames_base, self.mic_base)
        img = self.extractor.get_frame_bands(wp, fi)
        target = {"vmap": torch.zeros(self.img_h, self.img_w, dtype=torch.float32)}
        if self.augmentor: img, target = self.augmentor(img, target)
        return img, target

def collate_fn(batch): return [b[0] for b in batch], [b[1] for b in batch]
def worker_init_fn(wid): pass