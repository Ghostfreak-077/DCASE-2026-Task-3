# data/pipeline.py
import os
import math
from collections import OrderedDict
import numpy as np
import soundfile as sf
import torch
from data.salsa import SalsaLiteCore

def wav_path_from_seq_dir(seq_dir: str, frames_base: str, mic_base: str) -> str:
    return os.path.join(mic_base, os.path.relpath(seq_dir, frames_base) + ".wav")

class SalsaFeatureExtractor:
    def __init__(self, fs=24000, n_fft=512, hop_length=300, fmin_doa=40.0, fmax_doa=6000.0, fmax_spec=6000.0, ref_mic=0, context_frames=4, audio_cache_size=32, frame_cache_size=512):
        self.core = SalsaLiteCore(fs, n_fft, hop_length, fmin_doa, fmax_doa, fmax_spec, ref_mic)
        self.fs, self.hop_length, self.context_frames = fs, hop_length, context_frames
        self.frame_samples = fs // 10
        self.target_frames = max(1, round(self.frame_samples * (1 + context_frames) / hop_length))
        self._audio_cache, self._feat_cache = OrderedDict(), OrderedDict()
        self._audio_cache_size, self._frame_cache_size = audio_cache_size, frame_cache_size
        self.n_channels, self.n_time, self.freq_bins = 4, round(self.frame_samples / hop_length) + 1, self.core.freq_bins

    def get_frame_bands(self, wav_path: str, frame_idx: int) -> torch.Tensor:
        key = (wav_path, frame_idx)
        if key in self._feat_cache:
            self._feat_cache.move_to_end(key); return self._feat_cache[key].clone()
        if wav_path not in self._audio_cache:
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim == 1: audio = audio[:, np.newaxis]
            self._audio_cache[wav_path] = audio
            if len(self._audio_cache) > self._audio_cache_size: self._audio_cache.popitem(last=False)
        audio = self._audio_cache[wav_path]
        start = max(0, int(frame_idx * self.frame_samples - (self.context_frames * self.frame_samples // 2)))
        end = min(audio.shape[0], start + int((1 + self.context_frames) * self.frame_samples))
        chunk = np.zeros((int((1 + self.context_frames) * self.frame_samples), audio.shape[1]), dtype=np.float32)
        chunk[:end-start] = audio[start:end]
        feats = self.core(chunk, self.target_frames)[:, :self.n_time, :]
        if feats.shape[1] < self.n_time:
            feats = np.pad(feats, ((0,0), (0, self.n_time - feats.shape[1]), (0,0)))
        tensor = torch.from_numpy(feats)
        self._feat_cache[key] = tensor
        if len(self._feat_cache) > self._frame_cache_size: self._feat_cache.popitem(last=False)
        return tensor.clone()

    def clear_cache(self): self._audio_cache.clear(); self._feat_cache.clear()
    @property
    def feature_shape(self): return (self.n_channels, self.n_time, self.freq_bins)