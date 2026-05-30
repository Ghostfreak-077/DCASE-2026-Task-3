# data/salsa.py
import math
import numpy as np
import librosa

class SalsaLiteCore:
    SOUND_SPEED = 343.0
    def __init__(self, fs=24000, n_fft=512, hop_length=300, fmin_doa=40.0, fmax_doa=6000.0, fmax_spec=6000.0, ref_mic=0):
        self.fs, self.n_fft, self.hop_length, self.ref_mic = fs, n_fft, hop_length, ref_mic
        self.lo_bin, self.cut_bin = int(math.floor(fmin_doa * n_fft / fs)), int(math.floor(fmax_spec * n_fft / fs))
        self.freq_bins = self.cut_bin - self.lo_bin
        self.norm_freq = (np.arange(n_fft // 2 + 1, dtype=np.float32)[:, None] * (2.0 * math.pi * fs / (n_fft * self.SOUND_SPEED)))[self.lo_bin:self.cut_bin]
        self.norm_freq[0, 0] = 1.0

    def __call__(self, audio: np.ndarray, target_frames: int) -> np.ndarray:
        stfts = np.stack([librosa.stft(audio[:, c], n_fft=self.n_fft, hop_length=self.hop_length, center=True).T for c in range(audio.shape[1])], axis=0)[:, self.lo_bin:self.cut_bin, :]
        log_spec = librosa.power_to_db(np.abs(stfts[self.ref_mic])**2, ref=1.0)[np.newaxis, :, :]
        nipv = []
        for c in range(audio.shape[1]):
            if c != self.ref_mic: nipv.append((np.angle(stfts[self.ref_mic].conj() * stfts[c]) / self.norm_freq)[np.newaxis, :, :])
        return np.concatenate([log_spec, np.concatenate(nipv, axis=0)], axis=0).transpose(0, 2, 1).astype(np.float32)