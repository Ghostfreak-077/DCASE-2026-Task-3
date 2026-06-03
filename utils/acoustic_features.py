"""
acoustic_features.py
--------------------
SALSA-Lite feature extractor for DCASE 2026 Task 3 Track A.
Replaces the UpLAM-based AcousticFeatureExtractor from the baseline.

Output per frame: Tensor of shape (N_CH, T_FRAMES, FREQ_BINS)
  N_CH      = 4  (1 log-mel ref-mic + 3 normalised inter-channel phase vectors)
  T_FRAMES  = ceil(frame_samples / hop_length)  [~8 at 24kHz, hop=300]
  FREQ_BINS = floor(fmax_spec * n_fft / fs) - floor(fmin_doa * n_fft / fs)  [128]

All computation is CPU-side (called from DataLoader workers).
Results are LRU-cached per (wav_path, frame_idx) to avoid re-computing.
"""

import os
import re
import math
from collections import OrderedDict
from typing import Optional

import numpy as np
import librosa
import soundfile as sf
import torch


# ---------------------------------------------------------------------------
# Path helper (kept from baseline for DataLoader compatibility)
# ---------------------------------------------------------------------------
def wav_path_from_seq_dir(seq_dir: str, frames_base: str, mic_base: str) -> str:
    """Convert a frames directory path to the corresponding WAV path."""
    rel = os.path.relpath(seq_dir, frames_base)
    return os.path.join(mic_base, rel + ".wav")


# ---------------------------------------------------------------------------
# SALSA-Lite core (vectorised, CPU, pure numpy)
# ---------------------------------------------------------------------------
class _SalsaLiteCore:
    """
    Computes SALSA-Lite features from raw multi-channel audio.

    Features = [log-mel of ref mic (ch 0)]  +  [NIPV for ch 1,2,3 vs ch 0]

    NIPV (Normalised Inter-channel Phase Vector):
        nipv[k] = angle(X[0].conj() * X[k]) / (norm_freq * delta)
    where delta = 2*pi*fs / (n_fft * c).

    Output shape: (N_CH, n_time, freq_bins)  [channels-first, time x freq]
    """

    SOUND_SPEED = 343.0
    F_DTYPE = np.float32

    def __init__(
        self,
        fs: int = 24000,
        n_fft: int = 512,
        hop_length: int = 300,
        fmin_doa: float = 40.0,
        fmax_doa: float = 6000.0,
        fmax_spec: float = 6000.0,
        ref_mic: int = 0,
    ):
        self.fs         = fs
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.ref_mic    = ref_mic

        n_bins = n_fft // 2 + 1
        self.lo_bin  = int(math.floor(fmin_doa  * n_fft / float(fs)))
        self.up_bin  = int(math.floor(fmax_doa  * n_fft / float(fs)))
        self.cut_bin = int(math.floor(fmax_spec * n_fft / float(fs)))

        # Normalisation vector: shape (freq_bins, 1)
        delta = 2.0 * math.pi * fs / (n_fft * self.SOUND_SPEED)
        norm  = np.arange(n_bins, dtype=self.F_DTYPE)[:, None] * delta
        norm[0, 0] = 1.0  # avoid division by zero at DC
        self.norm_freq = norm[self.lo_bin : self.cut_bin]  # (freq_bins, 1)

        self.freq_bins = self.cut_bin - self.lo_bin  # 128

    # ------------------------------------------------------------------
    def _stft(self, wav_ch: np.ndarray, target_frames: int) -> np.ndarray:
        """Compute STFT for one channel, interpolated to target_frames."""
        S = librosa.stft(
            y          = np.asfortranarray(wav_ch),
            n_fft      = self.n_fft,
            hop_length  = self.hop_length,
            center     = True,
            window     = "hann",
            pad_mode   = "reflect",
        )  # (n_bins, n_frames_raw)
        S = S.T  # (n_frames_raw, n_bins)

        # Interpolate to target_frames (matches baseline interp_tensor logic)
        if S.shape[0] != target_frames:
            x_old = np.linspace(0, 1, S.shape[0])
            x_new = np.linspace(0, 1, target_frames)
            S_real = np.stack(
                [np.interp(x_new, x_old, S[:, b].real) for b in range(S.shape[1])], axis=1
            )
            S_imag = np.stack(
                [np.interp(x_new, x_old, S[:, b].imag) for b in range(S.shape[1])], axis=1
            )
            S = S_real + 1j * S_imag

        return S.T  # (n_bins, target_frames)

    def _gcc_phat(self, sig_i: np.ndarray, sig_j: np.ndarray, max_tau: int) -> np.ndarray:
        n = self.n_fft
        Si = np.fft.rfft(sig_i, n=n)
        Sj = np.fft.rfft(sig_j, n=n)
        R  = Si * np.conj(Sj)
        R /= (np.abs(R) + 1e-10)
        cc = np.fft.irfft(R, n=n)
        cc = np.concatenate([cc[-max_tau:], cc[:max_tau+1]])  # (2*max_tau+1,)
        return cc.astype(self.F_DTYPE)

    # ------------------------------------------------------------------
    def __call__(self, audio: np.ndarray, target_frames: int) -> np.ndarray:
        """
        Parameters
        ----------
        audio : (n_samples, n_channels) float32
        target_frames : int  — desired time dimension

        Returns
        -------
        features : (N_CH, target_frames, freq_bins)  float32
        """
        audio = audio.T.astype(self.F_DTYPE)  # (n_ch, n_samples)
        n_ch  = audio.shape[0]

        # Compute STFT for all channels
        stfts = np.stack(
            [self._stft(audio[c], target_frames) for c in range(n_ch)], axis=0
        )  # (n_ch, n_bins, target_frames)

        # Clip to [lo_bin, cut_bin]
        stfts = stfts[:, self.lo_bin : self.cut_bin, :]  # (n_ch, freq_bins, T)

        ref = self.ref_mic

        # --- Log-mel spectrogram of reference mic ---
        spec     = np.abs(stfts[ref]) ** 2  # (freq_bins, T)
        log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
        # shape: (1, T, freq_bins)
        log_spec = log_spec.T[np.newaxis, :, :]  # (1, T, freq_bins)

        # --- NIPV for non-ref mics ---
        nipv_list = []
        for c in range(n_ch):
            if c == ref:
                continue
            phase = np.angle(stfts[ref].conj() * stfts[c])  # (freq_bins, T)
            phase = phase / self.norm_freq                     # normalise
            # Zero out potentially aliasing bins above up_bin
            up = self.up_bin - self.lo_bin
            if up < self.freq_bins:
                phase[up:] = 0.0
            nipv_list.append(phase.T[np.newaxis, :, :])  # (1, T, freq_bins)

        nipv = np.concatenate(nipv_list, axis=0)  # (n_ch-1, T, freq_bins)

        features = np.concatenate([log_spec, nipv], axis=0)  # (4, T, freq_bins)

        # --- GCC-PHAT across all mic pairs ---
        max_tau = self.n_fft // 2
        pairs = [(i, j) for i in range(n_ch) for j in range(i+1, n_ch)]
        gcc_list = []
        for (i, j) in pairs:
            cc = self._gcc_phat(audio[i], audio[j], max_tau)  # (2*max_tau+1,)
            cc_interp = np.interp(
                np.linspace(0, 1, self.freq_bins),
                np.linspace(0, 1, len(cc)),
                cc
            )
            gcc_frame = np.tile(cc_interp, (target_frames, 1))[np.newaxis, :, :]  # (1, T, freq_bins)
            gcc_list.append(gcc_frame)

        gcc = np.concatenate(gcc_list, axis=0)  # (6, T, freq_bins)
        features = np.concatenate([features, gcc], axis=0)  # (10, T, freq_bins)
        return features.astype(self.F_DTYPE)


# ---------------------------------------------------------------------------
# Per-frame feature extractor with LRU cache
# ---------------------------------------------------------------------------
class SalsaFeatureExtractor:
    """
    Wraps _SalsaLiteCore to provide per-frame extraction with:
      - Whole-file audio cache (avoid re-reading WAV)
      - Per-frame feature LRU cache
      - Same public API as the baseline AcousticFeatureExtractor

    Parameters
    ----------
    fs           : sample rate (must match audio files)
    n_fft        : STFT window size
    hop_length   : STFT hop
    fmin_doa     : lower DOA frequency for NIPV
    fmax_doa     : upper DOA frequency for NIPV (aliasing cutoff)
    fmax_spec    : upper frequency for log-mel
    ref_mic      : reference microphone index (0-based)
    context_frames : how many audio frames to window around the target frame
                     (provides temporal context without adding latency)
                     0 = single frame; 2 = ±1 neighbour; 4 = ±2 neighbours
    audio_cache_size : max number of full WAV files to keep in RAM
    frame_cache_size : max number of per-frame tensors to cache
    """

    def __init__(
        self,
        fs: int                = 24000,
        n_fft: int             = 512,
        hop_length: int        = 300,
        fmin_doa: float        = 40.0,
        fmax_doa: float        = 6000.0,
        fmax_spec: float       = 6000.0,
        ref_mic: int           = 0,
        context_frames: int    = 4,
        audio_cache_size: int  = 32,
        frame_cache_size: int  = 512,
    ):
        self.core = _SalsaLiteCore(
            fs=fs, n_fft=n_fft, hop_length=hop_length,
            fmin_doa=fmin_doa, fmax_doa=fmax_doa, fmax_spec=fmax_spec,
            ref_mic=ref_mic,
        )
        self.fs              = fs
        self.hop_length      = hop_length
        self.context_frames  = context_frames
        self.frame_samples   = fs // 10            # 2400 samples @ 10 FPS
        self.target_frames   = max(
            1, round(self.frame_samples * (1 + context_frames) / hop_length)
        )

        self._audio_cache: OrderedDict = OrderedDict()
        self._feat_cache:  OrderedDict = OrderedDict()
        self._audio_cache_size = audio_cache_size
        self._frame_cache_size = frame_cache_size

        # Expose feature shape for downstream modules
        self.n_channels  = 10                      # 1 log-mel + 3 NIPV
        self.n_time      = round(self.frame_samples / hop_length) + 1   # ~8
        self.freq_bins   = self.core.freq_bins     # 128

    # ------------------------------------------------------------------
    def _load_audio(self, wav_path: str) -> np.ndarray:
        """Load and cache full WAV file. Returns (n_samples, n_ch) float32."""
        if wav_path in self._audio_cache:
            self._audio_cache.move_to_end(wav_path)
            return self._audio_cache[wav_path]

        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        audio, file_sr = sf.read(wav_path, dtype="float32")
        if file_sr != self.fs:
            raise ValueError(
                f"Sample rate mismatch: file={file_sr}, expected={self.fs}"
            )
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]

        self._audio_cache[wav_path] = audio
        if len(self._audio_cache) > self._audio_cache_size:
            self._audio_cache.popitem(last=False)
        return audio

    # ------------------------------------------------------------------
    def get_frame_bands(self, wav_path: str, frame_idx: int) -> torch.Tensor:
        """
        Extract SALSA-Lite features for a single frame.

        Parameters
        ----------
        wav_path  : path to 4-channel WAV file
        frame_idx : 0-based frame index at 10 FPS

        Returns
        -------
        Tensor of shape (N_CH, T, F)  float32  on CPU
          N_CH = 4
          T    = ~8  (single-frame time bins)
          F    = 128 (frequency bins)
        """
        cache_key = (wav_path, frame_idx)
        if cache_key in self._feat_cache:
            self._feat_cache.move_to_end(cache_key)
            return self._feat_cache[cache_key].clone()

        audio = self._load_audio(wav_path)  # (n_samples, n_ch)
        n_samples = audio.shape[0]

        half_ctx    = self.context_frames // 2
        # Centre-aligned window around the target frame
        centre_sample = int(frame_idx * self.frame_samples + self.frame_samples // 2)
        half_window   = int((1 + self.context_frames) * self.frame_samples // 2)

        start = max(0, centre_sample - half_window)
        end   = min(n_samples, centre_sample + half_window)

        # Pad with zeros if at clip boundaries
        chunk = np.zeros(
            (int((1 + self.context_frames) * self.frame_samples), audio.shape[1]),
            dtype=np.float32,
        )
        src_len  = end - start
        dst_off  = half_window - (centre_sample - start)
        chunk[dst_off : dst_off + src_len] = audio[start:end]

        # SALSA-Lite features on the windowed chunk
        # target_frames is the number of STFT frames expected
        feats = self.core(chunk, target_frames=self.target_frames)
        # feats: (N_CH, total_t_frames, freq_bins)

        # Crop to single-frame T by taking the central T time bins
        total_t = feats.shape[1]
        t_half  = self.n_time // 2
        t_start = max(0, total_t // 2 - t_half)
        t_end   = t_start + self.n_time
        if t_end > total_t:
            t_end   = total_t
            t_start = max(0, t_end - self.n_time)
        feats = feats[:, t_start:t_end, :]  # (N_CH, n_time, freq_bins)

        # Pad time dim if still short (clip boundary edge case)
        if feats.shape[1] < self.n_time:
            pad = np.zeros(
                (feats.shape[0], self.n_time - feats.shape[1], feats.shape[2]),
                dtype=np.float32,
            )
            feats = np.concatenate([feats, pad], axis=1)

        tensor = torch.from_numpy(feats)  # (4, n_time, freq_bins)

        self._feat_cache[cache_key] = tensor
        if len(self._feat_cache) > self._frame_cache_size:
            self._feat_cache.popitem(last=False)

        return tensor.clone()

    # ------------------------------------------------------------------
    def clear_cache(self):
        self._audio_cache.clear()
        self._feat_cache.clear()

    # ------------------------------------------------------------------
    @property
    def feature_shape(self):
        """(N_CH, T, F) — input shape expected by the model encoder."""
        return (self.n_channels, self.n_time, self.freq_bins)


# ---------------------------------------------------------------------------
# Acoustic Channel Swap (ACS) augmentation helper
# ---------------------------------------------------------------------------
def acoustic_channel_swap(feat: torch.Tensor) -> torch.Tensor:
    """
    Swap left↔right microphone pairs to create a mirrored spatial view.
    For a 4-channel tetrahedral array (M1-M4), swapping M1↔M3 and M2↔M4
    is equivalent to a left-right mirror.

    feat : (N_CH, T, F) tensor where:
           ch 0 = log-mel ref mic (ch 0 of audio)
           ch 1 = NIPV ch1 vs ch0
           ch 2 = NIPV ch2 vs ch0
           ch 3 = NIPV ch3 vs ch0

    For tetrahedral left-right swap:
      ch1 (front-left)  <-> ch2 (back-right)   => negate phase
      ch3 (back-left) sign flip

    Practical implementation: negate all NIPV channels (sign flip of phase
    = left-right mirror in the phase domain). Log-mel is symmetric so unchanged.
    This is the standard ACS used in DCASE 2024 top Track A systems.
    """
    out = feat.clone()
    out[1:] = -out[1:]   # negate all NIPV channels
    return out