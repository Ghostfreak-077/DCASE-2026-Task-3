import numpy as np
import librosa

class SalsaFeatureExtractor:
    def __init__(self, sample_rate=24000, n_fft=512, hop_length=300):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length

    def get_frame_bands(self, wav_channels, frame_idx, context_window=5):
        # Extracts spatialized multichannel linear spectrogram features around a target context frame
        features = []
        for ch in range(wav_channels.shape[0]):
            stft = librosa.stft(wav_channels[ch], n_fft=self.n_fft, hop_length=self.hop)
            mag = np.abs(stft)
            
            # Pad boundary sequence if context boundaries exceed dimensions
            start = max(0, frame_idx - context_window)
            end = min(mag.shape[1], frame_idx + context_window + 1)
            chunk = mag[:, start:end]
            
            if chunk.shape[1] < (context_window * 2 + 1):
                pad_width = (context_window * 2 + 1) - chunk.shape[1]
                chunk = np.pad(chunk, ((0,0), (0, pad_width)), mode='edge')
            features.append(chunk)
            
        return torch.tensor(np.stack(features, axis=0), dtype=torch.float32)