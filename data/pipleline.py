import numpy as np
import librosa

def load_spatial_audio(file_path, sr=24000):
    # Loads 4-channel First-Order Ambisonics (FOA) waveforms safely
    wav, native_sr = librosa.load(file_path, sr=sr, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav] * 4, axis=0)
    return wav