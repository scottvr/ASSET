import numpy as np
import librosa
import soundfile as sf
from .types import AudioArray, SpectrogramArray, FrequencyBands
from .math_utils import magnitude, angle, phase_difference, phase_coherence, rms

def to_mono(audio: AudioArray) -> AudioArray:
    """Convert audio to mono if stereo"""
    if len(audio.shape) == 1:
        return audio
    return librosa.to_mono(audio.T)

def create_spectrogram(audio: AudioArray, **stft_params) -> SpectrogramArray:
    """Create spectrogram with standard parameters"""
    return librosa.stft(audio, **stft_params)

def get_frequency_bins(sr: int, n_fft: int) -> np.ndarray:
    """Get frequency bins for STFT"""
    return librosa.fft_frequencies(sr=sr, n_fft=n_fft)

def get_band_mask(freq_bins: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
    """Get boolean mask for frequency band"""
    return (freq_bins >= low_freq) & (freq_bins <= high_freq)
