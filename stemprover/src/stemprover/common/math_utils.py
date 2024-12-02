import numpy as np
from .types import AudioArray, SpectrogramArray

def angle(complex_spec: SpectrogramArray) -> SpectrogramArray:
    """Get phase angle from complex spectrogram"""
    return np.angle(complex_spec)

def magnitude(complex_spec: SpectrogramArray) -> SpectrogramArray:
    """Get magnitude from complex spectrogram"""
    return np.abs(complex_spec)

def phase_difference(spec1: SpectrogramArray, spec2: SpectrogramArray) -> SpectrogramArray:
    """Compute phase difference between spectrograms"""
    return np.abs(angle(spec1) - angle(spec2))

def phase_coherence(phase_diff: SpectrogramArray) -> float:
    """Compute phase coherence from phase difference"""
    return float(np.mean(np.cos(phase_diff)))

def rms(array: AudioArray) -> float:
    """Compute root mean square"""
    return float(np.sqrt(np.mean(array ** 2)))

def db_scale(spec: SpectrogramArray, ref: float = None) -> SpectrogramArray:
    """Convert to dB scale"""
    return librosa.amplitude_to_db(magnitude(spec), ref=ref)
