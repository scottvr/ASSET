import numpy as np
import librosa
import soundfile as sf
from typing import Tuple
from .types import AudioArray, SpectrogramArray, FrequencyBand
from .audio_utils import get_band_mask

def calculate_band_energy(
    spec: SpectrogramArray,
    freqs: np.ndarray,
    band: Tuple[float, float],
    relative: bool = True
) -> float:
    """Calculate energy in specific frequency band"""
    band_mask = get_band_mask(freqs, band[0], band[1])
    band_energy = np.mean(np.abs(spec[band_mask]))
    
    if relative:
        total_energy = np.mean(np.abs(spec))
        return band_energy / (total_energy + 1e-8)
    return band_energy