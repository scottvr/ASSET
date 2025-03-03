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

def calculate_dynamic_range(audio: AudioArray) -> float:
    """Calculate dynamic range in dB"""
    # Use RMS with small windows
    frame_length = 2048
    hop_length = 512
        
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )
        
    db_range = librosa.amplitude_to_db(rms.max()) - librosa.amplitude_to_db(rms.min())
    return float(db_range)
    
def calculate_phase_complexity(vocal_spec: SpectrogramArray,
                              mix_spec: SpectrogramArray) -> float:
    """Measure complexity of phase relationships"""
    vocal_phase = np.angle(vocal_spec)
    mix_phase = np.angle(mix_spec)
        
    # Calculate phase differences and their variation
    phase_diff = np.abs(vocal_phase - mix_phase)
    return float(np.std(phase_diff))
    
def calculate_onset_variation(
    audio: AudioArray,
    sample_rate: int,
    normalize: bool = True
) -> float:
    """
    Calculate variation in onset strength as a measure of transitions.
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        normalize: Whether to normalize the variation score
        
    Returns:
        Float indicating amount of transition variation
    """
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate
    )
    
    variation = np.std(onset_env)
    
    if normalize:
        # Normalize to 0-1 range based on typical values
        variation = variation / (variation + 1.0)
        
    return float(variation)