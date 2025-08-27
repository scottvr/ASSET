import torch
import librosa
import numpy as np
from typing import Tuple

from ..analysis.artifacts.preprocessor import HighFrequencyArtifactPreprocessor

def audio_to_spectrogram(
    audio_tensor: torch.Tensor, 
    n_fft: int = 2048, 
    hop_length: int = 512
) -> torch.Tensor:
    """
    Converts a raw audio tensor into a 4-channel 'RGBA' spectrogram.
    - R, G, B channels are the magnitude spectrogram (normalized).
    - A channel is the phase spectrogram.
    """
    audio_numpy = audio_tensor.numpy()
    
    # 1. Compute STFT
    stft_result = librosa.stft(y=audio_numpy, n_fft=n_fft, hop_length=hop_length)
    
    # 2. Separate magnitude and phase
    magnitude, phase = librosa.magphase(stft_result)
    
    # 3. Normalize magnitude and create 3-channel image
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    magnitude_normalized = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min())
    magnitude_rgb = np.stack([magnitude_normalized] * 3, axis=-1)
    
    # 4. Convert to torch tensors
    magnitude_tensor = torch.from_numpy(magnitude_rgb).permute(2, 0, 1) # HWC to CHW
    phase_tensor = torch.from_numpy(phase).unsqueeze(0) # Add channel dim
    
    # 5. Combine to 4-channel tensor
    return torch.cat([magnitude_tensor, phase_tensor], dim=0)


def generate_training_pair(
    clean_audio: torch.Tensor,
    separated_audio: torch.Tensor,
    preprocessor: HighFrequencyArtifactPreprocessor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training pair for ControlNet
    Returns:
        condition: Control signal from preprocessor
        input_spec: Separated audio spectrogram (RGBA)
        target_spec: Clean audio spectrogram (RGBA)
    """
    # Convert both to spectrograms
    # The preprocessor expects the input to be in the 4-channel format
    clean_spec_rgba = audio_to_spectrogram(clean_audio)
    sep_spec_rgba = audio_to_spectrogram(separated_audio)
    
    # Generate control signal from the separated spectrogram
    # The preprocessor's forward method expects a batch, so we add a batch dimension
    condition = preprocessor(sep_spec_rgba.unsqueeze(0)).squeeze(0) # Remove batch dim after
    
    # The model's target should be the clean spectrogram
    # The model's input is the separated spectrogram
    return condition, sep_spec_rgba, clean_spec_rgba
