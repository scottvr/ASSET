from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import librosa
import numpy as np

from ..core.audio import AudioSegment
from ..core.types import ProcessingConfig

class EnhancementProcessor(ABC):
    """Base class for audio enhancement processors"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    @abstractmethod
    def enhance(self, audio: AudioSegment) -> AudioSegment:
        """Enhance audio segment"""
        pass
    
    @abstractmethod
    def validate(self, audio: AudioSegment) -> dict:
        """Validate enhancement results"""
        pass


def load_audio(path: str, sr: int = 44100, mono: bool = True) -> torch.Tensor:
    audio, _ = librosa.load(path, sr=sr, mono=mono)
    return torch.from_numpy(audio).float()


class HighFrequencyArtifactPreprocessor(nn.Module):
    def __init__(self, threshold_freq: float = 11000, sample_rate: int = 44100):
        super().__init__()
        self.threshold_freq = threshold_freq
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = x[:, 2:3]
        freq_bins = torch.linspace(0, self.sample_rate / 2, x.shape[2])
        high_freq_indices = (freq_bins > self.threshold_freq).nonzero(as_tuple=True)[0]
        if len(high_freq_indices) > 0:
            start_idx = high_freq_indices[0]
            high_freq_content = magnitude[:, :, start_idx:, :]
            mean = torch.mean(high_freq_content, dim=[2, 3], keepdim=True)
            std = torch.std(high_freq_content, dim=[2, 3], keepdim=True)
            attention = torch.sigmoid((high_freq_content - mean) / (std + 1e-6))
            mask = torch.ones_like(magnitude)
            mask[:, :, start_idx:, :] = attention
            return mask
        else:
            return torch.ones_like(magnitude)


def audio_to_spectrogram(audio_tensor: torch.Tensor, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
    audio_numpy = audio_tensor.numpy()
    stft_result = librosa.stft(y=audio_numpy, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft_result)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    # Handle case where magnitude_db is all -inf
    if np.isneginf(magnitude_db).all():
        magnitude_normalized = np.zeros_like(magnitude_db)
    else:
        magnitude_normalized = (magnitude_db - np.min(magnitude_db[np.isfinite(magnitude_db)])) / (
                    np.max(magnitude_db[np.isfinite(magnitude_db)]) - np.min(magnitude_db[np.isfinite(magnitude_db)]))

    magnitude_rgb = np.stack([magnitude_normalized] * 3, axis=-1)
    magnitude_tensor = torch.from_numpy(magnitude_rgb).permute(2, 0, 1)
    phase_tensor = torch.from_numpy(phase).unsqueeze(0)
    return torch.cat([magnitude_tensor.float(), phase_tensor.float()], dim=0)


def generate_training_pair(clean_audio: torch.Tensor, separated_audio: torch.Tensor,
                           preprocessor: HighFrequencyArtifactPreprocessor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_spec_rgba = audio_to_spectrogram(clean_audio)
    sep_spec_rgba = audio_to_spectrogram(separated_audio)
    condition = preprocessor(sep_spec_rgba.unsqueeze(0)).squeeze(0)
    return condition, sep_spec_rgba, clean_spec_rgba