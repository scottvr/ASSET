import torch
import torch.nn as nn
from typing import Tuple

class HighFrequencyArtifactPreprocessor(nn.Module):
    def __init__(self, threshold_freq: float = 11000, sample_rate: int = 44100):
        super().__init__()
        self.threshold_freq = threshold_freq
        self.sample_rate = sample_rate
<<<<<<< HEAD
        
=======

>>>>>>> jules
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGBA spectrogram tensor (B, 4, H, W)
        Returns:
            Single-channel attention map (B, 1, H, W)
        """
        # Extract magnitude from blue channel (or any of R, G, B as they are the same)
        magnitude = x[:, 2:3]  # (B, 1, H, W)
<<<<<<< HEAD
        
        # Calculate frequency bins corresponding to the height of the spectrogram
        freq_bins = torch.linspace(0, self.sample_rate / 2, x.shape[2])
        
        # Create a mask for frequencies above the threshold
        high_freq_indices = (freq_bins > self.threshold_freq).nonzero(as_tuple=True)[0]
        
        if len(high_freq_indices) > 0:
            start_idx = high_freq_indices[0]
            
            # Analyze high frequency content
            high_freq_content = magnitude[:, :, start_idx:, :]
            
            # Detect potential artifacts using local statistics (a simple form of attention)
            mean = torch.mean(high_freq_content, dim=[2, 3], keepdim=True)
            std = torch.std(high_freq_content, dim=[2, 3], keepdim=True)
            
            # Create an attention map for the high-frequency region
            attention = torch.sigmoid((high_freq_content - mean) / (std + 1e-6))
            
=======

        # Calculate frequency bins corresponding to the height of the spectrogram
        freq_bins = torch.linspace(0, self.sample_rate / 2, x.shape[2])

        # Create a mask for frequencies above the threshold
        high_freq_indices = (freq_bins > self.threshold_freq).nonzero(as_tuple=True)[0]

        if len(high_freq_indices) > 0:
            start_idx = high_freq_indices[0]

            # Analyze high frequency content
            high_freq_content = magnitude[:, :, start_idx:, :]

            # Detect potential artifacts using local statistics (a simple form of attention)
            mean = torch.mean(high_freq_content, dim=[2, 3], keepdim=True)
            std = torch.std(high_freq_content, dim=[2, 3], keepdim=True)

            # Create an attention map for the high-frequency region
            attention = torch.sigmoid((high_freq_content - mean) / (std + 1e-6))

>>>>>>> jules
            # Create a base mask and insert the attention map
            mask = torch.ones_like(magnitude)
            mask[:, :, start_idx:, :] = attention
            return mask
        else:
            # If no frequencies are above the threshold, return a map of ones (no change)
            return torch.ones_like(magnitude)
