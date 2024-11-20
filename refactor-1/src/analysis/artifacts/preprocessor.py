class HighFrequencyArtifactPreprocessor(nn.Module):
    def __init__(self, threshold_freq: float = 11000, sample_rate: int = 44100):
        super().__init__()
        self.threshold_freq = threshold_freq
        self.sample_rate = sample_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGBA spectrogram tensor (B, 4, H, W)
        Returns:
            Single-channel attention map (B, 1, H, W)
        """
        # Extract magnitude from blue channel
        magnitude = x[:, 2:3]  # (B, 1, H, W)
        
        # Calculate frequency bins
        freq_bins = torch.linspace(0, self.sample_rate/2, magnitude.shape[2])
        
        # Create frequency mask
        mask = torch.ones_like(magnitude)
        high_freq_idx = (freq_bins > self.threshold_freq).nonzero()
        
        if high_freq_idx.numel() > 0:
            # Analyze high frequency content
            high_freq_content = magnitude[:, :, high_freq_idx.squeeze():]
            
            # Detect potential artifacts using local statistics
            mean = torch.mean(high_freq_content, dim=-1, keepdim=True)
            std = torch.std(high_freq_content, dim=-1, keepdim=True)
            
            # Create attention map
            attention = torch.sigmoid(
                (high_freq_content - mean) / (std + 1e-6)
            )
            
            # Insert back into full mask
            mask[:, :, high_freq_idx.squeeze():] = attention
        
        return mask

def generate_training_pair(
    clean_audio: torch.Tensor,
    separated_audio: torch.Tensor,
    preprocessor: HighFrequencyArtifactPreprocessor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training pair for ControlNet
    Returns:
        condition: Control signal from preprocessor
        input_spec: Separated audio spectrogram
        target_spec: Clean audio spectrogram
    """
    # Convert both to spectrograms
    clean_spec = audio_to_spectrogram(clean_audio)
    sep_spec = audio_to_spectrogram(separated_audio)
    
    # Generate control signal
    condition = preprocessor(sep_spec)
    
    return condition, sep_spec, clean_spec
