import torch
import torch.nn as nn
from typing import Optional, List, Tuple

class ArtifactDetector(nn.Module):
    """Preprocessor that generates artifact control maps"""
    def __init__(self):
        super().__init__()
        # Lightweight conv layers for artifact detection
        self.detector = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),  # 4 channels: RGB + phase
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)  # Single channel artifact map
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGBA tensor where A channel contains phase information
        Returns:
            Single-channel artifact probability map
        """
        return torch.sigmoid(self.detector(x))

class PhaseAwareZeroConv(nn.Module):
    """Zero convolution block that preserves phase information"""
    def __init__(
        self, 
        input_channels: int,
        output_channels: int,
        phase_channels: int = 1
    ):
        super().__init__()
        self.main_conv = nn.Conv2d(input_channels, output_channels, 1)
        self.phase_conv = nn.Conv2d(phase_channels, phase_channels, 1)
        
        # Initialize to zero
        nn.init.zeros_(self.main_conv.weight)
        nn.init.zeros_(self.main_conv.bias)
        nn.init.zeros_(self.phase_conv.weight)
        nn.init.zeros_(self.phase_conv.bias)
        
    def forward(self, x: torch.Tensor, control: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Split main features and phase
        features, phase = x[:, :-1], x[:, -1:]
        
        # Apply zero convolutions
        out_features = self.main_conv(features * control)
        out_phase = self.phase_conv(phase * control)
        
        return torch.cat([out_features, out_phase], dim=1)

class PhaseAwareControlNet(nn.Module):
    """ControlNet adaptation for phase-aware spectrogram processing"""
    def __init__(
        self,
        base_model: nn.Module,
        control_channels: int = 1,
        phase_channels: int = 1
    ):
        super().__init__()
        self.base_model = base_model
        self.artifact_detector = ArtifactDetector()
        
        # Create zero conv layers for each injection point
        self.zero_convs = nn.ModuleList([
            PhaseAwareZeroConv(
                base_model.get_feature_channels(i),
                base_model.get_feature_channels(i),
                phase_channels
            )
            for i in range(base_model.num_injection_points)
        ])
        
    def forward(
        self, 
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Generate control signal
        control = self.artifact_detector(x)
        
        # Get intermediate features from frozen base model
        features = self.base_model.get_feature_pyramid(x, timestep, context)
        
        # Apply controlled features through zero convs
        controlled_features = []
        for feat, zero_conv in zip(features, self.zero_convs):
            controlled_features.append(zero_conv(feat, control))
            
        return self.base_model.forward_with_features(
            x,
            controlled_features,
            timestep,
            context
        )
