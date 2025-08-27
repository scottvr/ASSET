from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, NewType
import numpy as np
import matplotlib.pyplot as plt
from .audio import AudioSegment

# Define custom types for clarity
AudioArray = NewType('AudioArray', np.ndarray)
SpectrogramArray = NewType('SpectrogramArray', np.ndarray)

@dataclass
class ProcessingConfig:
    """Configuration for audio processing"""
    # Audio parameters
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    pad_mode: str = 'constant'
    
    # Image processing
    image_scale_factor: float = 1.0
    image_chunk_size: int = 512
    
    # ControlNet parameters
    enable_controlnet: bool = True
    controlnet_strength: float = 0.75
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    
    # Optimization
    torch_dtype: str = 'float16'
    attention_slice_size: Optional[int] = 1
    enable_xformers: bool = True
    enable_cuda_graph: bool = False
    
    # Artifact processing
    artifact_threshold_freq: float = 11000
    artifact_smoothing_sigma: float = 1.0
    temporal_smoothing: float = 2.0

@dataclass
class SeparationResult:
    """Data class for storing separation results"""
    clean_vocal: AudioSegment
    separated_vocal: AudioSegment
    enhanced_vocal: Optional[AudioSegment]
    accompaniment: AudioSegment
    mixed: AudioSegment
    file_paths: Dict[str, Path]
    analysis_path: Optional[Path] = None
    phase_analysis: Optional[Dict[str, Any]] = None
    artifact_analysis: Optional[Dict[str, str]] = None

@dataclass
class SegmentConfig:
    """Configuration for segment generation"""
    segment_length: float = 5.0
    overlap: float = 2.5
    min_vocal_energy: float = 0.1  # Threshold for keeping vocal segments
    sample_rate: int = 44100
    
    @property
    def segment_samples(self) -> int:
        return int(self.segment_length * self.sample_rate)
    
    @property
    def hop_samples(self) -> int:
        return int((self.segment_length - self.overlap) * self.sample_rate)
