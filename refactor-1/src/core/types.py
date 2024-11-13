from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
from .audio import AudioSegment

@dataclass
class SeparationResult:
    """Data class for storing separation results"""
    clean_vocal: AudioSegment
    separated_vocal: AudioSegment
    accompaniment: AudioSegment
    mixed: AudioSegment
    file_paths: Dict[str, Path]
    analysis_path: Optional[Path] = None
    phase_analysis: Optional[Dict[str, Any]] = None

@dataclass
class ProcessingConfig:
    """Configuration for audio processing"""
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    pad_mode: str = 'constant'
    image_scale_factor: float = 1.0
    image_chunk_size: int = 512
    torch_dtype: str = 'float16'
    attention_slice_size: Optional[int] = 1
    enable_xformers: bool = True
    enable_cuda_graph: bool = False
    diffusion_strength: float = 0.75
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
