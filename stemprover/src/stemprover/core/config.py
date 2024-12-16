from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional
from stemprover.core.types import ProcessingConfig

class SeparatorBackend(Enum):
    SPLEETER = auto()
    DEMUCS = auto()
    MDX = auto()

@dataclass
class SeparationProfile:
    """Processing profile configuration"""
    backend: SeparatorBackend
    preserve_high_freq: bool = False
    target_sample_rate: int = 44100
    min_segment_length: float = 5.0
    
    # Enhancement settings
    use_phase_aware_controlnet: bool = False
    use_high_freq_processor: bool = True
    artifact_reduction_config: Optional[ProcessingConfig] = None
    controlnet_model_path: Optional[Path] = None
