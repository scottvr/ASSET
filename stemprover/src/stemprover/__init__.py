"""Stemprover - audio stem separation enhancement tools"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Core imports
from stemprover.core.audio import AudioSegment
from stemprover.core.types import (
    ProcessingConfig,
    SeparationResult
)

# Separation components
from .separation.base import VocalSeparator
#from .separation.spleeter import SpleeterSeparator

# Analysis components
from .analysis.base import VocalAnalyzer
from .analysis.spectral import SpectralAnalyzer
from .analysis.phase import PhaseAnalyzer

# Future diffusion components
#from .diffusion.models import PhaseAwareLoRA
#from .diffusion.training import PhaseAwareTrainer
from stemprover.enhancement.controlnet import PhaseAwareControlNet

__all__ = [
    # Version
    '__version__',
    
    # Core
    'AudioSegment',
    'ProcessingConfig',
    'SeparationResult',
    
    # Separation
    'VocalSeparator',
    # 'SpleeterSeparator',
    
    # Analysis
    'VocalAnalyzer',
    'SpectralAnalyzer',
    'PhaseAnalyzer',
    
    # Diffusion
    'PhaseAwareLoRA',
    'PhaseAwareTrainer',
]
