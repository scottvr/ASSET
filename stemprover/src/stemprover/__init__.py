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
from .separation.spleeter import SpleeterSeparator

# Analysis components
from .analysis.base import VocalAnalyzer
from .analysis.spectral import SpectralAnalyzer
#from .analysis.phase import PhaseAnalyzer
# TODO: Implement PhaseAnalyzer using existing phase analysis infrastructure
# - Consolidate phase-related functions from common/math_utils.py
# - Incorporate phase complexity calculations from audio_utils.py
# - Consider alpha-channel visualization integration
# (Postponed until after band-split validation experiment)

# Future diffusion components
#from .diffusion.models import PhaseAwareLoRA
from .enhancement\
from .diffusion.training import PhaseAwareTrainer

__all__ = [
    # Version
    '__version__',
    
    # Core
    'AudioSegment',
    'ProcessingConfig',
    'SeparationResult',
    
    # Separation
    'VocalSeparator',
    'SpleeterSeparator',
    
    # Analysis
    'VocalAnalyzer',
    'SpectralAnalyzer',
    'PhaseAnalyzer',
    
    # Diffusion
    'PhaseAwareLoRA',
    'PhaseAwareTrainer',
]
