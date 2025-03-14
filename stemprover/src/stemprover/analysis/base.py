from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from ..core.audio import AudioSegment

class VocalAnalyzer(ABC):
    """Base class for vocal analysis"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def analyze(self, clean: AudioSegment, separated: AudioSegment) -> Path:
        """Perform analysis and return path to analysis results"""
        pass

    @abstractmethod
    def create_phase_spectrogram(self, audio: np.ndarray, sr: int):
        """Create spectrogram with phase preservation for analysis"""
        pass
