from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict
from ..core.types import SeparationResult
from ..core.audio import AudioSegment

class VocalSeparator(ABC):
    """Abstract base class for vocal separators"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def separate_and_analyze(self,
                           vocal_paths: Tuple[str, str],
                           accompaniment_paths: Tuple[str, str],
                           start_time: float = 0.0,
                           duration: float = 30.0,
                           run_analysis: bool = True) -> SeparationResult:
        """Perform separation and analysis"""
        pass

    @abstractmethod
    def _load_stereo_pair(self, left_path: str, right_path: str, 
                         start_time: float, duration: float) -> AudioSegment:
        """Load and process stereo pair"""
        pass

    @abstractmethod
    def _separate_vocals(self, mixed: AudioSegment) -> AudioSegment:
        """Perform vocal separation"""
        pass

    @abstractmethod
    def _save_audio_files(self, vocals: AudioSegment, 
                         accompaniment: AudioSegment,
                         mixed: AudioSegment, 
                         separated: AudioSegment,
                         start_time: float) -> Dict[str, Path]:
        """Save all audio files"""
        pass

    def cleanup(self):
        """Cleanup resources - override if needed"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
