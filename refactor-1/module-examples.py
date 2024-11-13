# src/core/audio.py
@dataclass
class AudioSegment:
    """Data class for audio segments with their metadata"""
    audio: np.ndarray
    sample_rate: int = 44100
    start_time: float = 0.0
    duration: float = 0.0
    
    # ... (rest of AudioSegment implementation)

# src/core/types.py
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

# src/separation/base.py
class VocalSeparator(ABC):
    """Abstract base class for vocal separators"""
    @abstractmethod
    def separate_and_analyze(self,
                           vocal_paths: Tuple[str, str],
                           accompaniment_paths: Tuple[str, str],
                           start_time: float = 0.0,
                           duration: float = 30.0) -> SeparationResult:
        pass

# src/separation/spleeter.py
class SpleeterSeparator(VocalSeparator):
    """Concrete implementation using Spleeter"""
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        # ... (rest of implementation)

# src/analysis/phase.py
class PhaseAnalyzer:
    """Phase-specific analysis tools"""
    def analyze_phase_preservation(self, 
                                 clean: AudioSegment,
                                 separated: AudioSegment) -> Dict[str, Any]:
        # ... (phase analysis implementation)

# src/diffusion/models.py
class PhaseAwareLoRA(nn.Module):
    """LoRA implementation for phase-aware processing"""
    def __init__(self, base_model: nn.Module):
        # ... (LoRA implementation)

# Usage example (examples/phase_aware_processing.py)
from src.core.types import SeparationResult
from src.separation.spleeter import SpleeterSeparator
from src.diffusion.models import PhaseAwareLoRA

def main():
    separator = SpleeterSeparator(output_dir="output")
    # ... (rest of example implementation)
