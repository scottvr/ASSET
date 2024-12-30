import tensorflow as tf
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
from spleeter.separator import Separator as SpleeterBase

from .base import VocalSeparator
from ..core.audio import AudioSegment
from ..core.types import SeparationResult
from ..io.audio import load_audio_file, save_audio_file
from ..analysis.spectral import SpectralAnalyzer

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

@dataclass
class SeparationResult:
    """Results from stem separation"""
    separated_vocal: AudioSegment
    separated_accompaniment: AudioSegment
    mixed: Optional[AudioSegment] = None
    file_paths: Optional[Dict[str, Path]] = None


class SpleeterSeparator(VocalSeparator):
    """Concrete implementation using Spleeter"""
    
    def __init__(self, output_dir: str = "output"):
        super().__init__(output_dir)
        # Defer TensorFlow setup until needed
        self.separator = None
        self.graph = None
        self.session = None

    def _setup_tensorflow(self):
        """Setup TensorFlow session and graph - called only when needed"""
        if self.separator is not None:
            return  # Already initialized
            
        # Create a new graph and session without resetting
        self.graph = tf.Graph()
        self.graph.as_default().__enter__()
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(graph=self.graph, config=config)
        self.session.as_default().__enter__()
        
        # Initialize Spleeter only after graph/session setup
        self.separator = SpleeterBase('spleeter:2stems')

    def separate(self, mixed: AudioSegment) -> SeparationResult:
        """Separate vocals and accompaniment from mixed audio"""
        self._setup_tensorflow()
        waveform = mixed.to_mono().audio.reshape(-1, 1)
        
        # Spleeter returns both vocals and accompaniment
        predictions = self.separator.separate(waveform)
        
        separated_vocal = AudioSegment(
            audio=predictions['vocals'].T,  # Convert back to our format
            sample_rate=mixed.sample_rate
        )
        
        separated_accompaniment = AudioSegment(
            audio=predictions['accompaniment'].T,
            sample_rate=mixed.sample_rate
        )
        
        return SeparationResult(
            separated_vocal=separated_vocal,
            separated_accompaniment=separated_accompaniment,
            mixed=mixed
        )

    def _save_audio_files(
        self,
        separated: AudioSegment,
        mixed: AudioSegment,
        start_time: float
    ) -> Dict[str, Path]:
        """Save separated and mixed audio"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.output_dir / f"separation_{timestamp}"
        save_dir.mkdir(exist_ok=True)

        files = {}
        segments = {
            'mix': mixed,
            'separated_vocal': separated
        }

        for name, segment in segments.items():
            path = save_dir / f"{name}_from_{start_time:.1f}s.wav"
            save_audio_file(segment, path)
            files[name] = path

        return files

    def separate_and_analyze(self,
                           vocal_paths: Tuple[str, str],
                           accompaniment_paths: Tuple[str, str],
                           start_time: float = 0.0,
                           duration: float = 30.0,
                           run_analysis: bool = True) -> SeparationResult:
        """Main method to perform separation and analysis"""
        # Ensure TensorFlow is set up
        self._setup_tensorflow()
        
        # Load audio
        vocals = self._load_stereo_pair(*vocal_paths, start_time, duration)
        accompaniment = self._load_stereo_pair(*accompaniment_paths, start_time, duration)

        # Create mix
        mixed = AudioSegment(
            audio=vocals.audio + accompaniment.audio,
            sample_rate=vocals.sample_rate,
            start_time=start_time,
            duration=duration
        )

        # Perform separation
        separated = self._separate_vocals(mixed)
        
        # Save files
        file_paths = self._save_audio_files(
            vocals, accompaniment, mixed, separated, start_time
        )

        result = SeparationResult(
            clean_vocal=vocals,
            separated_vocal=separated,
            accompaniment=accompaniment,
            mixed=mixed,
            file_paths=file_paths
        )

        # Run analysis if requested
        if run_analysis:
            result.analysis_path = self.analyzer.analyze(vocals, separated)

        return result

    def _load_stereo_pair(self, left_path: str, right_path: str, 
                         start_time: float, duration: float) -> AudioSegment:
        """Load and process stereo pair"""
        print(f"Loading {left_path}...")
        left, sr = load_audio_file(left_path, sr=44100, mono=True)
        print(f"Left channel length: {len(left)} samples ({len(left)/44100:.2f} seconds)")

        print(f"Loading {right_path}...")
        right, _ = load_audio_file(right_path, sr=44100, mono=True)
        print(f"Right channel length: {len(right)} samples ({len(right)/44100:.2f} seconds)")

        # Ensure same length
        min_length = min(len(left), len(right))
        left = left[:min_length]
        right = right[:min_length]
        print(f"Adjusted stereo length: {min_length} samples ({min_length/44100:.2f} seconds)")

        # Extract segment
        start_sample = int(start_time * 44100)
        duration_samples = int(duration * 44100)

        print(f'DBG: start_sample: {start_sample}, duration_samples: {duration_samples}, min_length: {min_length}') 

        if start_sample + duration_samples > min_length:
            print(f"Warning: Requested duration extends beyond audio length. Truncating.")
            duration_samples = min_length - start_sample
        
        left_segment = left[start_sample:start_sample + duration_samples]
        right_segment = right[start_sample:start_sample + duration_samples]

        # Stack to stereo
        stereo = np.vstack([left_segment, right_segment])
        
        return AudioSegment(
            audio=stereo,
            sample_rate=44100,
            start_time=start_time,
            duration=duration_samples/44100
        )

    def _separate_vocals(self, mixed: AudioSegment) -> AudioSegment:
        """Perform vocal separation"""
        # Convert to mono and reshape for Spleeter
        mix_mono = mixed.to_mono().audio
        mix_mono = mix_mono.reshape(-1, 1)

        print(f"Mix shape before separation: {mix_mono.shape}")
        print("Running separation...")
        
        separated = self.separator.separate(mix_mono)
        separated_vocals = separated['vocals']
        print(f"Separated vocals shape: {separated_vocals.shape}")
        
        # Since Spleeter returns (samples, channels), we should handle it accordingly
        if len(separated_vocals.shape) == 2:
            if separated_vocals.shape[1] == 2:
                # If it's stereo, convert to our preferred format (2, samples)
                separated_vocals = separated_vocals.T
            elif separated_vocals.shape[1] == 1:
                # If it's mono in (samples, 1) shape, convert to 1D array
                separated_vocals = separated_vocals.reshape(-1)
                
        print(f"Final separated vocals shape: {separated_vocals.shape}")
        
        if len(separated_vocals.shape) == 1:
            # If mono, duplicate to stereo to match input
            separated_vocals = np.vstack([separated_vocals, separated_vocals])
        
        print(f"Output separated vocals shape: {separated_vocals.shape}")
            
        if separated_vocals.shape[1] < 1000:  # Sanity check on the correct dimension
            raise ValueError(f"Separated vocals seem too short: {separated_vocals.shape}")
                
        return AudioSegment(
            audio=separated_vocals,
            sample_rate=mixed.sample_rate,
            start_time=mixed.start_time,
            duration=mixed.duration
        )

    def _save_audio_files(
        self,
        separation_result: SeparationResult,
        start_time: float
    ) -> Dict[str, Path]:
        """Save all separated audio files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.output_dir / f"separation_{timestamp}"
        save_dir.mkdir(exist_ok=True)

        files = {}
        segments = {
            'separated_vocal': separation_result.separated_vocal,
            'separated_accompaniment': separation_result.separated_accompaniment
        }
        
        if separation_result.mixed is not None:
            segments['mix'] = separation_result.mixed

        for name, segment in segments.items():
            path = save_dir / f"{name}_from_{start_time:.1f}s.wav"
            save_audio_file(segment, path)
            files[name] = path

        return files

    def separate_file(
        self,
        mixed_path: Union[str, Tuple[str, str]],
        start_time: float = 0.0,
        duration: float = 30.0,
        save_files: bool = True
    ) -> SeparationResult:
        """Separate vocals from audio file(s)"""
        if isinstance(mixed_path, tuple):
            mixed = self._load_stereo_pair(*mixed_path, start_time, duration)
        else:
            mixed = self._load_mono(mixed_path, start_time, duration)

        result = self.separate(mixed)
        
        if save_files:
            result.file_paths = self._save_audio_files(result, start_time)
            
        return result

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'separator') and hasattr(self.separator, '_get_model'):
                self.separator._get_model.cache_clear()
            
            if hasattr(self, 'session'):
                self.session.as_default().__exit__(None, None, None)
                self.session.close()
                delattr(self, 'session')
            
            if hasattr(self, 'graph'):
                self.graph.as_default().__exit__(None, None, None)
                delattr(self, 'graph')
            
        except Exception as e:
            print(f"Warning during cleanup: {str(e)}")

    @property
    def capabilities(self) -> Dict[str, any]:
        """Report capabilities/limitations of this backend"""
        return {
            "max_frequency": 11000,  # Hz
            "supports_stereo": True,
            "native_sample_rate": 22050,
            "recommended_min_segment": 5.0  # seconds
        }
