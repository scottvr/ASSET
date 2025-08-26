import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

from asteroid.models import DPRNNTasNet

from .base import VocalSeparator
from ..core.audio import AudioSegment
from ..core.types import SeparationResult
from ..io.audio import load_audio_file, save_audio_file


class AsteroidSeparator(VocalSeparator):
    """Concrete implementation using Asteroid."""

    def __init__(self, output_dir: str = "output"):
        super().__init__(Path(output_dir))
        self.model = DPRNNTasNet.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean")
        self.original_sr = 44100
        self.model_sr = 8000

    def separate(self, mixed: AudioSegment) -> SeparationResult:
        """Separate vocals and accompaniment from mixed audio"""
        # Resample to model's sample rate
        audio_resampled = librosa.resample(
            mixed.to_mono().audio, orig_sr=mixed.sample_rate, target_sr=self.model_sr
        )

        # Convert to tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0)

        # Separate
        separated_tensors = self.model.separate(audio_tensor)

        # Assuming the first source is vocals and the second is accompaniment
        vocals_8k = separated_tensors[0][0].detach().numpy()
        accompaniment_8k = separated_tensors[0][1].detach().numpy()

        # Resample back to original sample rate
        vocals_resampled = librosa.resample(
            vocals_8k, orig_sr=self.model_sr, target_sr=mixed.sample_rate
        )
        accompaniment_resampled = librosa.resample(
            accompaniment_8k, orig_sr=self.model_sr, target_sr=mixed.sample_rate
        )

        # Duplicate mono to stereo to match input format
        separated_vocal = AudioSegment(
            audio=np.vstack([vocals_resampled, vocals_resampled]),
            sample_rate=mixed.sample_rate
        )

        separated_accompaniment = AudioSegment(
            audio=np.vstack([accompaniment_resampled, accompaniment_resampled]),
            sample_rate=mixed.sample_rate
        )

        return SeparationResult(
            separated_vocal=separated_vocal,
            separated_accompaniment=separated_accompaniment,
            mixed=mixed
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
        mixed_path: str,
        start_time: float = 0.0,
        duration: float = 30.0,
        save_files: bool = True
    ) -> SeparationResult:
        """Separate vocals from an audio file"""
        audio, sr = load_audio_file(mixed_path, sr=self.original_sr, mono=True)
        
        start_sample = int(start_time * sr)
        end_sample = start_sample + int(duration * sr)
        audio_segment = audio[start_sample:end_sample]

        mixed = AudioSegment(
            audio=audio_segment,
            sample_rate=sr,
            start_time=start_time,
            duration=duration
        )

        result = self.separate(mixed)

        if save_files:
            result.file_paths = self._save_audio_files(result, start_time)

        return result

    def separate_and_analyze(self,
                           vocal_paths: Tuple[str, str],
                           accompaniment_paths: Tuple[str, str],
                           start_time: float = 0.0,
                           duration: float = 30.0,
                           run_analysis: bool = True) -> SeparationResult:
        """Main method to perform separation and analysis"""
        # This is a complex method that depends on other parts of the system
        # (e.g. analyzer). For now, I will implement a simplified version
        # that focuses on the separation part.
        raise NotImplementedError("separate_and_analyze is not fully implemented for AsteroidSeparator")

    def _load_stereo_pair(self, left_path: str, right_path: str,
                         start_time: float, duration: float) -> AudioSegment:
        """Load and process stereo pair"""
        raise NotImplementedError("_load_stereo_pair is not fully implemented for AsteroidSeparator")

    def _separate_vocals(self, mixed: AudioSegment) -> AudioSegment:
        """Perform vocal separation"""
        # This method is now effectively replaced by the `separate` method.
        # I will call `separate` and return the vocal part.
        result = self.separate(mixed)
        return result.separated_vocal

    def cleanup(self):
        """Cleanup resources"""
        self.model = None
        torch.cuda.empty_cache()

    @property
    def capabilities(self) -> Dict[str, any]:
        """Report capabilities/limitations of this backend"""
        return {
            "max_frequency": self.model_sr / 2,
            "supports_stereo": False, # model is mono
            "native_sample_rate": self.model_sr,
            "recommended_min_segment": 2.0
        }
