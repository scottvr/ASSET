from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import json
import librosa

from ..core.audio import AudioSegment
from .base import VocalAnalyzer
from ..types import SpectrogramArray, AudioArray
from ..utils import calculate_phase_complexity

class PhaseAnalyzer(VocalAnalyzer):
    """
    Analyzer focused on phase-related metrics between audio signals.
    """
    def __init__(self, output_dir: Path, n_fft: int = 2048, hop_length: int = 512):
        super().__init__(output_dir)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, clean: AudioSegment, separated: AudioSegment) -> Path:
        """
        Analyzes phase differences and coherence and saves results to a file.
        """
        # 1. Generate spectrograms
        clean_spec = self.create_phase_spectrogram(clean.audio, clean.sample_rate)
        separated_spec = self.create_phase_spectrogram(separated.audio, separated.sample_rate)

        # 2. Ensure spectrograms have the same shape
        min_shape = min(clean_spec.shape[1], separated_spec.shape[1])
        clean_spec = clean_spec[:, :min_shape]
        separated_spec = separated_spec[:, :min_shape]

        # 3. Calculate phase metrics
        phase_diff = np.abs(np.angle(clean_spec) - np.angle(separated_spec))
        phase_coherence = float(np.mean(np.cos(phase_diff)))
        phase_complexity = calculate_phase_complexity(separated_spec, clean_spec)

        # 4. Store results
        results = {
            "phase_coherence": phase_coherence,
            "phase_complexity": phase_complexity,
            "mean_phase_difference": float(np.mean(phase_diff)),
        }

        # 5. Save results to JSON
        output_name = clean.name or "unnamed_segment"
        output_path = self.output_dir / f"phase_analysis_{output_name}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        return output_path

    def create_phase_spectrogram(self, audio: AudioArray, sr: int) -> SpectrogramArray:
        """
        Create spectrogram with phase preservation for analysis.
        """
        return librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
