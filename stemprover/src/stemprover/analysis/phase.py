from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import json
import librosa

from ..core.audio import AudioSegment
from .base import VocalAnalyzer
from ..core.types import SpectrogramArray, AudioArray

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
        clean_spec = self.create_phase_spectrogram(clean.audio, clean.sr)
        separated_spec = self.create_phase_spectrogram(separated.audio, separated.sr)

        # 2. Ensure spectrograms have the same shape
        min_shape = min(clean_spec.shape[1], separated_spec.shape[1])
        clean_spec = clean_spec[:, :min_shape]
        separated_spec = separated_spec[:, :min_shape]

        # 3. Calculate phase metrics
        phase_diff = self._phase_difference(clean_spec, separated_spec)
        phase_coherence = self._phase_coherence(phase_diff)
        phase_complexity = self._calculate_phase_complexity(separated_spec, clean_spec) # Note: order might matter

        # 4. Store results
        results = {
            "phase_coherence": phase_coherence,
            "phase_complexity": phase_complexity,
            "mean_phase_difference": float(np.mean(phase_diff)),
        }

        # 5. Save results to JSON
        output_path = self.output_dir / f"phase_analysis_{clean.path.stem}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        return output_path

    def create_phase_spectrogram(self, audio: AudioArray, sr: int) -> SpectrogramArray:
        """
        Create spectrogram with phase preservation for analysis.
        """
        return librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)

    # Private methods migrated from utils
    def _angle(self, complex_spec: SpectrogramArray) -> SpectrogramArray:
        """Get phase angle from complex spectrogram"""
        return np.angle(complex_spec)

    def _magnitude(self, complex_spec: SpectrogramArray) -> SpectrogramArray:
        """Get magnitude from complex spectrogram"""
        return np.abs(complex_spec)

    def _phase_difference(self, spec1: SpectrogramArray, spec2: SpectrogramArray) -> SpectrogramArray:
        """Compute phase difference between spectrograms"""
        return np.abs(self._angle(spec1) - self._angle(spec2))

    def _phase_coherence(self, phase_diff: SpectrogramArray) -> float:
        """Compute phase coherence from phase difference"""
        return float(np.mean(np.cos(phase_diff)))

    def _calculate_phase_complexity(self, vocal_spec: SpectrogramArray,
                                   mix_spec: SpectrogramArray) -> float:
        """Measure complexity of phase relationships"""
        vocal_phase = self._angle(vocal_spec)
        mix_spec_phase = self._angle(mix_spec)

        # Calculate phase differences and their variation
        phase_diff = np.abs(vocal_phase - mix_spec_phase)
        return float(np.std(phase_diff))
