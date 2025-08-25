from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import librosa

from ...types import AudioArray, SpectrogramArray
from ...utils import create_spectrogram, calculate_onset_variation, calculate_band_energy
from ...core.types import ProcessingConfig
from .metrics import SegmentMetrics
from ...core.audio import AudioSegment
from ...core.types import (
    ProcessingConfig,
    SeparationResult
)

@dataclass
class FoundSegment:
    """Dataclass to hold information about a found segment."""
    start: int
    end: int
    metrics: SegmentMetrics
    time: float

class TestSegmentFinder:
    """Finds ideal segments for overfitting tests"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.vocal_bands = (200, 4000)  # Primary vocal frequency range
        self.high_freq = 11000          # Spleeter cutoff

    def analyze_segment(self,
                       vocal_segment: AudioSegment,
                       backing_segment: AudioSegment) -> SegmentMetrics:
        """Compute comprehensive metrics for a segment"""

        vocal = vocal_segment.audio
        backing = backing_segment.audio

        # Create spectrograms
        vocal_spec = create_spectrogram(
            vocal,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        mix_spec = create_spectrogram(
            vocal + backing,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )

        # Get frequency bins
        freqs = librosa.fft_frequencies(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft
        )

        # Calculate metrics
        vocal_clarity = self._calculate_vocal_clarity(
            vocal_spec, mix_spec, freqs
        )

        high_freq_content = self._calculate_high_freq_content(
            vocal_spec, freqs
        )

        dynamic_range = calculate_dynamic_range(vocal)

        phase_complexity = calculate_phase_complexity(
            vocal_spec, mix_spec
        )

        transition_score = self._calculate_transitions(vocal)

        # Compute overall score
        score = self._compute_score(
            vocal_clarity,
            high_freq_content,
            dynamic_range,
            phase_complexity,
            transition_score
        )

        return SegmentMetrics(
            vocal_clarity=vocal_clarity,
            high_freq_content=high_freq_content,
            dynamic_range=dynamic_range,
            phase_complexity=phase_complexity,
            transition_score=transition_score,
            score=score
        )

    def _calculate_vocal_clarity(self,
                               vocal_spec: SpectrogramArray,
                               mix_spec: SpectrogramArray,
                               freqs: np.ndarray) -> float:
        """Calculate vocal band clarity using band energy ratio"""
        vocal_energy = calculate_band_energy(
            vocal_spec,
            freqs,
            self.vocal_bands
        )
        mix_energy = calculate_band_energy(
            mix_spec,
            freqs,
            self.vocal_bands
        )
        return vocal_energy / (mix_energy + 1e-8)

    def _calculate_high_freq_content(self,
                                   spec: SpectrogramArray,
                                   freqs: np.ndarray) -> float:
        """Calculate high frequency content using band energy"""
        return calculate_band_energy(
            spec,
            freqs,
            (self.high_freq, freqs.max())
        )

    def _calculate_transitions(self, audio: AudioArray) -> float:
        """Score vocal transitions and variations"""
        return calculate_onset_variation(
            audio,
            self.config.sample_rate
        )

    def _compute_score(self,
                      vocal_clarity: float,
                      high_freq_content: float,
                      dynamic_range: float,
                      phase_complexity: float,
                      transition_score: float) -> float:
        """Compute overall suitability score"""
        # Weights for different criteria
        weights = {
            'baseline': {
                'vocal_clarity': 0.4,
                'high_freq_content': 0.3,
                'dynamic_range': 0.1,
                'phase_complexity': 0.1,
                'transition_score': 0.1
            },
            'challenge': {
                'vocal_clarity': 0.3,
                'high_freq_content': 0.2,
                'dynamic_range': 0.2,
                'phase_complexity': 0.15,
                'transition_score': 0.15
            },
            'complex': {
                'vocal_clarity': 0.2,
                'high_freq_content': 0.2,
                'dynamic_range': 0.2,
                'phase_complexity': 0.2,
                'transition_score': 0.2
            }
        }

        # Calculate scores for each test case type
        scores = {}
        for test_type, w in weights.items():
            scores[test_type] = (
                vocal_clarity * w['vocal_clarity'] +
                high_freq_content * w['high_freq_content'] +
                dynamic_range * w['dynamic_range'] +
                phase_complexity * w['phase_complexity'] +
                transition_score * w['transition_score']
            )

        # Return highest score
        return max(scores.values())

def find_best_segments(
    vocal_track: AudioSegment,
    backing_track: AudioSegment,
    segment_length_sec: float,
    hop_length_sec: float,
    config: ProcessingConfig,
    top_k: int = 5
) -> List[FoundSegment]:
    """Find the best segments for testing"""
    finder = TestSegmentFinder(config)
    segments = []

    segment_length = int(segment_length_sec * config.sample_rate)
    hop_length = int(hop_length_sec * config.sample_rate)

    for start in range(0, len(vocal_track.audio) - segment_length, hop_length):
        end = start + segment_length

        vocal_segment = vocal_track.slice(start / config.sample_rate, end / config.sample_rate)
        backing_segment = backing_track.slice(start / config.sample_rate, end / config.sample_rate)

        metrics = finder.analyze_segment(vocal_segment, backing_segment)

        segments.append(FoundSegment(
            start=start,
            end=end,
            metrics=metrics,
            time=start / config.sample_rate
        ))

    # Sort by score and return top_k
    segments.sort(key=lambda x: x.metrics.score, reverse=True)
    return segments[:top_k]
