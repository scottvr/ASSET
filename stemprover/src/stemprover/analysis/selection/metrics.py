from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np
from stemprover.core.audio import AudioSegment
from stemprover.core.types import (
    ProcessingConfig,
    SeparationResult
)

@dataclass
class SegmentMetrics:
    """Enhanced metrics including SDR for research comparability"""
    # Standard SDR metric used in research
    sdr: float              # Signal-to-distortion ratio
    
    # Our detailed metrics
    vocal_clarity: float    # Ratio of vocal band energy to total
    high_freq_content: float# Energy above 11kHz
    dynamic_range: float    # dB difference between peaks
    phase_complexity: float # Measure of phase relationships
    transition_score: float # Score for vocal transitions
    
    # Optional band-specific SDR scores
    band_sdrs: Optional[Dict[str, float]] = None
    
    # Overall scores
    research_score: float = 0.0  # SDR-based score for research comparison
    detailed_score: float = 0.0  # Our original detailed score

    def calculate_sdr(target: np.ndarray, estimated: np.ndarray) -> float:
        """Calculate SDR between target and estimated signals"""
        # Ensure signals are aligned and same length
        min_len = min(len(target), len(estimated))
        target = target[:min_len]
        estimated = estimated[:min_len]
        
        # Calculate SDR
        signal_power = np.sum(target ** 2)
        noise_power = np.sum((estimated - target) ** 2)
        
        return 10 * np.log10(signal_power / (noise_power + 1e-10))

    def calculate_band_sdrs(
        target: np.ndarray,
        estimated: np.ndarray,
        frequency_bands: Dict[str, Tuple[float, float]],
        sample_rate: int
    ) -> Dict[str, float]:
        """Calculate SDR for each frequency band"""
        band_sdrs = {}
        
        for band_name, (low_freq, high_freq) in frequency_bands.items():
            # Filter signals to band
            target_band = filter_to_band(target, low_freq, high_freq, sample_rate)
            estimated_band = filter_to_band(estimated, low_freq, high_freq, sample_rate)
            
            # Calculate SDR for this band
            band_sdrs[band_name] = calculate_sdr(target_band, estimated_band)
        
        return band_sdrs

class MetricsCalculator:
    """Calculator for both research-standard and detailed metrics"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def calculate_metrics(
        self,
        target: np.ndarray,
        estimated: np.ndarray,
        genre: Optional[str] = None
    ) -> SegmentMetrics:
        """Calculate comprehensive metrics"""
        # Calculate standard SDR
        sdr = SegmentMetrics.calculate_sdr(target, estimated)
        
        # Calculate band-specific SDRs
        band_sdrs = SegmentMetrics.calculate_band_sdrs(
            target, estimated,
            self.config.frequency_bands,
            self.config.sample_rate
        )
        
        # Calculate our detailed metrics
        vocal_clarity = self._calculate_vocal_clarity(estimated)
        high_freq_content = self._calculate_high_freq_content(estimated)
        dynamic_range = self._calculate_dynamic_range(estimated)
        phase_complexity = self._calculate_phase_complexity(target, estimated)
        transition_score = self._calculate_transition_score(estimated)
        
        # Create metrics object
        metrics = SegmentMetrics(
            sdr=sdr,
            vocal_clarity=vocal_clarity,
            high_freq_content=high_freq_content,
            dynamic_range=dynamic_range,
            phase_complexity=phase_complexity,
            transition_score=transition_score,
            band_sdrs=band_sdrs
        )
        
        # Calculate scores
        metrics.research_score = sdr  # Use SDR directly for research comparison
        metrics.detailed_score = self._calculate_detailed_score(metrics, genre)
        
        return metrics
    
    def _calculate_detailed_score(
        self,
        metrics: SegmentMetrics,
        genre: Optional[str] = None
    ) -> float:
        """Calculate detailed score with optional genre weighting"""
        if genre and genre in GENRE_WEIGHTS:
            weights = GENRE_WEIGHTS[genre]
        else:
            weights = DEFAULT_WEIGHTS
            
        score = (
            weights['clarity'] * metrics.vocal_clarity +
            weights['high_freq'] * metrics.high_freq_content +
            weights['dynamic'] * metrics.dynamic_range +
            weights['phase'] * metrics.phase_complexity +
            weights['transition'] * metrics.transition_score +
            weights['sdr'] * metrics.sdr  # Include SDR in detailed score
        )
        
        return score