from dataclasses import dataclass

@dataclass
class SegmentMetrics:
    """Metrics for evaluating segment suitability"""
    vocal_clarity: float      # Ratio of vocal band energy to total
    high_freq_content: float  # Energy above 11kHz
    dynamic_range: float      # dB difference between peaks
    phase_complexity: float   # Measure of phase relationships
    transition_score: float   # Score for vocal transitions
    score: float = 0.0       # Overall suitability score