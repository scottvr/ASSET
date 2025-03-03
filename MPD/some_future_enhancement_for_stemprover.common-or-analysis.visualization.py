# In stemprover.common.visualization or stemprover.analysis.visualization
# TODO (TBD):
# The processing, padding, and spectral analysis code seems to have evolved in parallel in different parts of the codebase as new needs arose.
#  consider a dedicated visualization/processing utilities module that all these components can share
# and here's something new to consider, after writing the padding code for the tools in MPD/

def create_consistent_spectrograms(
    audio_segments: Dict[str, AudioSegment],
    config: ProcessingConfig,
    normalize_lengths: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate spectrograms with consistent dimensions from multiple audio segments.
    
    Args:
        audio_segments: Dictionary of named AudioSegment objects
        config: ProcessingConfig containing FFT parameters
        normalize_lengths: If True, pad shorter segments to match the longest
        
    Returns:
        Dictionary of spectrograms with consistent dimensions
    """
    # Implementation using the ProcessingConfig parameters
    # ...
