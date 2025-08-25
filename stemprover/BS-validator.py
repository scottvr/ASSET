from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
import time

from stemprover.core.audio import AudioSegment
from stemprover.core.types import ProcessingConfig
from stemprover.analysis.spectral import SpectralAnalyzer
from stemprover.io.audio import load_audio_file
from stemprover.types import DEFAULT_FREQUENCY_BANDS, FrequencyBands

@dataclass
class BandValidationResult:
    """Dataclass to hold the validation results for a single band configuration."""
    energy_distribution: Dict[str, float]
    phase_coherence: float
    band_isolation: float
    processing_time: float

def run_minimal_validation(
    test_file: Path,
    band_configs: Dict[str, FrequencyBands],
    duration: float = 30.0
) -> Dict[str, BandValidationResult]:
    """
    Run minimal validation using existing tools and one test file.

    Args:
        test_file: Path to a mixed audio file.
        band_configs: Dictionary of frequency band configurations to test.
        duration: Length of audio to analyze in seconds.
    """
    results = {}

    # Load audio file
    audio_array, sample_rate = load_audio_file(test_file, mono=True)

    # Create and slice AudioSegment
    full_segment = AudioSegment(audio=audio_array, sample_rate=sample_rate)
    audio_segment = full_segment.slice(0, duration)

    for config_name, bands in band_configs.items():
        # Create spectral analyzer with this band configuration
        analyzer = SpectralAnalyzer(
            output_dir=Path("./temp_analysis"),  # Provide a temporary dir
            config=ProcessingConfig(
                sample_rate=audio_segment.sample_rate,
                n_fft=2048,
                hop_length=512
            ),
            frequency_bands=bands
        )

        # Analyze energy distribution and phase coherence
        analysis = analyzer.analyze_frequency_distribution(
            audio_segment,
            preserve_phase=True
        )

        # Calculate band separation metrics
        band_isolation = analyzer.calculate_band_isolation(
            audio_segment
        )

        results[config_name] = BandValidationResult(
            energy_distribution=analysis['energy_distribution'],
            phase_coherence=analysis.get('phase_coherence', 0.0),
            band_isolation=band_isolation,
            processing_time=analysis['processing_time']
        )

    return results

# Example usage
if __name__ == "__main__":
    # Define configurations to test
    original_bands: FrequencyBands = {
        "band1": (0, 4000),
        "band2": (4000, 8000),
        "band3": (8000, 12000),
        "band4": (12000, 20000)
    }

    # It's better to have a real file for testing, but for now, we'll just mock it.
    # In a real scenario, you would replace this with a path to an actual audio file.
    # For the script to run without errors, we'll create a dummy file.
    import numpy as np
    import soundfile as sf
    dummy_file = Path("dummy_audio.wav")
    if not dummy_file.exists():
        sr = 44100
        dummy_audio = np.random.randn(sr * 35) # 35 seconds of noise
        sf.write(dummy_file, dummy_audio, sr)

    # Run validation
    validation_results = run_minimal_validation(
        test_file=dummy_file,
        band_configs={
            'original': original_bands,
            'musical': DEFAULT_FREQUENCY_BANDS
        }
    )

    # Generate simple report
    print("\nValidation Results:")
    print("-" * 50)
    for config, metrics in validation_results.items():
        print(f"\n{config} configuration:")
        print(f"Band Isolation Score: {metrics.band_isolation:.3f}")
        print(f"Phase Coherence: {metrics.phase_coherence:.3f}")
        print(f"Processing Time: {metrics.processing_time:.3f}s")

        print("\nEnergy Distribution:")
        for band, energy in metrics.energy_distribution.items():
            print(f"  {band}: {energy:.2f}%")
