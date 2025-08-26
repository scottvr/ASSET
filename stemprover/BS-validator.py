from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple
from pathlib import Path
import time
import json

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

        energy_dist = {k: float(v) for k, v in analysis['energy_distribution'].items()}
        results[config_name] = BandValidationResult(
            energy_distribution=energy_dist,
            phase_coherence=float(analysis.get('phase_coherence', 0.0)),
            band_isolation=float(band_isolation),
            processing_time=float(analysis['processing_time'])
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

    # Use the clean mix from the golden dataset
    test_file = Path("golden_dataset/battery/mix_from_90.0s.wav")
    output_file = Path("band_split_validation_results.json")

    # Run validation
    validation_results = run_minimal_validation(
        test_file=test_file,
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

    # Save results to a file
    results_dict = {k: asdict(v) for k, v in validation_results.items()}
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"\nResults saved to {output_file}")
