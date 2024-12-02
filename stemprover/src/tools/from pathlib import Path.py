from pathlib import Path
from datetime import datetime
import torch

from separation.spleeter import SpleeterSeparator
from analysis.spectral import SpectralAnalyzer
from core.types import ProcessingConfig
from common.types import DEFAULT_FREQUENCY_BANDS

def run_battery_test(
    vocal_left: str,
    vocal_right: str,
    accompaniment_left: str,
    accompaniment_right: str,
    output_base: str = "battery_test",
    duration: float = 30.0
):
    """Run complete test pipeline on Battery stems with new common utils"""
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components with explicit frequency bands
    config = ProcessingConfig(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        # Add other settings as needed
    )
    
    separator = SpleeterSeparator(str(output_dir / "separation"))
    analyzer = SpectralAnalyzer(
        output_dir / "analysis",
        config=config,
        frequency_bands=DEFAULT_FREQUENCY_BANDS
    )
    
    try:
        # Process stems
        print("\nProcessing stems...")
        result = separator.separate_and_analyze(
            vocal_paths=(vocal_left, vocal_right),
            accompaniment_paths=(accompaniment_left, accompaniment_right),
            start_time=90.0,  # Same section as before
            duration=duration,
            run_analysis=True
        )
        
        # Additional spectral analysis
        print("\nRunning spectral analysis...")
        analysis_path = analyzer.analyze(result.clean_vocal, result.separated_vocal)
        
        print("\nProcessing complete!")
        print(f"Separation results: {output_dir}/separation")
        print(f"Analysis results: {analysis_path}")
        
        # Print frequency band analysis
        print("\nFrequency Band Analysis:")
        with open(analysis_path / "analysis.json", 'r') as f:
            import json
            analysis = json.load(f)
            for band, metrics in analysis.items():
                if band != "overall":
                    print(f"\n{band}:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.3f}")
            
            print("\nOverall Metrics:")
            for metric, value in analysis["overall"].items():
                print(f"  {metric}: {value:.3f}")
        
        return result, analysis_path
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Battery stem paths
    stems = {
        "vocal_l": "track-09.wav",
        "vocal_r": "track-10.wav",
        "accompaniment_l": "track-07.wav",
        "accompaniment_r": "track-08.wav"
    }
    
    # Update paths to match your system
    base_path = Path("battery_stems")
    stem_paths = {k: str(base_path / v) for k, v in stems.items()}
    
    # Run test
    result, analysis_path = run_battery_test(
        stem_paths["vocal_l"],
        stem_paths["vocal_r"],
        stem_paths["accompaniment_l"],
        stem_paths["accompaniment_r"]
    )