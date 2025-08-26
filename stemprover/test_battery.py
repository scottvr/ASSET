import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from stemprover import (ProcessingConfig, SpectralAnalyzer,
                        SpleeterSeparator)
from stemprover.analysis.selection.segment_finder import (FoundSegment,
                                                          find_best_segments)
from stemprover.core.audio import AudioSegment
from stemprover.io.audio import load_audio_file
from stemprover.types import DEFAULT_FREQUENCY_BANDS


@dataclass
class TimeRange:
    start: float
    end: float


@dataclass
class SegmentAnalysis:
    time_range: TimeRange
    metrics: FoundSegment
    spectral_analysis: dict


def run_battery_test(
    mix_path: Path,
    target_stem_path: Path,
    output_dir: Path,
    segment_length: float = 5.0,
    segment_hop: float = 2.5,
    top_k: int = 5
):
    """
    Runs a battery test using a mix, a target stem, and an output directory.
    """
    # 1. Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ProcessingConfig(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512
    )

    separator = SpleeterSeparator(str(output_dir / "separation"))
    analyzer = SpectralAnalyzer(
        output_dir / "analysis",
        config=config,
        frequency_bands=DEFAULT_FREQUENCY_BANDS
    )

    # 2. Load Audio
    print(f"Loading mix: {mix_path}")
    mix_audio, sr = load_audio_file(mix_path)
    print(f"Loading target stem: {target_stem_path}")
    target_stem_audio, _ = load_audio_file(target_stem_path)

    # Ensure audio is stereo for the separator
    if mix_audio.ndim == 1:
        mix_audio = np.stack([mix_audio, mix_audio])
    if target_stem_audio.ndim == 1:
        target_stem_audio = np.stack([target_stem_audio, target_stem_audio])

    mix_segment = AudioSegment(mix_audio, sr)
    target_stem_segment = AudioSegment(target_stem_audio, sr)

    # 3. Separate the full mix
    print("\nSeparating full mix...")
    # NOTE: SpleeterSeparator from the library separates the mix into 'vocal' and 'accompaniment'
    # We assume the 'vocal' part corresponds to our target stem.
    separation_result = separator.separate(mix_segment)

    # The result from `separate` is a SeparationResult object.
    separated_vocal_segment = separation_result.separated_vocal
    if separated_vocal_segment is None:
        raise ValueError("Separator did not return a vocal stem.")

    # 4. Find best segments for analysis
    print("\nFinding best segments for analysis...")
    best_segments = find_best_segments(
        vocal_track=target_stem_segment, # Ground truth
        backing_track=separated_vocal_segment, # What the separator produced
        segment_length_sec=segment_length,
        hop_length_sec=segment_hop,
        config=config,
        top_k=top_k
    )

    if not best_segments:
        print("No suitable segments found for analysis.")
        return

    # 5. Analyze each best segment in detail
    segment_analyses = []
    for idx, seg in enumerate(best_segments):
        start_time = seg.time
        end_time = start_time + segment_length

        print(f"\nAnalyzing segment {idx + 1}/{len(best_segments)} at {start_time:.2f}s")

        # Slice the already loaded/separated audio instead of re-reading/re-separating
        clean_slice = target_stem_segment.slice(start_time, end_time)
        separated_slice = separated_vocal_segment.slice(start_time, end_time)

        # Perform spectral analysis on the slice
        analysis_path = analyzer.analyze(
            clean_slice,
            separated_slice
        )

        # Load and store analysis results
        with open(analysis_path / "analysis.json", 'r') as f:
            analysis_data = json.load(f)
            segment_analyses.append(SegmentAnalysis(
                time_range=TimeRange(start=start_time, end=end_time),
                metrics=seg,
                spectral_analysis=analysis_data
            ))

    # 6. Save consolidated results
    results_path = output_dir / "segment_analysis.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'mix_path': str(mix_path),
                'target_stem_path': str(target_stem_path),
                'segment_length': segment_length,
                'segment_hop': segment_hop,
                'top_k': top_k
            },
            'segments': [asdict(s) for s in segment_analyses]
        }, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    # Define file paths
    golden_dataset_dir = Path("golden_dataset")
    mix_path = golden_dataset_dir / "clean_mix.wav"
    # For this test, we'll treat the sine wave as the "vocal" stem
    target_stem_path = golden_dataset_dir / "stem_sine.wav"
    
    # Define output directory
    output_dir = Path("test_output")
    
    print("Starting battery test...")
    run_battery_test(
        mix_path=mix_path,
        target_stem_path=target_stem_path,
        output_dir=output_dir
    )
    print("Battery test finished.")
