#!/usr/bin/env python
"""
Artifact Detection and Analysis Demo

This script demonstrates the capabilities of the Phase 1 implementation:
1. Generate synthetic audio with controlled artifacts
2. Analyze and visualize different artifact types
3. Detect artifacts in real audio files

Usage:
    python demo_artifact_analysis.py --generate  # Generate and analyze synthetic test data
    python demo_artifact_analysis.py --analyze path/to/file.wav  # Analyze existing audio file
    python demo_artifact_analysis.py --compare clean.wav artifact.wav  # Compare clean and artifacted audio
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import soundfile as sf
import json
import tempfile
import librosa
import sys
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import stemprover components
from stemprover.core.audio import AudioSegment
from stemprover.core.config import ProcessingConfig
from stemprover.analysis.artifacts.base import (
    ArtifactType, ArtifactParameters, ArtifactGenerator
)
from stemprover.analysis.artifacts.detector import (
    ArtifactDetector, ArtifactDetectionResult
)
from stemprover.analysis.artifacts.visualization import (
    ArtifactVisualizer, ArtifactVisualizationConfig
)


def generate_test_data(output_dir: Path):
    """Generate synthetic test data with various artifacts"""
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create processing config
    config = ProcessingConfig(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512
    )
    
    # Create a clean sine wave
    duration = 5.0  # seconds
    sample_rate = config.sample_rate
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Create a more complex signal with multiple frequencies
    signal = np.zeros(len(t))
    for freq in [440, 880, 1320]:  # A4, A5, E6
        signal += 0.2 * np.sin(2 * np.pi * freq * t)
    
    # Add a transient
    transient_pos = int(sample_rate * 2.5)  # at 2.5 seconds
    transient_width = int(sample_rate * 0.01)  # 10ms
    transient = np.linspace(0, 1, transient_width) * np.sin(2 * np.pi * 2000 * np.linspace(0, 0.01, transient_width))
    signal[transient_pos:transient_pos + transient_width] += transient
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Create stereo signal
    stereo_signal = np.stack([signal, signal])
    
    # Create clean audio segment
    clean_audio = AudioSegment(audio=stereo_signal, sample_rate=sample_rate)
    
    # Save clean audio
    clean_path = output_dir / "clean_reference.wav"
    sf.write(str(clean_path), stereo_signal.T, sample_rate)
    print(f"Generated clean reference audio: {clean_path}")
    
    # Initialize artifact generator
    artifact_generator = ArtifactGenerator(sample_rate=sample_rate)
    
    # Generate examples of each artifact type
    for artifact_type in ArtifactType:
        print(f"Generating {artifact_type.name} artifact...")
        
        # Create parameters for this artifact
        params = ArtifactParameters(
            artifact_type=artifact_type,
            intensity=0.7,  # Higher intensity for clear demonstration
            frequency_range=(500, 8000) if artifact_type == ArtifactType.SPECTRAL_HOLES else None
        )
        
        # For BACKGROUND_BLEED, we need a second signal
        if artifact_type == ArtifactType.BACKGROUND_BLEED:
            # Create a noise signal
            noise = np.random.normal(0, 0.05, size=stereo_signal.shape)
            params.additional_params['bleed_audio'] = noise
        
        # Apply artifact
        artifacted_audio = artifact_generator.apply_artifact(clean_audio, params)
        
        # Save artifacted audio
        artifact_path = output_dir / f"artifact_{artifact_type.name.lower()}.wav"
        sf.write(str(artifact_path), artifacted_audio.audio.T, sample_rate)
        print(f"Generated {artifact_type.name} artifact: {artifact_path}")
        
        # Create visualization
        viz_config = ArtifactVisualizationConfig()
        visualizer = ArtifactVisualizer(output_dir=output_dir, config=config, viz_config=viz_config)
        
        viz_path = visualizer.visualize_artifact(
            clean=clean_audio,
            artifact=artifacted_audio,
            artifact_type=artifact_type
        )
        print(f"Created visualization: {viz_path}")
        
        # Detect artifacts
        detector = ArtifactDetector(config=config)
        results = detector.detect_artifacts(clean_audio, artifacted_audio)
        
        # Save detection results
        results_path = output_dir / f"detection_{artifact_type.name.lower()}.json"
        with open(results_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"Saved detection results: {results_path}")
    
    # Create a visualization comparing all artifact types
    artifacts = {}
    for artifact_type in ArtifactType:
        artifact_path = output_dir / f"artifact_{artifact_type.name.lower()}.wav"
        if artifact_path.exists():
            audio_data, sr = sf.read(str(artifact_path))
            artifact_audio = AudioSegment(
                audio=audio_data.T if audio_data.ndim > 1 else audio_data.reshape(1, -1),
                sample_rate=sr
            )
            artifacts[artifact_type] = artifact_audio
    
    # Compare all artifacts
    comparison_path = visualizer.compare_artifact_types(
        reference=clean_audio,
        artifacts=artifacts
    )
    print(f"Created comparison visualization: {comparison_path}")
    
    print("\nGeneration complete! Analysis files saved to:", output_dir)


def analyze_audio_file(audio_path: Path, output_dir: Path = None):
    """Analyze artifacts in an audio file without reference"""
    # Create output directory if not provided
    if output_dir is None:
        output_dir = audio_path.parent / f"{audio_path.stem}_analysis"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load audio file
    audio_data, sample_rate = sf.read(str(audio_path))
    
    # Convert to proper format
    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(1, -1)
    elif audio_data.ndim == 2 and audio_data.shape[1] < audio_data.shape[0]:
        # Transpose if needed (samples, channels) -> (channels, samples)
        audio_data = audio_data.T
    
    audio = AudioSegment(audio=audio_data, sample_rate=sample_rate)
    
    # Create processing config
    config = ProcessingConfig(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512
    )
    
    # Detect artifacts
    print("Detecting artifacts...")
    detector = ArtifactDetector(config=config)
    results = detector.detect_artifacts_blind(audio)
    
    # Print detection results
    print("\nArtifact Detection Results:")
    print(f"Overall severity: {results.overall_severity:.2f}")
    print("\nDetected artifact types:")
    
    for artifact_type, confidence in sorted(
        results.artifact_types.items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        print(f"- {artifact_type.name}: {confidence:.2f}")
    
    # Save detection results
    results_path = output_dir / "artifact_detection.json"
    with open(results_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\nSaved detection results to: {results_path}")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Create spectrogram
    plt.figure(figsize=(12, 8))
    audio_mono = audio.audio.mean(axis=0)
    D = librosa.stft(audio_mono, n_fft=config.n_fft, hop_length=config.hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    librosa.display.specshow(
        S_db, 
        x_axis='time', 
        y_axis='log',
        sr=sample_rate,
        hop_length=config.hop_length
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Save spectrogram
    spec_path = output_dir / "spectrogram.png"
    plt.savefig(spec_path, dpi=150)
    plt.close()
    
    print(f"Saved spectrogram to: {spec_path}")
    
    # Create analysis summary
    summary_path = output_dir / "analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Artifact Analysis for: {audio_path.name}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall artifact severity: {results.overall_severity:.2f}\n\n")
        f.write("Detected artifact types:\n")
        
        for artifact_type, confidence in sorted(
            results.artifact_types.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            f.write(f"- {artifact_type.name}: {confidence:.2f}\n")
        
        f.write("\nFrequency ranges affected:\n")
        for artifact_type, ranges in results.freq_ranges.items():
            if ranges:
                f.write(f"{artifact_type.name}:\n")
                for low, high in ranges:
                    f.write(f"  {low:.0f} Hz - {high:.0f} Hz\n")
    
    print(f"Saved analysis summary to: {summary_path}")
    print("\nAnalysis complete!")


def compare_audio_files(clean_path: Path, artifact_path: Path, output_dir: Path = None):
    """Compare clean and artifacted audio files"""
    # Create output directory if not provided
    if output_dir is None:
        output_dir = artifact_path.parent / f"{artifact_path.stem}_comparison"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load audio files
    clean_data, clean_sr = sf.read(str(clean_path))
    artifact_data, artifact_sr = sf.read(str(artifact_path))
    
    # Check sample rates
    if clean_sr != artifact_sr:
        print(f"Warning: Sample rates differ ({clean_sr} vs {artifact_sr})")
        print("Resampling to match...")
        if clean_sr > artifact_sr:
            # Resample artifact to match clean
            artifact_data = librosa.resample(
                artifact_data.T if artifact_data.ndim > 1 else artifact_data, 
                orig_sr=artifact_sr, 
                target_sr=clean_sr
            )
            if artifact_data.ndim == 1:
                artifact_data = artifact_data.reshape(-1, 1).T
            sample_rate = clean_sr
        else:
            # Resample clean to match artifact
            clean_data = librosa.resample(
                clean_data.T if clean_data.ndim > 1 else clean_data,
                orig_sr=clean_sr,
                target_sr=artifact_sr
            )
            if clean_data.ndim == 1:
                clean_data = clean_data.reshape(-1, 1).T
            sample_rate = artifact_sr
    else:
        sample_rate = clean_sr
    
    # Convert to proper format
    if clean_data.ndim == 1:
        clean_data = clean_data.reshape(1, -1)
    elif clean_data.ndim == 2 and clean_data.shape[1] < clean_data.shape[0]:
        clean_data = clean_data.T
    
    if artifact_data.ndim == 1:
        artifact_data = artifact_data.reshape(1, -1)
    elif artifact_data.ndim == 2 and artifact_data.shape[1] < artifact_data.shape[0]:
        artifact_data = artifact_data.T
    
    # Match lengths
    min_length = min(clean_data.shape[1], artifact_data.shape[1])
    clean_data = clean_data[:, :min_length]
    artifact_data = artifact_data[:, :min_length]
    
    # Create audio segments
    clean_audio = AudioSegment(audio=clean_data, sample_rate=sample_rate)
    artifact_audio = AudioSegment(audio=artifact_data, sample_rate=sample_rate)
    
    # Create processing config
    config = ProcessingConfig(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512
    )
    
    # Detect artifacts
    print("Detecting artifacts...")
    detector = ArtifactDetector(config=config)
    results = detector.detect_artifacts(clean_audio, artifact_audio)
    
    # Print detection results
    print("\nArtifact Detection Results:")
    print(f"Overall severity: {results.overall_severity:.2f}")
    print("\nDetected artifact types:")
    
    for artifact_type, confidence in sorted(
        results.artifact_types.items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        print(f"- {artifact_type.name}: {confidence:.2f}")
    
    # Get most likely artifact
    most_likely, confidence = results.get_most_likely_artifact()
    print(f"\nMost likely artifact: {most_likely.name} ({confidence:.2f})")
    
    # Create visualizations
    print("Creating visualizations...")
    viz_config = ArtifactVisualizationConfig()
    visualizer = ArtifactVisualizer(output_dir=output_dir, config=config, viz_config=viz_config)
    
    # Visualize the most likely artifact
    if most_likely:
        viz_path = visualizer.visualize_artifact(
            clean=clean_audio,
            artifact=artifact_audio,
            artifact_type=most_likely
        )
        print(f"Created visualization: {viz_path}")
    
    # Create frequency band analysis
    frequency_bands = {
        "bass": (20, 250),
        "low_mid": (250, 1000),
        "mid": (1000, 4000),
        "high_mid": (4000, 8000),
        "high": (8000, 20000)
    }
    
    bands_path = visualizer.visualize_artifact_bands(
        reference=clean_audio,
        artifact=artifact_audio,
        frequency_bands=frequency_bands
    )
    print(f"Created frequency band analysis: {bands_path}")
    
    # Save detection results
    results_path = output_dir / "artifact_detection.json"
    with open(results_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"Saved detection results to: {results_path}")
    
    # Create difference waveform
    diff_audio = artifact_audio.audio - clean_audio.audio
    diff_path = output_dir / "difference.wav"
    sf.write(str(diff_path), diff_audio.T, sample_rate)
    print(f"Saved difference audio to: {diff_path}")
    
    print("\nComparison complete!")


def main():
    parser = argparse.ArgumentParser(description="Artifact Detection and Analysis Demo")
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument("--generate", action="store_true", help="Generate synthetic test data")
    group.add_argument("--analyze", type=str, help="Analyze an audio file for artifacts")
    group.add_argument("--compare", nargs=2, metavar=("CLEAN", "ARTIFACT"), 
                      help="Compare clean and artifacted audio files")
    
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    if args.generate:
        output_dir = Path(args.output) if args.output else Path("artifact_examples")
        generate_test_data(output_dir)
    
    elif args.analyze:
        audio_path = Path(args.analyze)
        if not audio_path.exists():
            print(f"Error: File not found: {audio_path}")
            return 1
        
        output_dir = Path(args.output) if args.output else None
        analyze_audio_file(audio_path, output_dir)
    
    elif args.compare:
        clean_path = Path(args.compare[0])
        artifact_path = Path(args.compare[1])
        
        if not clean_path.exists():
            print(f"Error: Clean file not found: {clean_path}")
            return 1
        
        if not artifact_path.exists():
            print(f"Error: Artifact file not found: {artifact_path}")
            return 1
        
        output_dir = Path(args.output) if args.output else None
        compare_audio_files(clean_path, artifact_path, output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
