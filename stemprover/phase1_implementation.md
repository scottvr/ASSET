# StemProver Phase 1 Implementation

This document describes the Phase 1 implementation of the StemProver project, focusing on the **Synthetic Artifact Generation & Validation** component.

## Overview

Phase 1 implements the infrastructure required to generate, analyze, and eventually fix various types of artifacts that can occur in AI-based audio stem separation. By creating synthetic test data with controlled artifacts, we can validate artifact reduction approaches before moving to more expensive real-world training data.

## Key Components

### 1. Artifact Generator

The `ArtifactGenerator` class (in `stemprover/src/stemprover/analysis/artifacts/base.py`) provides methods to apply various types of synthetic artifacts to clean audio, including:

- **Phase Incoherence**: Inconsistencies in phase relationships between channels or frequency bands
- **High Frequency Ringing**: Artificial high-frequency content often created by separation algorithms
- **Temporal Smearing**: Time-domain blurring of transient sounds
- **Spectral Holes**: Missing frequency content in specific bands
- **Quantization Noise**: Digital quantization artifacts from processing
- **Aliasing**: Frequency aliasing due to downsampling or poor reconstruction
- **Background Bleed**: Residual content from other stems bleeding through
- **Transient Suppression**: Reduction or loss of sharp transients

Each artifact type can be controlled with parameters like intensity, frequency range, and time range.

### 2. Enhanced MPD (Music Programmatic Dataset)

The enhanced `MPD/enhanced_wavgen.py` script extends the original MPD approach to incorporate the new artifact generation capabilities. This allows for:

- Creating synthetic stems with specific waveforms (sine, square, sawtooth, etc.)
- Applying controlled artifacts to these stems
- Generating clean and artifacted mixes
- Creating test cases with variations for algorithm validation

### 3. Artifact Analysis Tools

The following analysis tools have been implemented:

- **ArtifactDetector** (`stemprover/src/stemprover/analysis/artifacts/detector.py`): Detects and quantifies different types of artifacts in audio, with and without clean reference
- **ArtifactVisualizer** (`stemprover/src/stemprover/analysis/artifacts/visualization.py`): Creates visualizations for different artifact types, showing spectrograms, phase relationships, and other relevant metrics

### 4. Demo Script

The `stemprover/demo_artifact_analysis.py` script demonstrates the capabilities of the Phase 1 implementation:

- Generating synthetic audio with controlled artifacts
- Analyzing and visualizing different artifact types
- Detecting artifacts in real audio files
- Comparing clean and artifacted audio files

## Using the Implementation

### Generating Synthetic Test Data

```bash
python -m stemprover.demo_artifact_analysis --generate --output artifact_examples
```

This will generate examples of each artifact type, along with visualizations and analysis.

### Analyzing an Audio File

```bash
python -m stemprover.demo_artifact_analysis --analyze path/to/audio.wav --output analysis_output
```

This performs blind artifact detection on an audio file without requiring a clean reference.

### Comparing Clean and Artifacted Audio

```bash
python -m stemprover.demo_artifact_analysis --compare clean.wav artifacted.wav --output comparison_output
```

This analyzes the differences between clean and artifacted audio, identifying the types and locations of artifacts.

### Creating Custom Artifact Configurations

You can create custom test configurations using the JSON format in `MPD/artifact_example_config.json`. Run the enhanced wavgen script:

```bash
python MPD/enhanced_wavgen.py --config path/to/config.json
```

## Validation Metrics

The implementation includes comprehensive validation metrics:

- **Phase Coherence**: Measures phase relationship preservation
- **Frequency Response**: Analyzes frequency content preservation in different bands
- **Signal-to-Noise Ratio**: Quantifies the level of artifacts relative to the signal
- **Time-Frequency Analysis**: Specialized analysis for each artifact type

## Next Steps

With Phase 1 complete, the project is ready to move to Phase 2: Minimal ControlNet Implementation. The artifact generation and analysis tools developed in Phase 1 will be used to:

1. Generate training data with specific artifacts
2. Validate the effectiveness of enhancement approaches
3. Provide metrics for model performance evaluation

## Requirements

- Python 3.8+
- Libraries: numpy, scipy, librosa, soundfile, matplotlib, torch
- Install package in development mode: `pip install -e .`