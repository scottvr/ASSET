# StemProver Development Plan

## Current Project Status

**Goal of Project**: StemProver aims to reduce artifacts in audio stems separated through AI tools (like Spleeter, Demucs, etc.) by leveraging latent diffusion models with phase-aware processing, treating spectrograms as images to utilize powerful image-processing AI.

### Code Maturity Assessment:

**Most Mature Components**:
1. Core audio data handling infrastructure (AudioSegment class, basic processing)
2. Audio analysis components (spectral analysis, basic artifact detection)
3. Music Programmatic Dataset (MPD) testing pipeline for synthetic waveform generation

**Least Mature Components**:
1. Phase-aware ControlNet implementation (commented-out imports suggest early stage)
2. Training pipeline and dataset generation
3. PhaseAnalyzer (explicitly marked as TODO)
4. Model integration with diffusion-based enhancement

## Development Plan

### Phase 1: Synthetic Artifact Generation & Validation (4-6 weeks)

**Goal**: Enhance the MPD approach to generate synthetic data with specific artifact types that closely match real-world separation artifacts.

1. **Enhance MPD Synthetic Test Dataset (2-3 weeks)**
   - Extend waveform generator to create specific, controlled artifact types
   - Analyze real stem separations to identify and categorize common artifacts
   - Create programmatic versions of these artifacts in synthetic data
   - Implement comprehensive phase-aware metrics for evaluation

2. **Build Artifact Characterization Tools (1-2 weeks)**
   - Create tools to analyze and classify artifacts in both synthetic and real separated stems
   - Implement visualization for comparing artifacts across datasets
   - Develop metrics to measure similarity between synthetic and real artifacts

3. **Develop Basic Prototype Enhancement Pipeline (1-2 weeks)**
   - Create simple signal processing approaches to fix specific artifact types
   - Implement baseline methods for comparison
   - Set up evaluation pipeline for quick iteration

### Phase 2: Minimal ControlNet Implementation (6-8 weeks)

**Goal**: Create a minimal working prototype of phase-aware ControlNet for artifact reduction on synthetic data.

1. **Spectrogram Processing Infrastructure (2 weeks)**
   - Implement phase-preserving spectrogram transformations
   - Create efficient processing pipeline for spectrograms
   - Build visualization tools for spectrogram comparison

2. **Core ControlNet Integration (3-4 weeks)**
   - Implement minimal phase-aware ControlNet for spectrogram enhancement
   - Create adapter to interface with pre-trained diffusion models
   - Implement spectrogram-to-audio conversion preserving phase

3. **MPD Validation (1-2 weeks)**
   - Test ControlNet approach on synthetic data
   - Compare with baseline methods
   - Iterate on architecture based on results

### Phase 3: Mixed Data Validation (4-6 weeks)

**Goal**: Validate the approach on a mix of synthetic and real data to prove transferability.

1. **Real Data Preprocessing (1-2 weeks)**
   - Process selected stems from the existing hundreds of songs
   - Identify segments with clear artifacts for targeted testing
   - Create paired examples (clean/separated) for validation

2. **Transfer Testing (2-3 weeks)**
   - Test synthetic-trained models on real data segments
   - Measure performance gap between synthetic and real
   - Identify areas needing improvement for real-world application

3. **Economic Viability Analysis (1 week)**
   - Calculate potential training costs at various scales
   - Estimate model performance vs. cost tradeoffs
   - Determine go/no-go criteria for full-scale development

### Phase 4: Accelerated Platform Prototype (Conditional on Phase 3)

**Goal**: If Phase 3 validates the approach, run limited training on accelerated platform to prove full-scale viability.

1. **Limited Training Dataset Preparation (2-3 weeks)**
   - Prepare balanced dataset of synthetic and real examples
   - Focus on segments with high-quality ground truth and clear artifacts
   - Create efficient data loading pipeline for accelerated training

2. **Cloud Platform Setup (1-2 weeks)**
   - Set up training environment on chosen cloud platform
   - Implement efficient data pipeline for GPU/TPU training
   - Create monitoring and evaluation tools

3. **Limited Model Training (2-3 weeks)**
   - Train for limited epochs to validate approach
   - Measure performance improvements over baseline
   - Estimate full training requirements based on learning curves

### Phase 5: Full Model Development (Conditional on Phase 4)

**Goal**: If economically viable, implement full-scale training on an accelerated platform.

1. **Full Training Dataset Preparation (3-4 weeks)**
   - Curate comprehensive dataset from available songs
   - Generate separated stems with multiple separation methods
   - Create efficient training pairs with emphasis on diverse artifacts

2. **Scale Up Training (6-8 weeks)**
   - Deploy full training pipeline to accelerated platform
   - Implement progressive training for different artifact types
   - Train multiple specialized models if needed

3. **Evaluation and Production Readiness (3-4 weeks)**
   - Perform comprehensive evaluation on held-out test set
   - Optimize models for production deployment
   - Create user-friendly inference pipeline

## Implementation Priorities

These components need immediate attention:

1. **Enhance MPD for targeted artifact generation**
   - Add specific artifact types that mimic real-world separation issues
   - Create validation tools to compare synthetic vs real artifacts
   - Implement efficient generation pipeline for large-scale synthetic data

2. **Complete phase analysis infrastructure**
   - Implement missing PhaseAnalyzer class
   - Consolidate phase-related utilities
   - Create visualization tools for phase relationships

3. **Develop minimal ControlNet prototype**
   - Complete the architecture from enhancement/controlnet.py
   - Integrate with spectral processing pipeline
   - Create efficient inference pipeline

## Cost-Effective Validation Strategy

The key to maximizing return on investment is to validate as many assumptions as possible with the inexpensive MPD approach before committing to expensive training:

1. **Validate artifact generation**: Ensure synthetic artifacts match real-world artifacts in key characteristics
2. **Validate enhancement approach**: Test whether the phase-aware ControlNet can actually fix synthetic artifacts
3. **Validate transferability**: Test if improvements on synthetic data transfer to real data
4. **Validate economic viability**: Calculate training costs vs. expected quality improvements

This approach minimizes risk by ensuring each expensive step is preceded by cheaper validation steps.

## Conclusion

With the availability of hundreds of songs with studio stems, the project is well-positioned to succeed. The MPD approach provides an excellent way to validate core assumptions inexpensively before committing to full-scale training. By carefully building and validating the synthetic data approach first, we can minimize the risk and cost of the full model training while maximizing the likelihood of producing a high-quality artifact reduction system.