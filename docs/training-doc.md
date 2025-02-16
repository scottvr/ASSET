# StemProver Training Process Documentation

## Input Pipeline

### Data Sources
1. **Pristine Stems**
   - High-quality source material
   - Used for ground truth phase relationships
   - Source for generating training mixes

2. **Generated Mixes**
   - Created from pristine stems
   - Multiple mixing strategies to ensure robustness
   - Controlled degradation for training variety

3. **Separated Stems**
   - Examples of artifacts we aim to fix
   - Generated using various separation methods
   - Annotated with artifact types and locations

### Feature Extraction
1. **Spectrogram Generation**
   - Multi-resolution spectrograms for different frequency ranges
   - Phase-magnitude separation
   - Time-frequency analysis

2. **Phase Analysis**
   - Phase relationship extraction between stems and mix
   - Coherence measurements
   - Partial overlap detection

3. **Artifact Detection**
   - Spectral artifact identification
   - Phase inconsistency detection
   - Quality metrics computation

## Training Tasks

### 1. Phase Prediction
- Learn relationship between mix phase and stem phases
- Predict phase characteristics without full separation
- Focus on areas with overlapping partials

### 2. Artifact Detection
- Identify common separation artifacts
- Learn artifact signatures in both magnitude and phase
- Develop robust detection across different separators

### 3. Enhancement Training
- Primary task combining phase and artifact knowledge
- Progressive refinement of enhancement quality
- Balance between correction and preservation

## Loss Functions

### Phase Coherence Loss
- Measures accuracy of phase relationships
- Weighted by perceptual importance
- Focuses on critical frequency bands

### Spectral Loss
- Complex spectrum reconstruction
- Magnitude-phase balance
- Frequency-dependent weighting

### Perceptual Loss
- Audio quality metrics
- Human perception modeling
- Cross-frequency interactions

### Reconstruction Loss
- Direct waveform comparison
- Time-domain accuracy
- Artifact-specific penalties

## Training Strategy

### Phase 1: Base Enhancement
- Focus on basic artifact reduction
- Establish baseline performance
- Validate core architecture

### Phase 2: Phase-Aware Training
- Introduce phase prediction tasks
- Integrate phase information
- Refine phase relationships

### Phase 3: Joint Optimization
- Combine all training objectives
- Balance different loss components
- Optimize for real-world usage

### Phase 4: Fine-Tuning
- Specialize for specific use cases
- Optimize inference modes
- Performance optimization

## Validation Pipeline

### Metrics
1. **Phase Accuracy**
   - Phase coherence measurements
   - Relationship preservation
   - Partial handling accuracy

2. **Artifact Reduction**
   - Artifact removal success
   - Prevention of new artifacts
   - Quality consistency

3. **Audio Quality**
   - Perceptual quality metrics
   - Listening tests
   - Professional evaluation

4. **Generalization**
   - Cross-separator performance
   - Robustness to different inputs
   - Edge case handling

## Implementation Notes

### essential training flow
![basic diagram](https://github.com/scottvr/ASSET/blob/main/docs/stemprover-training_diagram-0.2.1.png)

### detailed training diagram
at least, this is the plan as of now.

![detailed training diagram](https://github.com/scottvr/ASSET/blob/main/docs/stemprover-training_diagram-2025-01-04-194124.png) (at least, this is the plan as of now.)

### Critical Considerations
1. Balance between phase awareness and artifact reduction
2. Efficient training pipeline for large datasets
3. Robust validation across different scenarios
4. Modular design for future expansion

### Performance Optimization
1. Batch processing strategies
2. Memory management for large spectrograms
3. Efficient loss computation
4. Training acceleration techniques
