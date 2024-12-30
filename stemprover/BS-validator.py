def run_minimal_validation(
    test_file: str,
    band_configs: List[Dict[str, Tuple[float, float]]],
    duration: float = 30.0  # 30 second sample
) -> Dict[str, Dict[str, float]]:
    """
    Run minimal validation using existing tools and one test file
    
    Args:
        test_file: Path to a mixed audio file we already have stems for
        band_configs: List of different frequency band configurations to test
        duration: Length of audio to analyze (shorter = faster)
    """
    results = {}
    
    # Load and prepare audio
    audio_segment = AudioSegment(
        audio=load_audio_file(test_file)[0],
        sample_rate=44100
    )
    
    # Truncate to desired duration
    samples = int(duration * 44100)
    audio_segment.audio = audio_segment.audio[:samples]
    
    for config_name, bands in band_configs.items():
        # Create spectral analyzer with this band configuration
        analyzer = SpectralAnalyzer(
            output_dir=None,  # Don't save files
            config=ProcessingConfig(
                sample_rate=44100,
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
        band_metrics = analyzer.calculate_band_isolation(
            audio_segment
        )
        
        results[config_name] = {
            'energy_distribution': analysis['energy_distribution'],
            'phase_coherence': analysis['phase_coherence'],
            'band_isolation': band_metrics,
            'processing_time': analysis['processing_time']
        }
    
    return results

# Example usage
if __name__ == "__main__":
    # Define configurations to test
    original_bands = {
        "band1": (0, 4000),
        "band2": (4000, 8000),
        "band3": (8000, 12000),
        "band4": (12000, 20000)
    }
    
    musical_bands = DEFAULT_FREQUENCY_BANDS
    
    # Run validation
    results = run_minimal_validation(
        test_file="path/to/existing/test/file.wav",
        band_configs={
            'original': original_bands,
            'musical': musical_bands
        }
    )
    
    # Generate simple report
    print("\nValidation Results:")
    print("-" * 50)
    for config, metrics in results.items():
        print(f"\n{config} configuration:")
        print(f"Band Isolation Score: {metrics['band_isolation']:.3f}")
        print(f"Phase Coherence: {metrics['phase_coherence']:.3f}")
        print(f"Processing Time: {metrics['processing_time']:.3f}s")
        
        print("\nEnergy Distribution:")
        for band, energy in metrics['energy_distribution'].items():
            print(f"  {band}: {energy:.2f}%")