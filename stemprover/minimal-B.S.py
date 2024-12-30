class MinimalBandSplitValidator:
    def __init__(self):
        # Original band configuration
        self.original_bands = {
            "band1": (0, 4000),
            "band2": (4000, 8000),
            "band3": (8000, 12000),
            "band4": (12000, 20000)
        }
        
        # Our musical bands
        self.musical_bands = DEFAULT_FREQUENCY_BANDS
        
        # Track metrics for comparison
        self.metrics = ['sdr', 'sir']
        
    def process_stems(
        self,
        stems: Dict[str, np.ndarray],
        band_config: Dict[str, Tuple[float, float]]
    ) -> Dict[str, np.ndarray]:
        """Process stems using specified band configuration"""
        processed = {}
        
        for stem_name, audio in stems.items():
            # Split into bands
            bands = []
            for _, (low, high) in band_config.items():
                band = filter_to_band(audio, low, high)
                bands.append(band)
            
            # Recombine
            processed[stem_name] = np.sum(bands, axis=0)
        
        return processed
    
    def validate_single_track(
        self,
        track_path: str,
        separators: List[str] = ['spleeter', 'demucs']  # Use pre-trained
    ) -> Dict:
        """Run validation on a single track"""
        results = {
            'original': {},
            'musical': {}
        }
        
        # Get separator outputs
        separator_outputs = get_separator_outputs(track_path, separators)
        
        # Process with both band configurations
        for config_name, band_config in [
            ('original', self.original_bands),
            ('musical', self.musical_bands)
        ]:
            # Process each separator's output
            processed_outputs = {}
            for sep_name, stems in separator_outputs.items():
                processed = self.process_stems(stems, band_config)
                processed_outputs[sep_name] = processed
            
            # Use ensemble selection
            best_stems = select_best_stems(processed_outputs)
            
            # Calculate metrics
            results[config_name] = calculate_metrics(best_stems)
        
        return results
    
    def validate_batch(
        self,
        tracks: Dict[str, List[str]]  # Genre -> track paths
    ) -> Dict:
        """Run validation on multiple tracks"""
        results = defaultdict(lambda: defaultdict(list))
        
        for genre, genre_tracks in tracks.items():
            print(f"\nProcessing {genre} tracks...")
            
            for track in tqdm(genre_tracks):
                try:
                    track_results = self.validate_single_track(track)
                    
                    # Aggregate results
                    for config in ['original', 'musical']:
                        for metric in self.metrics:
                            results[genre][f"{config}_{metric}"].append(
                                track_results[config][metric]
                            )
                except Exception as e:
                    print(f"Error processing {track}: {str(e)}")
                    continue
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate simple comparison report"""
        report = ["Band-Split Validation Results", "=" * 30, ""]
        
        for genre, metrics in results.items():
            report.extend([
                f"\n{genre.upper()}:",
                "-" * 20
            ])
            
            # Compare configurations
            for metric in self.metrics:
                orig_scores = metrics[f"original_{metric}"]
                music_scores = metrics[f"musical_{metric}"]
                
                diff = np.mean(music_scores) - np.mean(orig_scores)
                
                report.extend([
                    f"\n{metric.upper()}:",
                    f"Original: {np.mean(orig_scores):.3f} ± {np.std(orig_scores):.3f}",
                    f"Musical:  {np.mean(music_scores):.3f} ± {np.std(music_scores):.3f}",
                    f"Difference: {diff:.3f} ({'better' if diff > 0 else 'worse'})"
                ])
        
        return "\n".join(report)

def main():
    # Define test tracks
    tracks = {
        'rock': ['battery.wav'],
        'classical': ['orchestra.wav'],
        'rap': ['hiphop.wav']
    }
    
    # Run validation
    validator = MinimalBandSplitValidator()
    results = validator.validate_batch(tracks)
    
    # Generate and save report
    report = validator.generate_report(results)
    
    with open('bandsplit_validation_report.txt', 'w') as f:
        f.write(report)
    
    # Also save raw results for further analysis
    with open('bandsplit_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()