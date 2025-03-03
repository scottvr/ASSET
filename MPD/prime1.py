import numpy as np
from pathlib import Path
import soundfile as sf
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional

# Import stemprover components
from stemprover.core.config import ProcessingConfig
from stemprover.core.audio import AudioSegment
from stemprover.analysis.spectral import SpectralAnalyzer

@dataclass
class WaveformParams:
    """Parameters for waveform generation"""
    frequency: float  # Hz
    amplitude: float = 1.0
    phase_offset: float = 0.0  # radians
    start_time: float = 0.0    # seconds
    duration: float = 5.0      # seconds

class WaveformGenerator:
    """Generate pristine test waveforms for phase analysis"""
    
    def __init__(self, config: ProcessingConfig):
        self.sample_rate = config.sample_rate
        self.config = config
    
    def generate_sine(self, params: WaveformParams) -> AudioSegment:
        """Generate pure sine wave"""
        t = np.arange(0, params.duration, 1/self.sample_rate)
        start_idx = int(params.start_time * self.sample_rate)
        
        # Create silent buffer
        samples = np.zeros(int(params.duration * self.sample_rate))
        
        # Generate active portion
        active_t = t[start_idx:]
        wave = params.amplitude * np.sin(
            2 * np.pi * params.frequency * active_t + params.phase_offset
        )
        
        # Insert into buffer
        samples[start_idx:start_idx + len(wave)] = wave
        
        # Return as mono AudioSegment
        return AudioSegment(
            audio=samples.reshape(1, -1),
            sample_rate=self.sample_rate,
            start_time=0.0,
            duration=params.duration
        )
    
    def generate_square(self, params: WaveformParams) -> AudioSegment:
        """Generate square wave with controlled harmonics"""
        t = np.arange(0, params.duration, 1/self.sample_rate)
        start_idx = int(params.start_time * self.sample_rate)
        
        # Create silent buffer
        samples = np.zeros(int(params.duration * self.sample_rate))
        
        # Generate square wave using odd harmonics
        wave = np.zeros_like(t[start_idx:])
        for h in range(1, 31, 2):  # First 15 odd harmonics
            wave += (1/h) * np.sin(
                2 * np.pi * h * params.frequency * t[start_idx:] + params.phase_offset
            )
        
        # Normalize and apply amplitude
        wave = params.amplitude * wave / np.max(np.abs(wave))
        
        # Insert into buffer
        samples[start_idx:start_idx + len(wave)] = wave
        
        # Return as mono AudioSegment
        return AudioSegment(
            audio=samples.reshape(1, -1),
            sample_rate=self.sample_rate,
            start_time=0.0,
            duration=params.duration
        )
    
    def generate_test_case_1(self) -> Tuple[AudioSegment, AudioSegment, AudioSegment]:
        """Generate first phase test case with clean and artifact versions using prime frequencies"""
        # Parameters for test case 1 using prime-number based frequencies
        sine_params = WaveformParams(
            frequency=431.0,  # Prime frequency near A4
            amplitude=0.7,
            phase_offset=0.0,
            start_time=0.0
        )
        
        square_params = WaveformParams(
            frequency=223.0,  # Prime frequency near A3
            amplitude=0.5,
            phase_offset=0.0,
            start_time=0.25  # 250ms delay
        )
        
        # Generate clean stems
        sine = self.generate_sine(sine_params)
        square = self.generate_square(square_params)
        
        # Create clean mix
        clean_mix = AudioSegment(
            audio=sine.audio + square.audio,
            sample_rate=self.sample_rate
        )
        
        # Create artifact version with phase shift
        square_params.phase_offset = np.pi / 4  # 45 degrees
        square_shifted = self.generate_square(square_params)
        artifact_mix = AudioSegment(
            audio=sine.audio + square_shifted.audio,
            sample_rate=self.sample_rate
        )
        
        return clean_mix, artifact_mix, sine  # Return reference sine for validation

def validate_test_case(
    clean: AudioSegment,
    artifact: AudioSegment,
    reference: AudioSegment,
    analyzer: Optional[SpectralAnalyzer] = None,
    output_dir: Optional[Path] = None
) -> dict:
    """Validate the generated test case"""
    from stemprover.common.math_utils import phase_coherence, phase_difference
    from stemprover.common.audio_utils import create_spectrogram
    import os
    
    if analyzer is None:
        analyzer = SpectralAnalyzer(Path("validation"), ProcessingConfig())
    
    # Calculate phase metrics
    clean_coherence = phase_coherence(
        phase_difference(
            analyzer.create_phase_spectrogram(clean.audio, clean.sample_rate),
            analyzer.create_phase_spectrogram(reference.audio, reference.sample_rate)
        )
    )
    
    artifact_coherence = phase_coherence(
        phase_difference(
            analyzer.create_phase_spectrogram(artifact.audio, artifact.sample_rate),
            analyzer.create_phase_spectrogram(reference.audio, reference.sample_rate)
        )
    )
    
    results = {
        "clean_phase_coherence": clean_coherence,
        "artifact_phase_coherence": artifact_coherence,
        "coherence_difference": clean_coherence - artifact_coherence
    }
    
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # We need to use soundfile directly since AudioSegment in stemprover might not have a save method
        sf.write(output_dir / "clean_mix.wav", clean.audio[0], clean.sample_rate)
        sf.write(output_dir / "artifact_mix.wav", artifact.audio[0], artifact.sample_rate)
        sf.write(output_dir / "reference_sine.wav", reference.audio[0], reference.sample_rate)
        
        # Plot spectrograms
        plt.figure(figsize=(15, 10))
        
        plt.subplot(311)
        plt.title("Clean Mix Spectrogram")
        D = create_spectrogram(clean.audio[0])
        plt.imshow(np.abs(D), aspect='auto', origin='lower')
        
        plt.subplot(312)
        plt.title("Artifact Mix Spectrogram")
        D = create_spectrogram(artifact.audio[0])
        plt.imshow(np.abs(D), aspect='auto', origin='lower')
        
        plt.subplot(313)
        plt.title("Reference Sine Spectrogram")
        D = create_spectrogram(reference.audio[0])
        plt.imshow(np.abs(D), aspect='auto', origin='lower')
        
        plt.tight_layout()
        plt.savefig(output_dir / "spectrograms.png")
        plt.close()
        
        # Save analysis results
        with open(output_dir / "analysis.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Set up config and paths
    config = ProcessingConfig(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512
    )
    
    output_dir = Path("mpd_test_case_1")
    
    # Create generator and produce test case
    generator = WaveformGenerator(config)
    clean_mix, artifact_mix, reference = generator.generate_test_case_1()
    
    # Analyze and save results
    analyzer = SpectralAnalyzer(Path("analysis"), config)
    results = validate_test_case(
        clean_mix,
        artifact_mix,
        reference,
        analyzer,
        output_dir=output_dir
    )
    
    # Print results
    print("\nTest Case 1 Analysis Results:")
    print(f"Clean Phase Coherence: {results['clean_phase_coherence']:.3f}")
    print(f"Artifact Phase Coherence: {results['artifact_phase_coherence']:.3f}")
    print(f"Coherence Difference: {results['coherence_difference']:.3f}")
    print(f"\nOutputs saved to: {output_dir}")
