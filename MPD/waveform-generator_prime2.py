import numpy as np
import json
import argparse
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any

# Import stemprover components if available, otherwise define minimal versions
try:
    from stemprover.core.config import ProcessingConfig
    from stemprover.core.audio import AudioSegment
    from stemprover.analysis.spectral import SpectralAnalyzer
    from stemprover.common.math_utils import phase_coherence, phase_difference
    from stemprover.common.audio_utils import create_spectrogram
    STEMPROVER_AVAILABLE = True
except ImportError:
    STEMPROVER_AVAILABLE = False
    print("Warning: stemprover not available, using minimal implementations")
    
    @dataclass
    class ProcessingConfig:
        sample_rate: int = 44100
        n_fft: int = 2048
        hop_length: int = 512
    
    class AudioSegment:
        def __init__(self, audio: np.ndarray, sample_rate: int):
            self.audio = audio
            self.sample_rate = sample_rate
        
        def save(self, path: str):
            sf.write(path, self.audio.T if self.audio.ndim > 1 else self.audio, self.sample_rate)
    
    def create_spectrogram(audio: np.ndarray, **kwargs):
        return np.abs(np.fft.rfft(audio))
    
    def phase_coherence(phase_diff: np.ndarray) -> float:
        return float(np.mean(np.cos(phase_diff)))
    
    def phase_difference(spec1: np.ndarray, spec2: np.ndarray) -> np.ndarray:
        phase1 = np.angle(spec1) if np.iscomplexobj(spec1) else np.angle(np.fft.rfft(spec1))
        phase2 = np.angle(spec2) if np.iscomplexobj(spec2) else np.angle(np.fft.rfft(spec2))
        return np.abs(phase1 - phase2)
    
    class SpectralAnalyzer:
        def __init__(self, output_dir, config):
            self.output_dir = Path(output_dir)
            self.config = config
        
        def create_phase_spectrogram(self, audio, sample_rate):
            return np.fft.rfft(audio[0] if audio.ndim > 1 else audio)

@dataclass
class WaveformConfig:
    """Base configuration for all waveforms"""
    waveform_type: str
    frequency: float
    amplitude: float = 1.0
    phase_offset: float = 0.0  # radians
    start_time: float = 0.0    # seconds
    duration: float = 5.0      # seconds
    channel: str = "mono"      # mono, left, right, center, or a float pan value between -1 and 1
    
    # Optional modulation
    vibrato_rate: float = 0.0  # Hz, 0 means no vibrato
    vibrato_depth: float = 0.0 # Hz peak deviation
    tremolo_rate: float = 0.0  # Hz, 0 means no tremolo
    tremolo_depth: float = 0.0 # Amplitude deviation (0-1)
    
    # Envelope
    envelope: Dict[str, float] = field(default_factory=lambda: {"attack": 0.01, "decay": 0.1, "sustain": 0.7, "release": 0.2})
    
    # Effects
    reverb: Dict[str, float] = field(default_factory=lambda: {"amount": 0.0, "decay": 1.0, "damping": 0.5})

@dataclass
class TestCaseConfig:
    """Configuration for a complete test case"""
    name: str
    description: str
    waveforms: List[WaveformConfig]
    variations: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 5.0
    sample_rate: int = 44100
    output_dir: str = "output"

class EnhancedWaveformGenerator:
    """Generate test signals based on JSON configuration"""
    
    def __init__(self, config: ProcessingConfig):
        self.sample_rate = config.sample_rate
        self.config = config
    
    def generate_from_config(self, test_config: TestCaseConfig) -> Dict[str, AudioSegment]:
        """Generate all audio files for a test case"""
        results = {}
        
        # Generate base stems
        stems = {}
        for wave_config in test_config.waveforms:
            stem = self._generate_waveform(wave_config)
            stems[wave_config.waveform_type] = stem
        
        # Create clean mix
        clean_audio = np.zeros((2, int(test_config.duration * self.sample_rate)))
        for stem in stems.values():
            # Account for mono/stereo differences
            stem_audio = stem.audio
            if stem_audio.shape[0] == 1 and clean_audio.shape[0] == 2:
                stem_audio = np.repeat(stem_audio, 2, axis=0)
            clean_audio[:stem_audio.shape[0], :stem_audio.shape[1]] += stem_audio
        
        results["clean_mix"] = AudioSegment(
            audio=clean_audio,
            sample_rate=self.sample_rate
        )
        
        # Add stems to results
        for name, stem in stems.items():
            results[f"stem_{name}"] = stem
        
        # Generate variations
        for i, variation in enumerate(test_config.variations):
            modified_stems = dict(stems)
            
            # Apply modifications to specified stems
            for stem_name, modifications in variation.get("modifications", {}).items():
                if stem_name in stems:
                    for wave_config in test_config.waveforms:
                        if wave_config.waveform_type == stem_name:
                            # Create a modified config with updated parameters
                            modified_config = WaveformConfig(**asdict(wave_config))
                            for attr, value in modifications.items():
                                if hasattr(modified_config, attr):
                                    setattr(modified_config, attr, value)
                            
                            # Generate modified stem
                            modified_stems[stem_name] = self._generate_waveform(modified_config)
            
            # Create variant mix
            variant_audio = np.zeros((2, int(test_config.duration * self.sample_rate)))
            for stem in modified_stems.values():
                stem_audio = stem.audio
                if stem_audio.shape[0] == 1 and variant_audio.shape[0] == 2:
                    stem_audio = np.repeat(stem_audio, 2, axis=0)
                variant_audio[:stem_audio.shape[0], :stem_audio.shape[1]] += stem_audio
            
            results[f"variation_{i}"] = AudioSegment(
                audio=variant_audio,
                sample_rate=self.sample_rate
            )
        
        return results
    
    def _generate_waveform(self, config: WaveformConfig) -> AudioSegment:
        """Generate a waveform based on configuration"""
        duration = config.duration
        t = np.arange(0, duration, 1/self.sample_rate)
        start_idx = int(config.start_time * self.sample_rate)
        num_samples = int(duration * self.sample_rate)
        
        # Create silent buffer
        is_stereo = config.channel not in ["mono", "left", "right", "center"]
        if is_stereo or config.channel in ["left", "right", "center"]:
            samples = np.zeros((2, num_samples))
        else:
            samples = np.zeros((1, num_samples))
        
        # Generate base waveform
        base_frequency = config.frequency
        if config.vibrato_rate > 0:
            # Apply frequency modulation (vibrato)
            frequency_mod = config.vibrato_depth * np.sin(2 * np.pi * config.vibrato_rate * t[start_idx:])
            frequency = base_frequency + frequency_mod
            phase = np.cumsum(2 * np.pi * frequency / self.sample_rate)
        else:
            phase = 2 * np.pi * base_frequency * t[start_idx:] + config.phase_offset
        
        # Generate the appropriate waveform
        if config.waveform_type == "sine":
            wave = np.sin(phase)
        elif config.waveform_type == "square":
            # Generate square wave using odd harmonics
            wave = np.zeros_like(t[start_idx:])
            for h in range(1, 31, 2):  # First 15 odd harmonics
                wave += (1/h) * np.sin(h * phase)
            wave = wave / np.max(np.abs(wave))  # Normalize
        elif config.waveform_type == "sawtooth":
            # Generate sawtooth wave using harmonics
            wave = np.zeros_like(t[start_idx:])
            for h in range(1, 31):  # First 30 harmonics
                wave += (1/h) * np.sin(h * phase)
            wave = wave / np.max(np.abs(wave))  # Normalize
        elif config.waveform_type == "triangle":
            # Generate triangle wave
            wave = 2 * np.abs(2 * (t[start_idx:] * base_frequency - np.floor(t[start_idx:] * base_frequency + 0.5))) - 1
        else:
            # Default to sine
            wave = np.sin(phase)
        
        # Apply tremolo if specified
        if config.tremolo_rate > 0:
            tremolo = 1.0 - config.tremolo_depth * (0.5 + 0.5 * np.sin(2 * np.pi * config.tremolo_rate * t[start_idx:]))
            wave *= tremolo
        
        # Apply envelope
        env = np.ones_like(wave)
        attack_samples = int(config.envelope["attack"] * self.sample_rate)
        decay_samples = int(config.envelope["decay"] * self.sample_rate)
        release_samples = int(config.envelope["release"] * self.sample_rate)
        
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
        if decay_samples > 0:
            sustain_level = config.envelope["sustain"]
            decay_end = attack_samples + decay_samples
            env[attack_samples:decay_end] = np.linspace(1, sustain_level, decay_samples)
        
        sustain_end = len(wave) - release_samples
        if release_samples > 0 and sustain_end > 0:
            env[sustain_end:] = np.linspace(config.envelope["sustain"], 0, release_samples)
        
        wave *= env
        
        # Apply amplitude
        wave *= config.amplitude
        
        # Pan or position in stereo field
        if config.channel == "mono":
            stereo_wave = wave.reshape(1, -1)
        elif config.channel == "left":
            stereo_wave = np.zeros((2, len(wave)))
            stereo_wave[0, :] = wave
        elif config.channel == "right":
            stereo_wave = np.zeros((2, len(wave)))
            stereo_wave[1, :] = wave
        elif config.channel == "center":
            stereo_wave = np.zeros((2, len(wave)))
            stereo_wave[0, :] = wave * 0.7071  # -3dB pan law
            stereo_wave[1, :] = wave * 0.7071
        else:
            # Convert pan value (-1 to 1) to left/right gains
            try:
                pan = float(config.channel)
                pan = np.clip(pan, -1, 1)
                
                # Constant power panning
                left_gain = np.cos((pan + 1) * np.pi / 4)
                right_gain = np.sin((pan + 1) * np.pi / 4)
                
                stereo_wave = np.zeros((2, len(wave)))
                stereo_wave[0, :] = wave * left_gain
                stereo_wave[1, :] = wave * right_gain
            except ValueError:
                # Default to mono if channel value is invalid
                stereo_wave = wave.reshape(1, -1)
        
        # Apply reverb (very simple implementation - just a decaying echo)
        if config.reverb["amount"] > 0:
            reverb_len = int(config.reverb["decay"] * self.sample_rate)
            if reverb_len > 0:
                reverb_wave = np.zeros_like(stereo_wave)
                for i in range(10):  # 10 echos
                    delay = int((i + 1) * self.sample_rate * 0.05)  # 50ms between echos
                    if delay < stereo_wave.shape[1]:
                        echo_gain = config.reverb["amount"] * (1 - config.reverb["damping"]) ** i
                        end_idx = stereo_wave.shape[1] - delay
                        reverb_wave[:, delay:] += stereo_wave[:, :end_idx] * echo_gain
                
                stereo_wave = stereo_wave + reverb_wave
        
        # Insert into full buffer
        active_len = stereo_wave.shape[1]
        if start_idx + active_len <= samples.shape[1]:
            samples[:stereo_wave.shape[0], start_idx:start_idx + active_len] = stereo_wave
        
        # Return as AudioSegment
        return AudioSegment(
            audio=samples,
            sample_rate=self.sample_rate
        )

def validate_test_case(
    generator_results: Dict[str, AudioSegment],
    analyzer: Optional[SpectralAnalyzer] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Validate and analyze the generated test signals"""
    results = {}
    
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save all audio files
        for name, audio in generator_results.items():
            sf.write(output_dir / f"{name}.wav", 
                    audio.audio.T if audio.audio.ndim > 1 else audio.audio, 
                    audio.sample_rate)
        
        # Create spectrograms
        plt.figure(figsize=(15, 10))
        subplot_idx = 1
        for name, audio in generator_results.items():
            plt.subplot(len(generator_results), 1, subplot_idx)
            plt.title(f"{name} Spectrogram")
            audio_data = audio.audio[0] if audio.audio.ndim > 1 and audio.audio.shape[0] == 1 else audio.audio.mean(axis=0)
            D = create_spectrogram(audio_data)
            plt.imshow(np.log1p(np.abs(D)), aspect='auto', origin='lower')
            subplot_idx += 1
        
        plt.tight_layout()
        plt.savefig(output_dir / "spectrograms.png")
        plt.close()
        
        # Phase coherence analysis if stems are available
        if "clean_mix" in generator_results and any(k.startswith("stem_") for k in generator_results.keys()):
            results["phase_coherence"] = {}
            reference_stem = next((v for k, v in generator_results.items() if k.startswith("stem_")), None)
            
            if reference_stem and analyzer:
                for name, audio in generator_results.items():
                    if name != f"stem_{reference_stem}":
                        coh = phase_coherence(
                            phase_difference(
                                analyzer.create_phase_spectrogram(audio.audio, audio.sample_rate),
                                analyzer.create_phase_spectrogram(reference_stem.audio, reference_stem.sample_rate)
                            )
                        )
                        results["phase_coherence"][name] = coh
        
        # Save analysis results
        with open(output_dir / "analysis.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def load_test_config(config_path: str) -> TestCaseConfig:
    """Load test configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Convert waveform configs to proper objects
    waveforms = []
    for wave_data in config_data.get("waveforms", []):
        waveforms.append(WaveformConfig(**wave_data))
    
    # Create TestCaseConfig
    config_data["waveforms"] = waveforms
    return TestCaseConfig(**config_data)

def save_test_config(config: TestCaseConfig, output_path: str):
    """Save test configuration to JSON file"""
    config_dict = asdict(config)
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def get_example_config() -> TestCaseConfig:
    """Create an example configuration for Spleeter-friendly test signals"""
    return TestCaseConfig(
        name="vocal_instrumental_separation",
        description="Test case designed to create realistic vocal/instrumental separation scenario",
        duration=5.0,
        sample_rate=44100,
        output_dir="mpd_spleeter_test",
        waveforms=[
            # "Vocal" component - center-panned sine with vibrato
            WaveformConfig(
                waveform_type="sine",
                frequency=431.0,  # Prime frequency near A4
                amplitude=0.7,
                phase_offset=0.0,
                start_time=0.1,
                duration=4.8,
                channel="center",
                vibrato_rate=5.5,  # Hz, typical vocal vibrato
                vibrato_depth=4.0,  # Hz deviation
                tremolo_rate=7.2,   # Subtle amplitude variation
                tremolo_depth=0.15,
                envelope={"attack": 0.05, "decay": 0.1, "sustain": 0.8, "release": 0.5},
                reverb={"amount": 0.2, "decay": 1.2, "damping": 0.6}
            ),
            # "Instrumental" component 1 - left-panned square wave
            WaveformConfig(
                waveform_type="square",
                frequency=223.0,  # Prime frequency near A3
                amplitude=0.5,
                phase_offset=0.0,
                start_time=0.0,
                duration=5.0,
                channel="-0.7",   # Panned left
                envelope={"attack": 0.01, "decay": 0.05, "sustain": 0.9, "release": 0.3},
                reverb={"amount": 0.3, "decay": 1.5, "damping": 0.4}
            ),
            # "Instrumental" component 2 - right-panned sawtooth
            WaveformConfig(
                waveform_type="sawtooth",
                frequency=293.0,  # Near D4
                amplitude=0.4,
                phase_offset=0.0,
                start_time=0.25,  # Slight delay
                duration=4.5,
                channel="0.7",    # Panned right
                envelope={"attack": 0.02, "decay": 0.07, "sustain": 0.85, "release": 0.4},
                reverb={"amount": 0.3, "decay": 1.5, "damping": 0.4}
            )
        ],
        variations=[
            {
                "modifications": {
                    "sine": {
                        "phase_offset": np.pi / 4  # 45 degree phase shift to vocal
                    }
                }
            },
            {
                "modifications": {
                    "square": {
                        "phase_offset": np.pi / 3,  # Phase shift to left instrumental
                        "amplitude": 0.6  # Slightly louder
                    },
                    "sawtooth": {
                        "amplitude": 0.3  # Slightly quieter
                    }
                }
            }
        ]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test signals from configuration")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--create-example", action="store_true", help="Create an example configuration file")
    parser.add_argument("--example-path", default="example_config.json", help="Path for example configuration file")
    parser.add_argument("--output-dir", help="Output directory for generated files")
    args = parser.parse_args()
    
    if args.create_example:
        config = get_example_config()
        save_test_config(config, args.example_path)
        print(f"Example configuration saved to {args.example_path}")
        exit(0)
    
    # Get configuration
    if args.config:
        config = load_test_config(args.config)
    else:
        config = get_example_config()
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Set up processing config
    proc_config = ProcessingConfig(
        sample_rate=config.sample_rate,
        n_fft=2048,
        hop_length=512
    )
    
    # Create generator and produce test case
    generator = EnhancedWaveformGenerator(proc_config)
    output = generator.generate_from_config(config)
    
    # Analyze and save results
    analyzer = SpectralAnalyzer(Path("analysis"), proc_config) if STEMPROVER_AVAILABLE else None
    results = validate_test_case(
        output,
        analyzer,
        output_dir=Path(config.output_dir)
    )
    
    # Print summary
    print(f"\nGenerated {len(output)} audio files:")
    for name in output.keys():
        print(f"- {name}.wav")
    
    print(f"\nResults saved to: {config.output_dir}")
