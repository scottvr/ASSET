import numpy as np
import json
import argparse
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any, Type
from enum import Enum
import librosa
import copy

# Import stemprover components
try:
    from stemprover.core.config import ProcessingConfig
    from stemprover.core.audio import AudioSegment
    from stemprover.analysis.spectral import SpectralAnalyzer
    from stemprover.common.math_utils import phase_coherence, phase_difference
    from stemprover.common.audio_utils import create_spectrogram
    from stemprover.analysis.artifacts.base import (
        ArtifactType, ArtifactParameters, ArtifactGenerator
    )
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
        # Handle different sized inputs by padding the shorter one
        if not np.iscomplexobj(spec1) and not np.iscomplexobj(spec2):
            # Both are time-domain signals - pad to match length
            if len(spec1) > len(spec2):
                spec2 = np.pad(spec2, (0, len(spec1) - len(spec2)))
            elif len(spec2) > len(spec1):
                spec1 = np.pad(spec1, (0, len(spec2) - len(spec1)))
        
        phase1 = np.angle(spec1) if np.iscomplexobj(spec1) else np.angle(np.fft.rfft(spec1))
        phase2 = np.angle(spec2) if np.iscomplexobj(spec2) else np.angle(np.fft.rfft(spec2))
        
        # If we still have a size mismatch after FFT (rare but possible with complex inputs)
        if phase1.shape != phase2.shape:
            min_size = min(len(phase1), len(phase2))
            phase1 = phase1[:min_size]
            phase2 = phase2[:min_size]
            
        return np.abs(phase1 - phase2)
    
    class SpectralAnalyzer:
        def __init__(self, output_dir, config):
            self.output_dir = Path(output_dir)
            self.config = config
        
        def create_phase_spectrogram(self, audio, sample_rate):
            # Extract channel data and ensure 1D
            audio_data = audio[0] if audio.ndim > 1 else audio
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
            return np.fft.rfft(audio_data)
            
    # Minimal implementations for ArtifactType and ArtifactParameters
    class ArtifactType(Enum):
        PHASE_INCOHERENCE = 1
        HIGH_FREQUENCY_RINGING = 2
        TEMPORAL_SMEARING = 3
        SPECTRAL_HOLES = 4
        QUANTIZATION_NOISE = 5
        ALIASING = 6
        BACKGROUND_BLEED = 7
        TRANSIENT_SUPPRESSION = 8
        
        @classmethod
        def as_dict(cls) -> Dict[str, str]:
            return {
                'PHASE_INCOHERENCE': 'Phase inconsistencies between channels or frequency bands',
                'HIGH_FREQUENCY_RINGING': 'Artificial high-frequency content created by separation',
                'TEMPORAL_SMEARING': 'Time-domain blurring of transient sounds',
                'SPECTRAL_HOLES': 'Missing frequency content in specific bands',
                'QUANTIZATION_NOISE': 'Digital quantization artifacts from processing',
                'ALIASING': 'Frequency aliasing due to downsampling or poor reconstruction',
                'BACKGROUND_BLEED': 'Residual content from other stems bleeding through',
                'TRANSIENT_SUPPRESSION': 'Reduction or loss of sharp transients'
            }
    
    @dataclass
    class ArtifactParameters:
        artifact_type: ArtifactType
        intensity: float = 0.5
        frequency_range: Optional[Tuple[float, float]] = None
        time_range: Optional[Tuple[float, float]] = None
        additional_params: Dict[str, Any] = field(default_factory=dict)
    
    class ArtifactGenerator:
        def __init__(self, sample_rate: int = 44100):
            self.sample_rate = sample_rate
            
        def apply_artifact(self, audio: AudioSegment, params: ArtifactParameters) -> AudioSegment:
            # Minimal implementation that just returns the input audio
            return audio

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
class ArtifactConfig:
    """Configuration for artifact generation"""
    artifact_type: str  # Matches ArtifactType enum name
    target_stems: List[str]  # Which stems to apply this to
    intensity: float = 0.5
    frequency_range: Optional[List[float]] = None
    time_range: Optional[List[float]] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_parameters(self) -> ArtifactParameters:
        """Convert to ArtifactParameters object"""
        # Convert string to enum
        artifact_enum = getattr(ArtifactType, self.artifact_type)
        
        # Convert lists to tuples if present
        freq_range = tuple(self.frequency_range) if self.frequency_range else None
        time_range = tuple(self.time_range) if self.time_range else None
        
        return ArtifactParameters(
            artifact_type=artifact_enum,
            intensity=self.intensity,
            frequency_range=freq_range,
            time_range=time_range,
            additional_params=self.additional_params
        )

@dataclass
class TestCaseConfig:
    """Configuration for a complete test case"""
    name: str
    description: str
    waveforms: List[WaveformConfig]
    variations: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[ArtifactConfig] = field(default_factory=list)
    duration: float = 5.0
    sample_rate: int = 44100
    output_dir: str = "output"
    enable_phase_analysis: bool = True
    enable_spectral_analysis: bool = True
    auto_pad_to_max: bool = True  # Auto-pad waveforms to ensure consistent lengths

class EnhancedWaveformGenerator:
    """Generate test signals based on JSON configuration"""
    
    def __init__(self, config: ProcessingConfig):
        self.sample_rate = config.sample_rate
        self.config = config
        if STEMPROVER_AVAILABLE:
            self.artifact_generator = ArtifactGenerator(sample_rate=config.sample_rate)
    
    def generate_from_config(self, test_config: TestCaseConfig) -> Dict[str, AudioSegment]:
        """Generate all audio files for a test case"""
        results = {}
        
        # Auto-pad waveforms to ensure consistent lengths if requested
        if getattr(test_config, 'auto_pad_to_max', False):
            max_duration = max(wave_config.duration + wave_config.start_time 
                            for wave_config in test_config.waveforms)
            for wave_config in test_config.waveforms:
                wave_config.duration = max_duration - wave_config.start_time
        
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
        
        # Apply artifacts if stemprover is available
        if STEMPROVER_AVAILABLE and test_config.artifacts:
            print(f"Applying {len(test_config.artifacts)} artifacts...")
            
            # Copy clean stems for artifact generation
            artifact_stems = {name: copy.deepcopy(stem) for name, stem in stems.items()}
            
            # Apply artifacts to specified stems
            for artifact_config in test_config.artifacts:
                params = artifact_config.to_parameters()
                
                # For BACKGROUND_BLEED, we need to provide the bleed source
                if params.artifact_type == ArtifactType.BACKGROUND_BLEED:
                    if 'bleed_source' not in artifact_config.additional_params:
                        print(f"Warning: BACKGROUND_BLEED artifact missing 'bleed_source' parameter")
                        continue
                    
                    bleed_source = artifact_config.additional_params['bleed_source']
                    if bleed_source not in stems:
                        print(f"Warning: Bleed source '{bleed_source}' not found in stems")
                        continue
                    
                    # Add bleed audio to params
                    params.additional_params['bleed_audio'] = stems[bleed_source].audio
                
                # Apply to each target stem
                for stem_name in artifact_config.target_stems:
                    waveform_key = stem_name.replace("stem_", "")
                    
                    if waveform_key in artifact_stems:
                        print(f"Applying {params.artifact_type.name} to {stem_name}...")
                        artifact_stems[waveform_key] = self.artifact_generator.apply_artifact(
                            artifact_stems[waveform_key], params
                        )
                    else:
                        print(f"Warning: Target stem '{stem_name}' not found")
            
            # Create artifact mix
            artifact_audio = np.zeros((2, int(test_config.duration * self.sample_rate)))
            for stem in artifact_stems.values():
                stem_audio = stem.audio
                if stem_audio.shape[0] == 1 and artifact_audio.shape[0] == 2:
                    stem_audio = np.repeat(stem_audio, 2, axis=0)
                artifact_audio[:stem_audio.shape[0], :stem_audio.shape[1]] += stem_audio
            
            # Add artifact mix and individual artifact stems to results
            results["artifact_mix"] = AudioSegment(
                audio=artifact_audio,
                sample_rate=self.sample_rate
            )
            
            for name, stem in artifact_stems.items():
                results[f"artifact_stem_{name}"] = stem
        
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
        elif config.waveform_type == "noise":
            # Generate white noise
            wave = np.random.normal(0, 1, len(t[start_idx:]))
            wave = wave / np.max(np.abs(wave))  # Normalize
        elif config.waveform_type == "pink_noise":
            # Generate pink noise (1/f spectrum)
            white_noise = np.random.normal(0, 1, len(t[start_idx:]))
            # Convert to frequency domain
            fft = np.fft.rfft(white_noise)
            # Create 1/f filter
            f = np.fft.rfftfreq(len(white_noise))
            f[0] = 1  # Avoid division by zero
            fft = fft / np.sqrt(f)
            # Convert back to time domain
            wave = np.fft.irfft(fft)
            wave = wave / np.max(np.abs(wave))  # Normalize
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
    output_dir: Optional[Path] = None,
    skip_wav_export: bool = False,
    enable_phase_analysis: bool = True,
    enable_spectral_analysis: bool = True
) -> Dict[str, Any]:
    """Validate and analyze the generated test signals"""
    results = {}
    
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save all audio files unless we're in analyze-only mode
        if not skip_wav_export:
            for name, audio in generator_results.items():
                output_path = output_dir / f"{name}.wav"
                if not output_path.exists():
                    sf.write(str(output_path), 
                            audio.audio.T if audio.audio.ndim > 1 else audio.audio, 
                            audio.sample_rate)
                    print(f"Saved {output_path}")
                else:
                    print(f"File {output_path} already exists, skipping")
        
        # Create spectrograms if analyzer is available
        if analyzer and enable_spectral_analysis:
            # 1. Find the longest audio for consistent time axis
            max_length = max(audio.audio.shape[1] for audio in generator_results.values())
            
            plt.figure(figsize=(15, 10))
            subplot_idx = 1
            for name, audio in generator_results.items():
                plt.subplot(len(generator_results), 1, subplot_idx)
                plt.title(f"{name} Spectrogram")
                
                # Use first channel if stereo, or the only channel if mono
                audio_data = audio.audio[0] if audio.audio.ndim > 1 and audio.audio.shape[0] > 1 else audio.audio.flatten()
                
                # Pad if needed for consistent time axis
                if len(audio_data) < max_length:
                    audio_data = np.pad(audio_data, (0, max_length - len(audio_data)))
                
                # Create spectrogram with consistent parameters
                D = librosa.stft(audio_data, n_fft=2048, hop_length=512)
                
                # Use log scale with improved color normalization
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                plt.imshow(S_db, aspect='auto', origin='lower', cmap='viridis', vmin=-80, vmax=0)
                
                # Add colorbar and better axis labels
                plt.colorbar(format='%+2.0f dB')
                plt.ylabel('Frequency bin')
                
                subplot_idx += 1
            
            # Save spectrograms plot
            plt.tight_layout()
            plt.savefig(output_dir / "spectrograms.png")
            plt.close()
        
        # Phase coherence analysis
        if enable_phase_analysis:
            # Compare clean mix to original stems
            if "clean_mix" in generator_results:
                results["clean_phase_coherence"] = {}
                clean_mix = generator_results["clean_mix"]
                
                # Calculate phase coherence between clean mix and stems
                for name, audio in generator_results.items():
                    if name.startswith("stem_"):
                        if analyzer:
                            # Use analyzer for more accurate phase analysis
                            coh = phase_coherence(
                                phase_difference(
                                    analyzer.create_phase_spectrogram(clean_mix.audio, clean_mix.sample_rate),
                                    analyzer.create_phase_spectrogram(audio.audio, audio.sample_rate)
                                )
                            )
                        else:
                            # Fallback to simpler phase analysis with padding to handle different-length inputs
                            clean_audio = clean_mix.audio.flatten() if clean_mix.audio.ndim > 0 else clean_mix.audio
                            stem_audio = audio.audio.flatten() if audio.audio.ndim > 0 else audio.audio
                            
                            coh = phase_coherence(
                                phase_difference(clean_audio, stem_audio)
                            )
                        
                        results["clean_phase_coherence"][name] = coh
                
                # Average phase coherence
                if results["clean_phase_coherence"]:
                    results["clean_phase_coherence"] = np.mean(list(results["clean_phase_coherence"].values()))
            
            # Compare artifact mix to clean mix if available
            if "artifact_mix" in generator_results and "clean_mix" in generator_results:
                artifact_mix = generator_results["artifact_mix"]
                clean_mix = generator_results["clean_mix"]
                
                if analyzer:
                    artifact_coh = phase_coherence(
                        phase_difference(
                            analyzer.create_phase_spectrogram(artifact_mix.audio, artifact_mix.sample_rate),
                            analyzer.create_phase_spectrogram(clean_mix.audio, clean_mix.sample_rate)
                        )
                    )
                else:
                    # Consistent handling of potentially different-length inputs
                    artifact_audio = artifact_mix.audio.flatten() if artifact_mix.audio.ndim > 0 else artifact_mix.audio
                    clean_audio = clean_mix.audio.flatten() if clean_mix.audio.ndim > 0 else clean_mix.audio
                    
                    artifact_coh = phase_coherence(
                        phase_difference(artifact_audio, clean_audio)
                    )
                
                results["artifact_phase_coherence"] = artifact_coh
                results["coherence_difference"] = artifact_coh - results.get("clean_phase_coherence", 0)
        
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
    
    # Convert artifact configs to proper objects if present
    artifacts = []
    for artifact_data in config_data.get("artifacts", []):
        artifacts.append(ArtifactConfig(**artifact_data))
    
    # Create TestCaseConfig
    config_data["waveforms"] = waveforms
    config_data["artifacts"] = artifacts
    return TestCaseConfig(**config_data)

def save_test_config(config: TestCaseConfig, output_path: str):
    """Save test configuration to JSON file"""
    config_dict = asdict(config)
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def get_artifact_example_config() -> TestCaseConfig:
    """Create an example configuration with artifacts for stem separation testing"""
    return TestCaseConfig(
        name="stem_separation_artifacts",
        description="Test case with synthetic artifacts that mimic real-world separation issues",
        duration=5.0,
        sample_rate=44100,
        output_dir="mpd_artifact_test",
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
        # Enhanced with specific artifact types
        artifacts=[
            # Phase incoherence in the vocal
            ArtifactConfig(
                artifact_type="PHASE_INCOHERENCE",
                target_stems=["sine"],
                intensity=0.4,
                frequency_range=[800, 4000]
            ),
            # High frequency ringing in square wave
            ArtifactConfig(
                artifact_type="HIGH_FREQUENCY_RINGING",
                target_stems=["square"],
                intensity=0.6,
                frequency_range=[6000, 16000]
            ),
            # Spectral holes in sawtooth
            ArtifactConfig(
                artifact_type="SPECTRAL_HOLES",
                target_stems=["sawtooth"],
                intensity=0.5,
                frequency_range=[500, 3000]
            ),
            # Background bleed of square into sine
            ArtifactConfig(
                artifact_type="BACKGROUND_BLEED",
                target_stems=["sine"],
                intensity=0.3,
                additional_params={"bleed_source": "square"}
            )
        ],
        # Also include standard variations
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
        ],
        enable_phase_analysis=True,
        enable_spectral_analysis=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test signals with controllable artifacts")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--create-example", action="store_true", help="Create an example configuration file")
    parser.add_argument("--example-path", default="artifact_example_config.json", help="Path for example configuration file")
    parser.add_argument("--output-dir", help="Output directory for generated files")
    parser.add_argument("--analyze-only", action="store_true", help="Skip generation and only analyze existing WAV files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip generation if output files already exist")
    parser.add_argument("--no-phase-analysis", action="store_true", help="Skip phase analysis")
    parser.add_argument("--no-spectral-analysis", action="store_true", help="Skip spectral analysis")
    args = parser.parse_args()
    
    if args.create_example:
        config = get_artifact_example_config()
        save_test_config(config, args.example_path)
        print(f"Example configuration saved to {args.example_path}")
        exit(0)
    
    # Get configuration
    if args.config:
        config = load_test_config(args.config)
    else:
        config = get_artifact_example_config()
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Override analysis options if specified
    if args.no_phase_analysis:
        config.enable_phase_analysis = False
    if args.no_spectral_analysis:
        config.enable_spectral_analysis = False
    
    # Set up processing config
    proc_config = ProcessingConfig(
        sample_rate=config.sample_rate,
        n_fft=2048,
        hop_length=512
    )
    
    output_dir = Path(config.output_dir)
    output = {}
    
    # Check if we should analyze existing files or generate new ones
    if args.analyze_only:
        print("Analyze-only mode: Skipping generation and analyzing existing files...")
        # Load existing WAV files
        output_dir.mkdir(exist_ok=True, parents=True)
        for wav_file in output_dir.glob("*.wav"):
            name = wav_file.stem
            audio_data, sr = sf.read(str(wav_file))
            # Convert to the format expected by validate_test_case
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            elif audio_data.ndim == 2 and audio_data.shape[1] < audio_data.shape[0]:
                # Transpose if needed (samples, channels) -> (channels, samples)
                audio_data = audio_data.T
            output[name] = AudioSegment(audio=audio_data, sample_rate=sr)
        
        if not output:
            print(f"No WAV files found in {output_dir}. Nothing to analyze.")
            exit(1)
    else:
        # Check if files already exist and we should skip generation
        should_skip = False
        if args.skip_existing:
            expected_files = ["clean_mix.wav"]
            expected_files.extend([f"stem_{w.waveform_type}.wav" for w in config.waveforms])
            
            if config.artifacts:
                expected_files.append("artifact_mix.wav")
                expected_files.extend([f"artifact_stem_{w.waveform_type}.wav" for w in config.waveforms])
            
            expected_files.extend([f"variation_{i}.wav" for i in range(len(config.variations))])
            
            existing_files = [f.name for f in output_dir.glob("*.wav")]
            missing_files = [f for f in expected_files if f not in existing_files]
            
            if not missing_files:
                print(f"All expected files already exist in {output_dir}. Skipping generation.")
                should_skip = True
                # Load existing files
                for wav_file in output_dir.glob("*.wav"):
                    name = wav_file.stem
                    audio_data, sr = sf.read(str(wav_file))
                    if audio_data.ndim == 1:
                        audio_data = audio_data.reshape(1, -1)
                    elif audio_data.ndim == 2 and audio_data.shape[1] < audio_data.shape[0]:
                        audio_data = audio_data.T
                    output[name] = AudioSegment(audio=audio_data, sample_rate=sr)
            else:
                print(f"Missing files: {missing_files}. Will generate all files.")
        
        if not should_skip:
            # Create generator and produce test case
            print("Generating waveforms...")
            generator = EnhancedWaveformGenerator(proc_config)
            output = generator.generate_from_config(config)
    
    # Analyze and save results
    print("Analyzing audio files...")
    analyzer = SpectralAnalyzer(Path("analysis"), proc_config) if STEMPROVER_AVAILABLE else None
    results = validate_test_case(
        output,
        analyzer,
        output_dir=output_dir,
        skip_wav_export=args.analyze_only,  # Skip saving WAVs if in analyze-only mode
        enable_phase_analysis=config.enable_phase_analysis,
        enable_spectral_analysis=config.enable_spectral_analysis
    )
    
    # Print summary
    print(f"\nGenerated {len(output)} audio files:")
    for name in output.keys():
        print(f"- {name}.wav")
    
    print(f"\nResults saved to: {config.output_dir}")
    
    # Print analysis results summary
    if results:
        print("\nAnalysis Summary:")
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue:.4f}")
            else:
                print(f"  {key}: {value:.4f}")