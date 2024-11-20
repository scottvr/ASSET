from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, Tuple

@dataclass
class SeparationProfile:
    backend: SeparatorBackend
    preserve_high_freq: bool = False
    target_sample_rate: int = 44100
    min_segment_length: float = 5.0
    # Added enhancement parameters
    use_phase_aware_controlnet: bool = False
    use_high_freq_processor: bool = True
    artifact_reduction_config: Optional[ProcessingConfig] = None
    controlnet_model_path: Optional[Path] = None

@dataclass
class SeparationResult:
    clean_vocal: AudioSegment
    separated_vocal: AudioSegment
    enhanced_vocal: Optional[AudioSegment]  # After artifact reduction
    accompaniment: AudioSegment
    mixed: AudioSegment
    analysis_path: Optional[Path] = None
    phase_analysis: Optional[Dict[str, Any]] = None
    artifact_analysis: Optional[Dict[str, str]] = None

class StemProcessor:
    def __init__(self, profile: SeparationProfile):
        self.profile = profile
        self.separator = self._create_separator()
        self.analyzer = VocalSeparationAnalyzer(Path("analysis_output"))
        
        # Initialize enhancement components
        self.artifact_processor = IntegratedHighFrequencyProcessor(
            "enhancement_output",
            config=profile.artifact_reduction_config or ProcessingConfig()
        ) if profile.use_high_freq_processor else None
        
        self.controlnet = PhaseAwareControlNet.from_pretrained(
            profile.controlnet_model_path
        ) if profile.use_phase_aware_controlnet else None

    def process_stems(self, 
                     vocal_paths: Tuple[str, str],
                     accompaniment_paths: Tuple[str, str],
                     start_time: float = 0.0,
                     duration: float = 30.0) -> SeparationResult:
        try:
            self.separator.setup()
            
            # Load and prepare audio
            vocals = self._load_stereo_pair(*vocal_paths, start_time, duration)
            accompaniment = self._load_stereo_pair(*accompaniment_paths, start_time, duration)
            mixed = AudioSegment(
                audio=vocals.audio + accompaniment.audio,
                sample_rate=vocals.sample_rate,
                start_time=start_time,
                duration=duration
            )

            # Initial separation
            separated = self.separator.separate(mixed)
            base_analysis = self.analyzer.analyze(vocals, separated)
            
            # Enhancement pipeline
            enhanced = separated
            artifact_analysis = None
            
            # Apply high-frequency artifact reduction if enabled
            if self.artifact_processor:
                enhanced_audio, analysis_files = self.artifact_processor.process_and_analyze(
                    enhanced.audio
                )
                enhanced = AudioSegment(
                    audio=enhanced_audio,
                    sample_rate=enhanced.sample_rate,
                    start_time=enhanced.start_time,
                    duration=enhanced.duration
                )
                artifact_analysis = analysis_files
            
            # Apply ControlNet enhancement if enabled
            if self.controlnet:
                enhanced = self._apply_controlnet_enhancement(enhanced)
            
            return SeparationResult(
                clean_vocal=vocals,
                separated_vocal=separated,
                enhanced_vocal=enhanced,
                accompaniment=accompaniment,
                mixed=mixed,
                analysis_path=base_analysis,
                artifact_analysis=artifact_analysis
            )
        finally:
            self.separator.cleanup()

    def _apply_controlnet_enhancement(self, audio: AudioSegment) -> AudioSegment:
        # Convert to spectrogram
        spec = self.artifact_processor.create_complex_spectrogram(
            audio.audio
        ) if self.artifact_processor else None
        
        # Process through ControlNet
        enhanced_spec = self.controlnet(spec)
        
        # Convert back to audio
        enhanced_audio = self.artifact_processor.spectrogram_to_audio(
            enhanced_spec
        ) if self.artifact_processor else None
        
        return AudioSegment(
            audio=enhanced_audio,
            sample_rate=audio.sample_rate,
            start_time=audio.start_time,
            duration=audio.duration
        )