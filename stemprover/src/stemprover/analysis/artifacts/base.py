from ...common.types import AudioArray, SpectrogramArray, TensorType
from ...common.audio_utils import create_spectrogram
from ...core.audio import AudioSegment

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import scipy.signal
import time
import psutil
import librosa
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum, auto

class ArtifactType(Enum):
    """Enumeration of common artifact types in audio separation"""
    PHASE_INCOHERENCE = auto()
    HIGH_FREQUENCY_RINGING = auto()
    TEMPORAL_SMEARING = auto()
    SPECTRAL_HOLES = auto()
    QUANTIZATION_NOISE = auto()
    ALIASING = auto()
    BACKGROUND_BLEED = auto()
    TRANSIENT_SUPPRESSION = auto()
    
    @classmethod
    def as_dict(cls) -> Dict[str, str]:
        """Get all artifact types as a dictionary with descriptions"""
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
    """Parameters for generating specific artifacts"""
    artifact_type: ArtifactType
    intensity: float = 0.5  # 0.0 to 1.0 scale of artifact intensity
    frequency_range: Optional[Tuple[float, float]] = None  # Hz range affected
    time_range: Optional[Tuple[float, float]] = None  # seconds range affected
    additional_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationMetrics:
    """Stores comparison metrics between original and processed audio"""
    phase_coherence: float
    frequency_response: Dict[str, float]
    signal_to_noise: float
    processing_time: float
    memory_usage: float
    
    def as_dict(self) -> Dict[str, float]:
        return {
            "phase_coherence": self.phase_coherence,
            "freq_response_mean": np.mean(list(self.frequency_response.values())),
            "freq_response_std": np.std(list(self.frequency_response.values())),
            "snr": self.signal_to_noise,
            "processing_time": self.processing_time,
            "memory_usage": self.memory_usage
        }

class ArtifactProcessor(ABC):
    """Base class for different artifact processing approaches"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.frequency_bands = {
            "sub_bass": (20, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "high_mid": (2000, 4000),
            "presence": (4000, 6000),
            "brilliance": (6000, 20000)
        }
    
    @abstractmethod
    def process(self, 
                audio_segment: AudioSegment,
                artifact_type: str) -> AudioSegment:
        """Process audio to remove specified artifact type"""
        pass
    
    def validate(self, 
                clean: AudioSegment,
                processed: AudioSegment,
                original_artifacts: AudioSegment) -> ValidationMetrics:
        """Calculate validation metrics comparing processed output to clean reference"""
        
        # Calculate phase coherence
        phase_coherence = self._measure_phase_coherence(
            clean.audio, processed.audio
        )
        
        # Analyze frequency response
        freq_response = self._analyze_frequency_response(
            clean.audio, processed.audio
        )
        
        # Calculate SNR using original artifacts as noise reference
        snr = self._calculate_snr(
            clean.audio,
            processed.audio,
            original_artifacts.audio
        )
        
        return ValidationMetrics(
            phase_coherence=phase_coherence,
            frequency_response=freq_response,
            signal_to_noise=snr,
            processing_time=0.0,  # Set by wrapper
            memory_usage=0.0  # Set by wrapper
        )
    
    def _measure_phase_coherence(self,
                               clean: np.ndarray,
                               processed: np.ndarray) -> float:
        """Measure phase coherence between clean and processed audio"""
        # Use STFT for phase analysis
        n_fft = 2048
        hop_length = 512
        
        clean_stft = librosa.stft(clean.ravel(), n_fft=n_fft, hop_length=hop_length)
        proc_stft = librosa.stft(processed.ravel(), n_fft=n_fft, hop_length=hop_length)
        
        # Calculate phase difference
        clean_phase = np.angle(clean_stft)
        proc_phase = np.angle(proc_stft)
        phase_diff = np.abs(clean_phase - proc_phase)
        
        # Return mean phase coherence (1 = perfect, 0 = random)
        return float(np.mean(np.cos(phase_diff)))
    
    def _analyze_frequency_response(self,
                                  clean: np.ndarray,
                                  processed: np.ndarray) -> Dict[str, float]:
        """Analyze frequency response preservation in different bands"""
        results = {}
        
        for band_name, (low, high) in self.frequency_bands.items():
            # Filter both signals to band
            clean_band = self._bandpass_filter(clean.ravel(), low, high)
            proc_band = self._bandpass_filter(processed.ravel(), low, high)
            
            # Calculate RMS difference
            clean_rms = np.sqrt(np.mean(clean_band ** 2))
            proc_rms = np.sqrt(np.mean(proc_band ** 2))
            
            # Store ratio (1 = perfect preservation)
            results[band_name] = proc_rms / clean_rms if clean_rms > 0 else 0.0
            
        return results
    
    def _bandpass_filter(self,
                        audio: np.ndarray,
                        low_freq: float,
                        high_freq: float) -> np.ndarray:
        """Apply bandpass filter to audio"""
        nyquist = self.sample_rate // 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design filter
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        return scipy.signal.filtfilt(b, a, audio)
    
    def _calculate_snr(self,
                      clean: np.ndarray,
                      processed: np.ndarray,
                      noise: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        signal_power = np.mean(clean.ravel() ** 2)
        noise_power = np.mean((processed.ravel() - clean.ravel()) ** 2)
        
        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

class ArtifactGenerator:
    """Generates synthetic artifacts that mimic real-world separation issues"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def apply_artifact(self, audio: AudioSegment, params: ArtifactParameters) -> AudioSegment:
        """Apply specified artifact to clean audio"""
        artifact_type = params.artifact_type
        
        # Create a copy to avoid modifying the original
        audio_data = audio.audio.copy()
        
        if artifact_type == ArtifactType.PHASE_INCOHERENCE:
            audio_data = self._apply_phase_incoherence(audio_data, params)
        elif artifact_type == ArtifactType.HIGH_FREQUENCY_RINGING:
            audio_data = self._apply_high_frequency_ringing(audio_data, params)
        elif artifact_type == ArtifactType.TEMPORAL_SMEARING:
            audio_data = self._apply_temporal_smearing(audio_data, params)
        elif artifact_type == ArtifactType.SPECTRAL_HOLES:
            audio_data = self._apply_spectral_holes(audio_data, params)
        elif artifact_type == ArtifactType.QUANTIZATION_NOISE:
            audio_data = self._apply_quantization_noise(audio_data, params)
        elif artifact_type == ArtifactType.ALIASING:
            audio_data = self._apply_aliasing(audio_data, params)
        elif artifact_type == ArtifactType.BACKGROUND_BLEED:
            if 'bleed_audio' not in params.additional_params:
                raise ValueError("Background bleed artifact requires 'bleed_audio' parameter")
            audio_data = self._apply_background_bleed(audio_data, params)
        elif artifact_type == ArtifactType.TRANSIENT_SUPPRESSION:
            audio_data = self._apply_transient_suppression(audio_data, params)
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        return AudioSegment(audio=audio_data, sample_rate=audio.sample_rate)
    
    def _apply_phase_incoherence(self, audio: np.ndarray, params: ArtifactParameters) -> np.ndarray:
        """Apply phase incoherence artifact"""
        intensity = params.intensity
        freq_range = params.frequency_range or (1000, 8000)  # Default to mid-high range
        
        # Convert to frequency domain
        n_fft = 2048
        hop_length = 512
        
        # Process each channel separately but maintain their phase relationship
        result = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            # STFT
            stft = librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop_length)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
            
            # Create mask for frequency range
            mask = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
            
            # Apply phase shift to masked frequencies
            phase = np.angle(stft)
            magnitude = np.abs(stft)
            
            # Random phase shift based on intensity
            max_shift = np.pi * intensity
            phase_shift = np.random.uniform(-max_shift, max_shift, size=phase[mask, :].shape)
            
            # Apply phase shift to target frequencies
            new_phase = phase.copy()
            new_phase[mask, :] += phase_shift
            
            # Reconstruct with new phase
            stft_modified = magnitude * np.exp(1j * new_phase)
            
            # Inverse STFT
            result[ch] = librosa.istft(stft_modified, hop_length=hop_length, length=audio.shape[1])
        
        return result
    
    def _apply_high_frequency_ringing(self, audio: np.ndarray, params: ArtifactParameters) -> np.ndarray:
        """Apply high frequency ringing artifacts"""
        intensity = params.intensity
        freq_range = params.frequency_range or (6000, 16000)  # Default to high range
        
        # Convert to frequency domain
        n_fft = 2048
        hop_length = 512
        result = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            # STFT
            stft = librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop_length)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
            
            # Create mask for frequency range
            mask = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
            
            # Enhance high frequencies
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Add synthetic "ringing" by boosting and adding oscillations
            ringing_factor = 1.0 + intensity * 2.0  # Scale by intensity
            magnitude[mask, :] *= ringing_factor
            
            # Add oscillating pattern in time (simulating ringing)
            t = np.arange(magnitude.shape[1])
            oscillation = np.sin(2 * np.pi * 0.1 * t) * intensity
            for i in range(sum(mask)):
                idx = np.where(mask)[0][i]
                magnitude[idx, :] *= (1 + oscillation)
            
            # Reconstruct with modified magnitude
            stft_modified = magnitude * np.exp(1j * phase)
            
            # Inverse STFT
            result[ch] = librosa.istft(stft_modified, hop_length=hop_length, length=audio.shape[1])
        
        return result
    
    def _apply_temporal_smearing(self, audio: np.ndarray, params: ArtifactParameters) -> np.ndarray:
        """Apply temporal smearing artifact"""
        intensity = params.intensity
        
        # Design a smearing filter (basically a lowpass in the temporal modulation domain)
        kernel_size = int(0.03 * self.sample_rate * intensity)  # Adjusted by intensity
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
            
        kernel = np.hanning(kernel_size)
        kernel = kernel / kernel.sum()  # Normalize
        
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            # Apply convolution for smearing effect
            result[ch] = scipy.signal.convolve(audio[ch], kernel, mode='same')
            
        return result
    
    def _apply_spectral_holes(self, audio: np.ndarray, params: ArtifactParameters) -> np.ndarray:
        """Apply spectral holes artifacts"""
        intensity = params.intensity
        freq_range = params.frequency_range or (500, 4000)  # Default to mid range
        
        # Number of spectral holes to create
        n_holes = int(1 + intensity * 5)  # More holes with higher intensity
        
        # Generate random hole centers within the frequency range
        min_freq, max_freq = freq_range
        hole_centers = np.random.uniform(min_freq, max_freq, n_holes)
        
        # Hole width based on intensity
        hole_widths = np.random.uniform(50, 300 * intensity, n_holes)
        
        # Convert to frequency domain
        n_fft = 2048
        hop_length = 512
        result = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            # STFT
            stft = librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop_length)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
            
            # Create mask for each spectral hole
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply spectral holes
            for i in range(n_holes):
                center = hole_centers[i]
                width = hole_widths[i]
                
                # Create Gaussian-shaped hole
                hole_mask = np.exp(-0.5 * ((freqs - center) / (width/2)) ** 2)
                hole_mask = 1 - hole_mask * intensity  # Scale by intensity
                
                # Apply the hole mask (column-wise multiplication)
                magnitude = magnitude * hole_mask[:, np.newaxis]
            
            # Reconstruct with modified magnitude
            stft_modified = magnitude * np.exp(1j * phase)
            
            # Inverse STFT
            result[ch] = librosa.istft(stft_modified, hop_length=hop_length, length=audio.shape[1])
        
        return result
    
    def _apply_quantization_noise(self, audio: np.ndarray, params: ArtifactParameters) -> np.ndarray:
        """Apply quantization noise artifacts"""
        intensity = params.intensity
        
        # Calculate effective bit depth reduction based on intensity
        # Full 16-bit at intensity=0, down to 4-bit at intensity=1
        bit_depth = int(16 - intensity * 12)
        
        # Apply bit depth reduction
        result = audio.copy()
        max_val = np.max(np.abs(audio))
        if max_val > 0:  # Prevent division by zero
            factor = 2 ** bit_depth
            result = np.round(audio / max_val * factor) / factor * max_val
            
        return result
    
    def _apply_aliasing(self, audio: np.ndarray, params: ArtifactParameters) -> np.ndarray:
        """Apply aliasing artifacts through downsampling and poor reconstruction"""
        intensity = params.intensity
        
        # Calculate effective sample rate reduction
        downsample_factor = 1 + int(intensity * 8)  # More downsampling with higher intensity
        if downsample_factor <= 1:
            return audio.copy()
            
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            # Downsample by taking every Nth sample
            downsampled = audio[ch, ::downsample_factor]
            
            # Poorly upsample through nearest neighbor interpolation
            upsampled = np.repeat(downsampled, downsample_factor)
            
            # Ensure same length as original
            result[ch, :len(upsampled)] = upsampled[:audio.shape[1]]
            
        return result
    
    def _apply_background_bleed(self, audio: np.ndarray, params: ArtifactParameters) -> np.ndarray:
        """Apply background bleed artifacts by mixing in other content"""
        intensity = params.intensity
        bleed_audio = params.additional_params['bleed_audio']
        
        # Ensure bleed audio is compatible shape
        if bleed_audio.shape[1] < audio.shape[1]:
            padded = np.zeros_like(audio)
            padded[:, :bleed_audio.shape[1]] = bleed_audio
            bleed_audio = padded
        else:
            bleed_audio = bleed_audio[:, :audio.shape[1]]
        
        # Mix in bleed audio based on intensity
        mix_ratio = intensity * 0.3  # Max 30% bleed at intensity=1
        result = (1 - mix_ratio) * audio + mix_ratio * bleed_audio
        
        return result
    
    def _apply_transient_suppression(self, audio: np.ndarray, params: ArtifactParameters) -> np.ndarray:
        """Apply transient suppression artifacts"""
        intensity = params.intensity
        
        # Detect transients using spectral flux
        n_fft = 512  # Smaller window to better capture transients
        hop_length = 128
        
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            # STFT
            stft = librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Calculate spectral flux (difference between consecutive frames)
            flux = np.zeros(magnitude.shape[1])
            for t in range(1, magnitude.shape[1]):
                flux[t] = np.sum((magnitude[:, t] - magnitude[:, t-1]).clip(min=0))
            
            # Normalize flux
            if np.max(flux) > 0:
                flux = flux / np.max(flux)
            
            # Identify transient regions
            threshold = 0.4  # Threshold for transient detection
            transient_frames = flux > threshold
            
            # Create a suppression mask
            suppression = np.ones(magnitude.shape[1])
            suppression[transient_frames] = 1.0 - intensity * 0.9  # Suppress transients by up to 90%
            
            # Apply suppression
            magnitude = magnitude * suppression[np.newaxis, :]
            
            # Reconstruct
            stft_modified = magnitude * np.exp(1j * phase)
            
            # Inverse STFT
            result[ch] = librosa.istft(stft_modified, hop_length=hop_length, length=audio.shape[1])
            
        return result

def run_validation(
    processor: ArtifactProcessor,
    test_cases: List[Tuple[AudioSegment, AudioSegment, AudioSegment]],
    artifact_types: List[str]
) -> Dict[str, List[ValidationMetrics]]:
    """Run validation suite on processor"""
    
    results = {artifact_type: [] for artifact_type in artifact_types}
    
    for clean, artifacts, mixed in test_cases:
        for artifact_type in artifact_types:
            # Time and memory measurement wrapper
            start_time = time.time()
            start_mem = psutil.Process().memory_info().rss
            
            # Process audio
            processed = processor.process(mixed, artifact_type)
            
            # Get metrics
            metrics = processor.validate(clean, processed, artifacts)
            metrics.processing_time = time.time() - start_time
            metrics.memory_usage = (
                psutil.Process().memory_info().rss - start_mem
            ) / 1024 / 1024  # MB
            
            results[artifact_type].append(metrics)
    
    return results