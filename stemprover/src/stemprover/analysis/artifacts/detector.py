import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from enum import Enum

from ...core.audio import AudioSegment
from ...core.types import ProcessingConfig
from ...common.math_utils import phase_difference, phase_coherence
from .base import ArtifactType


@dataclass
class ArtifactDetectionResult:
    """Results from artifact detection"""
    artifact_types: Dict[ArtifactType, float]  # Type to confidence mapping
    time_ranges: Dict[ArtifactType, List[Tuple[float, float]]]  # Type to list of (start, end) times
    freq_ranges: Dict[ArtifactType, List[Tuple[float, float]]]  # Type to list of (low, high) freqs
    overall_severity: float  # 0.0 to 1.0 overall severity
    
    def get_most_likely_artifact(self) -> Tuple[ArtifactType, float]:
        """Get the most likely artifact type and its confidence"""
        if not self.artifact_types:
            return None, 0.0
        
        most_likely = max(self.artifact_types.items(), key=lambda x: x[1])
        return most_likely
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "artifact_types": {atype.name: conf for atype, conf in self.artifact_types.items()},
            "time_ranges": {atype.name: ranges for atype, ranges in self.time_ranges.items()},
            "freq_ranges": {atype.name: ranges for atype, ranges in self.freq_ranges.items()},
            "overall_severity": self.overall_severity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArtifactDetectionResult':
        """Create from dictionary (JSON deserialization)"""
        # Convert string keys back to ArtifactType enum
        artifact_types = {
            getattr(ArtifactType, name): conf 
            for name, conf in data["artifact_types"].items()
        }
        
        time_ranges = {
            getattr(ArtifactType, name): ranges 
            for name, ranges in data["time_ranges"].items()
        }
        
        freq_ranges = {
            getattr(ArtifactType, name): ranges 
            for name, ranges in data["freq_ranges"].items()
        }
        
        return cls(
            artifact_types=artifact_types,
            time_ranges=time_ranges,
            freq_ranges=freq_ranges,
            overall_severity=data["overall_severity"]
        )


class ArtifactDetector:
    """Base class for artifact detection"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize detector"""
        self.config = config or ProcessingConfig()
        
    def detect_artifacts(self, 
                        clean: AudioSegment,
                        artifact: AudioSegment) -> ArtifactDetectionResult:
        """
        Detect artifacts by comparing clean and potentially artifacted audio
        
        Args:
            clean: Clean reference audio
            artifact: Audio with potential artifacts
            
        Returns:
            Detection results with artifact types and their likelihoods
        """
        # Initialize result containers
        artifact_types = {}
        time_ranges = {atype: [] for atype in ArtifactType}
        freq_ranges = {atype: [] for atype in ArtifactType}
        
        # Detect phase incoherence
        phase_score, phase_time_ranges, phase_freq_ranges = self._detect_phase_incoherence(
            clean, artifact
        )
        artifact_types[ArtifactType.PHASE_INCOHERENCE] = phase_score
        time_ranges[ArtifactType.PHASE_INCOHERENCE] = phase_time_ranges
        freq_ranges[ArtifactType.PHASE_INCOHERENCE] = phase_freq_ranges
        
        # Detect high frequency ringing
        ringing_score, ringing_time_ranges, ringing_freq_ranges = self._detect_high_freq_ringing(
            clean, artifact
        )
        artifact_types[ArtifactType.HIGH_FREQUENCY_RINGING] = ringing_score
        time_ranges[ArtifactType.HIGH_FREQUENCY_RINGING] = ringing_time_ranges
        freq_ranges[ArtifactType.HIGH_FREQUENCY_RINGING] = ringing_freq_ranges
        
        # Detect temporal smearing
        smearing_score, smearing_time_ranges = self._detect_temporal_smearing(
            clean, artifact
        )
        artifact_types[ArtifactType.TEMPORAL_SMEARING] = smearing_score
        time_ranges[ArtifactType.TEMPORAL_SMEARING] = smearing_time_ranges
        
        # Detect spectral holes
        holes_score, holes_freq_ranges = self._detect_spectral_holes(
            clean, artifact
        )
        artifact_types[ArtifactType.SPECTRAL_HOLES] = holes_score
        freq_ranges[ArtifactType.SPECTRAL_HOLES] = holes_freq_ranges
        
        # Detect quantization noise
        quant_score = self._detect_quantization_noise(
            clean, artifact
        )
        artifact_types[ArtifactType.QUANTIZATION_NOISE] = quant_score
        
        # Detect aliasing
        aliasing_score, aliasing_freq_ranges = self._detect_aliasing(
            clean, artifact
        )
        artifact_types[ArtifactType.ALIASING] = aliasing_score
        freq_ranges[ArtifactType.ALIASING] = aliasing_freq_ranges
        
        # Detect background bleed (no clean/artifact comparison possible here,
        # but we can look for inconsistent energy patterns)
        bleed_score = self._detect_background_bleed(
            artifact
        )
        artifact_types[ArtifactType.BACKGROUND_BLEED] = bleed_score
        
        # Detect transient suppression
        transient_score, transient_time_ranges = self._detect_transient_suppression(
            clean, artifact
        )
        artifact_types[ArtifactType.TRANSIENT_SUPPRESSION] = transient_score
        time_ranges[ArtifactType.TRANSIENT_SUPPRESSION] = transient_time_ranges
        
        # Calculate overall severity (weighted average of all scores)
        weights = {
            ArtifactType.PHASE_INCOHERENCE: 1.0,
            ArtifactType.HIGH_FREQUENCY_RINGING: 1.0,
            ArtifactType.TEMPORAL_SMEARING: 1.0,
            ArtifactType.SPECTRAL_HOLES: 0.8,
            ArtifactType.QUANTIZATION_NOISE: 0.6,
            ArtifactType.ALIASING: 0.7,
            ArtifactType.BACKGROUND_BLEED: 0.8,
            ArtifactType.TRANSIENT_SUPPRESSION: 0.9
        }
        
        weighted_sum = sum(score * weights[atype] for atype, score in artifact_types.items())
        weight_sum = sum(weights.values())
        overall_severity = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        return ArtifactDetectionResult(
            artifact_types=artifact_types,
            time_ranges=time_ranges,
            freq_ranges=freq_ranges,
            overall_severity=overall_severity
        )
    
    def detect_artifacts_blind(self, audio: AudioSegment) -> ArtifactDetectionResult:
        """
        Detect artifacts in audio without clean reference
        
        Args:
            audio: Audio with potential artifacts
            
        Returns:
            Detection results with artifact types and their likelihoods
        """
        # Initialize result containers
        artifact_types = {}
        time_ranges = {atype: [] for atype in ArtifactType}
        freq_ranges = {atype: [] for atype in ArtifactType}
        
        # Detect high frequency ringing - look for unusual energy patterns
        ringing_score, ringing_time_ranges, ringing_freq_ranges = self._detect_high_freq_ringing_blind(
            audio
        )
        artifact_types[ArtifactType.HIGH_FREQUENCY_RINGING] = ringing_score
        time_ranges[ArtifactType.HIGH_FREQUENCY_RINGING] = ringing_time_ranges
        freq_ranges[ArtifactType.HIGH_FREQUENCY_RINGING] = ringing_freq_ranges
        
        # Detect spectral holes - look for unusual gaps in spectrum
        holes_score, holes_freq_ranges = self._detect_spectral_holes_blind(
            audio
        )
        artifact_types[ArtifactType.SPECTRAL_HOLES] = holes_score
        freq_ranges[ArtifactType.SPECTRAL_HOLES] = holes_freq_ranges
        
        # Detect quantization noise - analyze sample distribution
        quant_score = self._detect_quantization_noise_blind(
            audio
        )
        artifact_types[ArtifactType.QUANTIZATION_NOISE] = quant_score
        
        # Detect aliasing - look for unusual high frequency content
        aliasing_score, aliasing_freq_ranges = self._detect_aliasing_blind(
            audio
        )
        artifact_types[ArtifactType.ALIASING] = aliasing_score
        freq_ranges[ArtifactType.ALIASING] = aliasing_freq_ranges
        
        # Detect phase issues - look for stereo inconsistencies
        if audio.audio.shape[0] > 1:  # Only for stereo audio
            phase_score = self._detect_phase_issues_blind(
                audio
            )
            artifact_types[ArtifactType.PHASE_INCOHERENCE] = phase_score
        else:
            artifact_types[ArtifactType.PHASE_INCOHERENCE] = 0.0
        
        # Cannot reliably detect temporal smearing without reference
        artifact_types[ArtifactType.TEMPORAL_SMEARING] = 0.0
        
        # Cannot reliably detect background bleed without reference
        artifact_types[ArtifactType.BACKGROUND_BLEED] = 0.0
        
        # Cannot reliably detect transient suppression without reference
        artifact_types[ArtifactType.TRANSIENT_SUPPRESSION] = 0.0
        
        # Calculate overall severity (weighted average of all scores)
        # Lower weights for blind detection due to less certainty
        weights = {
            ArtifactType.PHASE_INCOHERENCE: 0.7,
            ArtifactType.HIGH_FREQUENCY_RINGING: 0.8,
            ArtifactType.TEMPORAL_SMEARING: 0.0,  # Cannot detect blindly
            ArtifactType.SPECTRAL_HOLES: 0.7,
            ArtifactType.QUANTIZATION_NOISE: 0.5,
            ArtifactType.ALIASING: 0.6,
            ArtifactType.BACKGROUND_BLEED: 0.0,  # Cannot detect blindly
            ArtifactType.TRANSIENT_SUPPRESSION: 0.0  # Cannot detect blindly
        }
        
        weighted_sum = sum(score * weights[atype] for atype, score in artifact_types.items())
        weight_sum = sum(weights.values())
        overall_severity = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        return ArtifactDetectionResult(
            artifact_types=artifact_types,
            time_ranges=time_ranges,
            freq_ranges=freq_ranges,
            overall_severity=overall_severity
        )
    
    def _detect_phase_incoherence(self, 
                                clean: AudioSegment, 
                                artifact: AudioSegment) -> Tuple[float, List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Detect phase incoherence artifacts
        
        Returns:
            Tuple of (confidence score, time ranges, frequency ranges)
        """
        # Compute STFTs
        n_fft = 2048
        hop_length = 512
        
        clean_mono = clean.audio.mean(axis=0) if clean.audio.ndim > 1 else clean.audio.flatten()
        artifact_mono = artifact.audio.mean(axis=0) if artifact.audio.ndim > 1 else artifact.audio.flatten()
        
        clean_stft = librosa.stft(clean_mono, n_fft=n_fft, hop_length=hop_length)
        artifact_stft = librosa.stft(artifact_mono, n_fft=n_fft, hop_length=hop_length)
        
        # Calculate phase difference
        phase_diff = np.abs(np.angle(clean_stft) - np.angle(artifact_stft))
        
        # Calculate coherence (1 = perfect, 0 = random)
        coherence = np.cos(phase_diff)
        
        # Calculate confidence score (1 - average coherence, scaled)
        # Higher score means more phase incoherence
        score = (1 - np.mean(coherence)) * 2.0  # Scale for better range
        score = min(1.0, max(0.0, score))  # Clamp to [0, 1]
        
        # Find time and frequency ranges with significant phase issues
        significant_issues = coherence < 0.5  # Threshold for significant issues
        
        # Get frequency bins for affected ranges
        freqs = librosa.fft_frequencies(sr=clean.sample_rate, n_fft=n_fft)
        freq_ranges = []
        
        # Calculate average coherence per frequency bin
        avg_coherence_by_freq = np.mean(coherence, axis=1)
        
        # Find continuous frequency ranges with low coherence
        problematic_bins = np.where(avg_coherence_by_freq < 0.5)[0]
        
        if len(problematic_bins) > 0:
            # Group adjacent bins
            groups = []
            current_group = [problematic_bins[0]]
            
            for i in range(1, len(problematic_bins)):
                if problematic_bins[i] == problematic_bins[i-1] + 1:
                    current_group.append(problematic_bins[i])
                else:
                    groups.append(current_group)
                    current_group = [problematic_bins[i]]
            
            groups.append(current_group)
            
            # Convert bin groups to frequency ranges
            for group in groups:
                if len(group) > 3:  # Ignore very small ranges
                    low_freq = freqs[group[0]]
                    high_freq = freqs[group[-1]]
                    freq_ranges.append((float(low_freq), float(high_freq)))
        
        # Calculate average coherence per time frame
        frames = significant_issues.shape[1]
        times = librosa.frames_to_time(
            np.arange(frames),
            sr=clean.sample_rate,
            hop_length=hop_length
        )
        
        # Find time ranges with significant issues
        time_ranges = []
        
        # Significant issue if more than 25% of frequency bins have low coherence
        frames_with_issues = np.mean(significant_issues, axis=0) > 0.25
        
        if np.any(frames_with_issues):
            # Group adjacent problematic frames
            problematic_frames = np.where(frames_with_issues)[0]
            
            if len(problematic_frames) > 0:
                # Group adjacent frames
                groups = []
                current_group = [problematic_frames[0]]
                
                for i in range(1, len(problematic_frames)):
                    if problematic_frames[i] == problematic_frames[i-1] + 1:
                        current_group.append(problematic_frames[i])
                    else:
                        groups.append(current_group)
                        current_group = [problematic_frames[i]]
                
                groups.append(current_group)
                
                # Convert frame groups to time ranges
                for group in groups:
                    if len(group) > 2:  # Ignore very short issues
                        start_time = times[group[0]]
                        end_time = times[group[-1]]
                        time_ranges.append((float(start_time), float(end_time)))
        
        return score, time_ranges, freq_ranges
    
    def _detect_high_freq_ringing(self, 
                                clean: AudioSegment, 
                                artifact: AudioSegment) -> Tuple[float, List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Detect high frequency ringing artifacts
        
        Returns:
            Tuple of (confidence score, time ranges, frequency ranges)
        """
        # Compute STFTs
        n_fft = 2048
        hop_length = 512
        
        clean_mono = clean.audio.mean(axis=0) if clean.audio.ndim > 1 else clean.audio.flatten()
        artifact_mono = artifact.audio.mean(axis=0) if artifact.audio.ndim > 1 else artifact.audio.flatten()
        
        clean_stft = librosa.stft(clean_mono, n_fft=n_fft, hop_length=hop_length)
        artifact_stft = librosa.stft(artifact_mono, n_fft=n_fft, hop_length=hop_length)
        
        # Get magnitudes
        clean_mag = np.abs(clean_stft)
        artifact_mag = np.abs(artifact_stft)
        
        # Calculate frequency bins
        freqs = librosa.fft_frequencies(sr=clean.sample_rate, n_fft=n_fft)
        
        # Focus on high frequencies (above 5000 Hz)
        high_freq_mask = freqs > 5000
        
        if not np.any(high_freq_mask):
            return 0.0, [], []  # No high frequencies to analyze
        
        # Calculate energy ratio in high frequencies
        clean_energy = np.sum(clean_mag[high_freq_mask, :] ** 2)
        artifact_energy = np.sum(artifact_mag[high_freq_mask, :] ** 2)
        
        # Higher score if artifact has more high frequency energy
        if clean_energy > 0:
            energy_ratio = artifact_energy / clean_energy
            # Score is 0 if ratio <= 1 (no extra energy)
            # Score increases as ratio increases, capped at 1.0
            score = min(1.0, max(0.0, (energy_ratio - 1.0)))
        else:
            # If clean has no energy but artifact does, that's suspicious
            score = 1.0 if artifact_energy > 0 else 0.0
        
        # Identify frequency ranges with ringing
        freq_ranges = []
        
        # Look at each high frequency bin
        energy_ratio_by_freq = np.zeros(len(freqs))
        
        for i in range(len(freqs)):
            if freqs[i] > 5000:
                clean_bin_energy = np.sum(clean_mag[i, :] ** 2)
                artifact_bin_energy = np.sum(artifact_mag[i, :] ** 2)
                
                if clean_bin_energy > 0:
                    energy_ratio_by_freq[i] = artifact_bin_energy / clean_bin_energy
                elif artifact_bin_energy > 0:
                    energy_ratio_by_freq[i] = float('inf')  # Artifact energy but no clean energy
                else:
                    energy_ratio_by_freq[i] = 1.0  # No energy in either
        
        # Find ranges with high energy ratio
        significant_ratio = energy_ratio_by_freq > 1.5  # 50% more energy
        
        # Group adjacent bins
        problematic_bins = np.where(significant_ratio)[0]
        
        if len(problematic_bins) > 0:
            groups = []
            current_group = [problematic_bins[0]]
            
            for i in range(1, len(problematic_bins)):
                if problematic_bins[i] == problematic_bins[i-1] + 1:
                    current_group.append(problematic_bins[i])
                else:
                    groups.append(current_group)
                    current_group = [problematic_bins[i]]
            
            groups.append(current_group)
            
            # Convert bin groups to frequency ranges
            for group in groups:
                if len(group) > 3 and freqs[group[0]] > 5000:  # Ensure high frequency
                    low_freq = freqs[group[0]]
                    high_freq = freqs[group[-1]]
                    freq_ranges.append((float(low_freq), float(high_freq)))
        
        # Find time ranges with ringing
        time_ranges = []
        
        # Look for oscillating patterns in high frequencies over time
        frames = clean_mag.shape[1]
        frame_energy_ratio = np.zeros(frames)
        
        for i in range(frames):
            clean_frame_energy = np.sum(clean_mag[high_freq_mask, i] ** 2)
            artifact_frame_energy = np.sum(artifact_mag[high_freq_mask, i] ** 2)
            
            if clean_frame_energy > 0:
                frame_energy_ratio[i] = artifact_frame_energy / clean_frame_energy
            elif artifact_frame_energy > 0:
                frame_energy_ratio[i] = float('inf')
            else:
                frame_energy_ratio[i] = 1.0
        
        # Detect ringing by looking for oscillating energy patterns
        # (high energy followed by lower energy repeatedly)
        times = librosa.frames_to_time(
            np.arange(frames),
            sr=clean.sample_rate,
            hop_length=hop_length
        )
        
        frames_with_ringing = frame_energy_ratio > 1.5
        
        if np.any(frames_with_ringing):
            # Group adjacent frames
            problematic_frames = np.where(frames_with_ringing)[0]
            
            if len(problematic_frames) > 0:
                groups = []
                current_group = [problematic_frames[0]]
                
                for i in range(1, len(problematic_frames)):
                    if problematic_frames[i] == problematic_frames[i-1] + 1:
                        current_group.append(problematic_frames[i])
                    else:
                        groups.append(current_group)
                        current_group = [problematic_frames[i]]
                
                groups.append(current_group)
                
                # Convert frame groups to time ranges
                for group in groups:
                    if len(group) > 2:  # Ignore very short issues
                        start_time = times[group[0]]
                        end_time = times[group[-1]]
                        time_ranges.append((float(start_time), float(end_time)))
        
        return score, time_ranges, freq_ranges
    
    def _detect_temporal_smearing(self, 
                                clean: AudioSegment, 
                                artifact: AudioSegment) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Detect temporal smearing artifacts
        
        Returns:
            Tuple of (confidence score, time ranges)
        """
        from scipy.signal import hilbert
        
        # Get envelope
        clean_mono = clean.audio.mean(axis=0) if clean.audio.ndim > 1 else clean.audio.flatten()
        artifact_mono = artifact.audio.mean(axis=0) if artifact.audio.ndim > 1 else artifact.audio.flatten()
        
        clean_env = np.abs(hilbert(clean_mono))
        artifact_env = np.abs(hilbert(artifact_mono))
        
        # Calculate envelope derivative to identify transients
        clean_env_diff = np.diff(clean_env)
        artifact_env_diff = np.diff(artifact_env)
        
        # Normalize for comparison
        if np.max(np.abs(clean_env_diff)) > 0:
            clean_env_diff = clean_env_diff / np.max(np.abs(clean_env_diff))
        
        if np.max(np.abs(artifact_env_diff)) > 0:
            artifact_env_diff = artifact_env_diff / np.max(np.abs(artifact_env_diff))
        
        # Calculate temporal smearing score
        # (Mean absolute difference between derivatives)
        smearing = np.mean(np.abs(clean_env_diff - artifact_env_diff))
        
        # Scale to 0-1 range (empirically determined scale)
        score = min(1.0, smearing * 5.0)
        
        # Find time ranges with significant smearing
        sample_rate = clean.sample_rate
        threshold = 0.2  # Threshold for significant envelope difference
        
        # Calculate difference between normalized derivatives
        diff = np.abs(clean_env_diff - artifact_env_diff)
        
        # Find continuous regions with high difference
        significant_diff = diff > threshold
        
        # Convert to time ranges
        time_ranges = []
        
        if np.any(significant_diff):
            # Group adjacent samples
            problematic_samples = np.where(significant_diff)[0]
            
            if len(problematic_samples) > 0:
                groups = []
                current_group = [problematic_samples[0]]
                
                for i in range(1, len(problematic_samples)):
                    if problematic_samples[i] == problematic_samples[i-1] + 1:
                        current_group.append(problematic_samples[i])
                    else:
                        groups.append(current_group)
                        current_group = [problematic_samples[i]]
                
                groups.append(current_group)
                
                # Convert sample groups to time ranges
                for group in groups:
                    if len(group) > sample_rate // 100:  # Ignore very short issues
                        start_time = group[0] / sample_rate
                        end_time = group[-1] / sample_rate
                        time_ranges.append((float(start_time), float(end_time)))
        
        return score, time_ranges
    
    def _detect_spectral_holes(self, 
                             clean: AudioSegment, 
                             artifact: AudioSegment) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Detect spectral holes artifacts
        
        Returns:
            Tuple of (confidence score, frequency ranges)
        """
        # Compute STFTs
        n_fft = 2048
        hop_length = 512
        
        clean_mono = clean.audio.mean(axis=0) if clean.audio.ndim > 1 else clean.audio.flatten()
        artifact_mono = artifact.audio.mean(axis=0) if artifact.audio.ndim > 1 else artifact.audio.flatten()
        
        clean_stft = librosa.stft(clean_mono, n_fft=n_fft, hop_length=hop_length)
        artifact_stft = librosa.stft(artifact_mono, n_fft=n_fft, hop_length=hop_length)
        
        # Get magnitudes
        clean_mag = np.abs(clean_stft)
        artifact_mag = np.abs(artifact_stft)
        
        # Calculate average spectrum
        clean_spectrum = np.mean(clean_mag, axis=1)
        artifact_spectrum = np.mean(artifact_mag, axis=1)
        
        # Calculate spectral ratio
        spectral_ratio = np.ones_like(clean_spectrum)
        mask = clean_spectrum > 1e-6  # Avoid division by zero
        spectral_ratio[mask] = artifact_spectrum[mask] / clean_spectrum[mask]
        
        # Identify holes (regions where ratio is significantly less than 1)
        hole_threshold = 0.5  # Less than 50% of energy
        hole_mask = spectral_ratio < hole_threshold
        
        # Calculate confidence score based on number and severity of holes
        num_holes = np.sum(hole_mask)
        freq_bins = len(clean_spectrum)
        
        # Score increases with percentage of spectrum affected
        score = min(1.0, num_holes / (freq_bins * 0.2))  # Scale so 20% of spectrum = score of 1.0
        
        # Get frequency ranges for holes
        freqs = librosa.fft_frequencies(sr=clean.sample_rate, n_fft=n_fft)
        freq_ranges = []
        
        if np.any(hole_mask):
            # Group adjacent bins
            hole_bins = np.where(hole_mask)[0]
            
            if len(hole_bins) > 0:
                groups = []
                current_group = [hole_bins[0]]
                
                for i in range(1, len(hole_bins)):
                    if hole_bins[i] == hole_bins[i-1] + 1:
                        current_group.append(hole_bins[i])
                    else:
                        groups.append(current_group)
                        current_group = [hole_bins[i]]
                
                groups.append(current_group)
                
                # Convert bin groups to frequency ranges
                for group in groups:
                    if len(group) > 3:  # Ignore very small holes
                        low_freq = freqs[group[0]]
                        high_freq = freqs[group[-1]]
                        freq_ranges.append((float(low_freq), float(high_freq)))
        
        return score, freq_ranges
    
    def _detect_quantization_noise(self, 
                                 clean: AudioSegment, 
                                 artifact: AudioSegment) -> float:
        """
        Detect quantization noise artifacts
        
        Returns:
            Confidence score
        """
        # Get audio samples
        clean_mono = clean.audio.mean(axis=0) if clean.audio.ndim > 1 else clean.audio.flatten()
        artifact_mono = artifact.audio.mean(axis=0) if artifact.audio.ndim > 1 else artifact.audio.flatten()
        
        # Calculate difference
        diff = artifact_mono - clean_mono
        
        # Normalize signals for comparison
        if np.max(np.abs(clean_mono)) > 0:
            clean_mono = clean_mono / np.max(np.abs(clean_mono))
        
        if np.max(np.abs(artifact_mono)) > 0:
            artifact_mono = artifact_mono / np.max(np.abs(artifact_mono))
        
        # Analyze histograms for evidence of quantization
        clean_hist, _ = np.histogram(clean_mono, bins=1000, range=(-1, 1))
        artifact_hist, _ = np.histogram(artifact_mono, bins=1000, range=(-1, 1))
        
        # Count number of non-zero bins
        clean_bins = np.sum(clean_hist > 0)
        artifact_bins = np.sum(artifact_hist > 0)
        
        # Calculate ratio of bin counts
        if clean_bins > 0:
            bin_ratio = artifact_bins / clean_bins
        else:
            bin_ratio = 1.0  # No information
        
        # Calculate quantization score
        # Lower bin ratio indicates more quantization
        if bin_ratio < 1.0:
            # Score increases as ratio decreases
            score = 1.0 - bin_ratio
        else:
            score = 0.0  # No quantization detected
        
        return score
    
    def _detect_aliasing(self, 
                       clean: AudioSegment, 
                       artifact: AudioSegment) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Detect aliasing artifacts
        
        Returns:
            Tuple of (confidence score, frequency ranges)
        """
        # Compute STFTs
        n_fft = 2048
        hop_length = 512
        
        clean_mono = clean.audio.mean(axis=0) if clean.audio.ndim > 1 else clean.audio.flatten()
        artifact_mono = artifact.audio.mean(axis=0) if artifact.audio.ndim > 1 else artifact.audio.flatten()
        
        clean_stft = librosa.stft(clean_mono, n_fft=n_fft, hop_length=hop_length)
        artifact_stft = librosa.stft(artifact_mono, n_fft=n_fft, hop_length=hop_length)
        
        # Get magnitudes
        clean_mag = np.abs(clean_stft)
        artifact_mag = np.abs(artifact_stft)
        
        # Calculate frequency bins
        freqs = librosa.fft_frequencies(sr=clean.sample_rate, n_fft=n_fft)
        
        # Calculate average spectrum
        clean_spectrum = np.mean(clean_mag, axis=1)
        artifact_spectrum = np.mean(artifact_mag, axis=1)
        
        # Calculate spectral difference
        spectral_diff = artifact_spectrum - clean_spectrum
        
        # Find regions where artifact has more energy (potential aliases)
        alias_mask = spectral_diff > 0
        
        # Calculate aliasing metric
        # (Energy in frequencies present in artifact but not in clean)
        positive_diff_energy = np.sum(spectral_diff[alias_mask]**2)
        total_clean_energy = np.sum(clean_spectrum**2)
        
        if total_clean_energy > 0:
            aliasing_ratio = positive_diff_energy / total_clean_energy
        else:
            aliasing_ratio = 0.0
        
        # Scale to 0-1 range
        score = min(1.0, aliasing_ratio * 5.0)  # Empirically determined scale
        
        # Find frequency ranges with significant excess energy
        freq_ranges = []
        
        significant_diff = spectral_diff > 0.1 * np.max(clean_spectrum)
        
        if np.any(significant_diff):
            # Group adjacent bins
            alias_bins = np.where(significant_diff)[0]
            
            if len(alias_bins) > 0:
                groups = []
                current_group = [alias_bins[0]]
                
                for i in range(1, len(alias_bins)):
                    if alias_bins[i] == alias_bins[i-1] + 1:
                        current_group.append(alias_bins[i])
                    else:
                        groups.append(current_group)
                        current_group = [alias_bins[i]]
                
                groups.append(current_group)
                
                # Convert bin groups to frequency ranges
                for group in groups:
                    if len(group) > 3:  # Ignore very small ranges
                        low_freq = freqs[group[0]]
                        high_freq = freqs[group[-1]]
                        freq_ranges.append((float(low_freq), float(high_freq)))
        
        return score, freq_ranges
    
    def _detect_background_bleed(self, artifact: AudioSegment) -> float:
        """
        Detect background bleed artifacts
        
        Returns:
            Confidence score
        """
        # Without a reference, we'll look for spectral patterns
        # that might indicate background bleed
        
        # For now, return a placeholder score
        # This would need a more sophisticated approach with a trained model
        return 0.1  # Low confidence without reference
    
    def _detect_transient_suppression(self, 
                                    clean: AudioSegment, 
                                    artifact: AudioSegment) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Detect transient suppression artifacts
        
        Returns:
            Tuple of (confidence score, time ranges)
        """
        # Compute onset strength
        from librosa.onset import onset_strength
        
        clean_mono = clean.audio.mean(axis=0) if clean.audio.ndim > 1 else clean.audio.flatten()
        artifact_mono = artifact.audio.mean(axis=0) if artifact.audio.ndim > 1 else artifact.audio.flatten()
        
        hop_length = 512
        
        clean_onset = onset_strength(
            y=clean_mono, 
            sr=clean.sample_rate,
            hop_length=hop_length
        )
        
        artifact_onset = onset_strength(
            y=artifact_mono, 
            sr=artifact.sample_rate,
            hop_length=hop_length
        )
        
        # Normalize for comparison
        if np.max(clean_onset) > 0:
            clean_onset = clean_onset / np.max(clean_onset)
        
        if np.max(artifact_onset) > 0:
            artifact_onset = artifact_onset / np.max(artifact_onset)
        
        # Calculate suppression metrics
        transient_threshold = 0.2  # Threshold for detecting significant transients
        clean_transients = clean_onset > transient_threshold
        artifact_transients = artifact_onset > transient_threshold
        
        # Count preserved vs. suppressed transients
        preserved = np.sum(np.logical_and(clean_transients, artifact_transients))
        suppressed = np.sum(np.logical_and(clean_transients, np.logical_not(artifact_transients)))
        
        if np.sum(clean_transients) > 0:
            suppression_ratio = suppressed / np.sum(clean_transients)
        else:
            suppression_ratio = 0.0
        
        # Score increases with suppression ratio
        score = suppression_ratio
        
        # Find time ranges with suppressed transients
        frames = len(clean_onset)
        times = librosa.frames_to_time(
            np.arange(frames),
            sr=clean.sample_rate,
            hop_length=hop_length
        )
        
        # Detect suppressed transients
        suppressed_mask = np.logical_and(
            clean_transients,
            np.logical_not(artifact_transients)
        )
        
        # Convert to time ranges
        time_ranges = []
        
        if np.any(suppressed_mask):
            # Group adjacent frames
            suppressed_frames = np.where(suppressed_mask)[0]
            
            if len(suppressed_frames) > 0:
                groups = []
                current_group = [suppressed_frames[0]]
                
                for i in range(1, len(suppressed_frames)):
                    if suppressed_frames[i] <= suppressed_frames[i-1] + 2:  # Allow small gaps
                        current_group.append(suppressed_frames[i])
                    else:
                        groups.append(current_group)
                        current_group = [suppressed_frames[i]]
                
                groups.append(current_group)
                
                # Convert frame groups to time ranges
                for group in groups:
                    start_time = times[group[0]]
                    end_time = times[group[-1]]
                    
                    # Add buffer around transient
                    buffer = 0.05  # 50ms buffer
                    time_ranges.append((float(start_time - buffer), float(end_time + buffer)))
        
        return score, time_ranges
    
    # Methods for blind detection (no clean reference)
    
    def _detect_high_freq_ringing_blind(self, 
                                      audio: AudioSegment) -> Tuple[float, List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Detect high frequency ringing without clean reference"""
        # Compute STFT
        n_fft = 2048
        hop_length = 512
        
        audio_mono = audio.audio.mean(axis=0) if audio.audio.ndim > 1 else audio.audio.flatten()
        
        audio_stft = librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop_length)
        
        # Get magnitude
        audio_mag = np.abs(audio_stft)
        
        # Calculate frequency bins
        freqs = librosa.fft_frequencies(sr=audio.sample_rate, n_fft=n_fft)
        
        # Focus on high frequencies (above 5000 Hz)
        high_freq_mask = freqs > 5000
        
        if not np.any(high_freq_mask):
            return 0.0, [], []  # No high frequencies to analyze
        
        # Look for oscillating patterns in high frequencies
        # (indicating ringing)
        high_freq_mag = audio_mag[high_freq_mask, :]
        
        # Calculate time-frequency energy variance
        # (ringing tends to have high temporal variance in specific frequency bins)
        temporal_var = np.var(high_freq_mag, axis=1)
        mean_var = np.mean(temporal_var)
        
        # Normalize by mean energy to get relative variance
        mean_energy = np.mean(high_freq_mag, axis=1)
        relative_var = np.zeros_like(temporal_var)
        mask = mean_energy > 0
        relative_var[mask] = temporal_var[mask] / mean_energy[mask]
        
        # High relative variance suggests ringing
        ringing_score = min(1.0, np.mean(relative_var) * 2.0)  # Scale for better range
        
        # Find frequency ranges with high variance
        freq_ranges = []
        high_freqs = freqs[high_freq_mask]
        high_var_mask = relative_var > 2.0  # Threshold for high variance
        
        if np.any(high_var_mask):
            # Group adjacent bins
            var_bins = np.where(high_var_mask)[0]
            
            if len(var_bins) > 0:
                groups = []
                current_group = [var_bins[0]]
                
                for i in range(1, len(var_bins)):
                    if var_bins[i] == var_bins[i-1] + 1:
                        current_group.append(var_bins[i])
                    else:
                        groups.append(current_group)
                        current_group = [var_bins[i]]
                
                groups.append(current_group)
                
                # Convert bin groups to frequency ranges
                for group in groups:
                    if len(group) > 2:  # Ignore very small ranges
                        low_freq = high_freqs[group[0]]
                        high_freq = high_freqs[group[-1]]
                        freq_ranges.append((float(low_freq), float(high_freq)))
        
        # Find time ranges with ringing
        time_ranges = []
        frames = audio_mag.shape[1]
        times = librosa.frames_to_time(
            np.arange(frames),
            sr=audio.sample_rate,
            hop_length=hop_length
        )
        
        # Look for regions with high high-frequency energy
        high_energy_frames = np.mean(high_freq_mag, axis=0) > np.mean(high_freq_mag) * 1.5
        
        if np.any(high_energy_frames):
            # Group adjacent frames
            energy_frames = np.where(high_energy_frames)[0]
            
            if len(energy_frames) > 0:
                groups = []
                current_group = [energy_frames[0]]
                
                for i in range(1, len(energy_frames)):
                    if energy_frames[i] == energy_frames[i-1] + 1:
                        current_group.append(energy_frames[i])
                    else:
                        groups.append(current_group)
                        current_group = [energy_frames[i]]
                
                groups.append(current_group)
                
                # Convert frame groups to time ranges
                for group in groups:
                    if len(group) > 2:  # Ignore very short issues
                        start_time = times[group[0]]
                        end_time = times[group[-1]]
                        time_ranges.append((float(start_time), float(end_time)))
        
        return ringing_score, time_ranges, freq_ranges
    
    def _detect_spectral_holes_blind(self, 
                                   audio: AudioSegment) -> Tuple[float, List[Tuple[float, float]]]:
        """Detect spectral holes without clean reference"""
        # Compute STFT
        n_fft = 2048
        hop_length = 512
        
        audio_mono = audio.audio.mean(axis=0) if audio.audio.ndim > 1 else audio.audio.flatten()
        
        audio_stft = librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop_length)
        
        # Get magnitude
        audio_mag = np.abs(audio_stft)
        
        # Calculate average spectrum
        spectrum = np.mean(audio_mag, axis=1)
        
        # Calculate frequency bins
        freqs = librosa.fft_frequencies(sr=audio.sample_rate, n_fft=n_fft)
        
        # Filter out DC and near-silent regions
        spectrum = spectrum[1:]
        freqs = freqs[1:]
        
        # Smooth the spectrum to find expected contour
        from scipy.signal import savgol_filter
        
        if len(spectrum) > 11:  # Need enough points for filter
            smoothed_spectrum = savgol_filter(spectrum, 11, 3)
        else:
            smoothed_spectrum = spectrum
        
        # Look for dips in the spectrum compared to smoothed version
        ratio = np.ones_like(spectrum)
        mask = smoothed_spectrum > 1e-6
        ratio[mask] = spectrum[mask] / smoothed_spectrum[mask]
        
        # Find holes (large dips in spectrum)
        hole_threshold = 0.3  # Less than 30% of expected energy
        hole_mask = ratio < hole_threshold
        
        # Calculate confidence score
        num_holes = np.sum(hole_mask)
        freq_bins = len(spectrum)
        
        # Score increases with percentage of spectrum affected
        score = min(1.0, num_holes / (freq_bins * 0.1))  # Scale so 10% of spectrum = score of 1.0
        
        # Find frequency ranges for holes
        freq_ranges = []
        
        if np.any(hole_mask):
            # Group adjacent bins
            hole_bins = np.where(hole_mask)[0]
            
            if len(hole_bins) > 0:
                groups = []
                current_group = [hole_bins[0]]
                
                for i in range(1, len(hole_bins)):
                    if hole_bins[i] == hole_bins[i-1] + 1:
                        current_group.append(hole_bins[i])
                    else:
                        groups.append(current_group)
                        current_group = [hole_bins[i]]
                
                groups.append(current_group)
                
                # Convert bin groups to frequency ranges
                for group in groups:
                    if len(group) > 3:  # Ignore very small holes
                        low_freq = freqs[group[0]]
                        high_freq = freqs[group[-1]]
                        freq_ranges.append((float(low_freq), float(high_freq)))
        
        return score, freq_ranges
    
    def _detect_quantization_noise_blind(self, audio: AudioSegment) -> float:
        """Detect quantization noise without clean reference"""
        # Get audio samples
        audio_mono = audio.audio.mean(axis=0) if audio.audio.ndim > 1 else audio.audio.flatten()
        
        # Normalize for analysis
        if np.max(np.abs(audio_mono)) > 0:
            audio_mono = audio_mono / np.max(np.abs(audio_mono))
        
        # Analyze histogram for evidence of quantization
        hist, bin_edges = np.histogram(audio_mono, bins=1000, range=(-1, 1))
        
        # Calculate bin utilization
        utilization = np.sum(hist > 0) / len(hist)
        
        # Look for regular patterns in histogram (signs of quantization)
        # Count number of non-empty bins
        non_empty_bins = np.sum(hist > 0)
        
        # Calculate average distance between non-empty bins
        non_empty_indices = np.where(hist > 0)[0]
        if len(non_empty_indices) > 1:
            distances = np.diff(non_empty_indices)
            distance_counts = np.bincount(distances)
            regularity = np.max(distance_counts) / np.sum(distance_counts)
        else:
            regularity = 0.0
        
        # High regularity and low utilization indicate quantization
        score = (regularity * 0.7 + (1 - utilization) * 0.3)
        
        return score
    
    def _detect_aliasing_blind(self, 
                             audio: AudioSegment) -> Tuple[float, List[Tuple[float, float]]]:
        """Detect aliasing without clean reference"""
        # Compute STFT
        n_fft = 2048
        hop_length = 512
        
        audio_mono = audio.audio.mean(axis=0) if audio.audio.ndim > 1 else audio.audio.flatten()
        
        audio_stft = librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop_length)
        
        # Get magnitude
        audio_mag = np.abs(audio_stft)
        
        # Calculate frequency bins
        freqs = librosa.fft_frequencies(sr=audio.sample_rate, n_fft=n_fft)
        
        # Calculate average spectrum
        spectrum = np.mean(audio_mag, axis=1)
        
        # Look for unusual spikes in high frequencies
        # Normally high frequencies have less energy in natural signals
        high_freq_mask = freqs > 10000
        
        if not np.any(high_freq_mask):
            return 0.0, []  # No high frequencies to analyze
        
        # Calculate average spectral slope in lower frequencies
        low_freq_mask = (freqs > 1000) & (freqs < 8000)
        
        if not np.any(low_freq_mask):
            return 0.0, []  # Not enough frequency range
        
        # Calculate expected spectrum decline
        log_low_freqs = np.log10(freqs[low_freq_mask])
        log_low_spectrum = np.log10(spectrum[low_freq_mask] + 1e-10)
        
        # Linear regression to find spectral slope
        from scipy.stats import linregress
        
        slope, intercept, _, _, _ = linregress(log_low_freqs, log_low_spectrum)
        
        # Predict expected high frequency spectrum
        log_high_freqs = np.log10(freqs[high_freq_mask])
        expected_log_spectrum = intercept + slope * log_high_freqs
        expected_spectrum = 10 ** expected_log_spectrum
        
        # Compare actual to expected
        actual_spectrum = spectrum[high_freq_mask]
        ratio = np.zeros_like(actual_spectrum)
        mask = expected_spectrum > 1e-10
        ratio[mask] = actual_spectrum[mask] / expected_spectrum[mask]
        
        # Unusually high energy could indicate aliasing
        aliasing_threshold = 2.0  # Twice expected energy
        aliasing_mask = ratio > aliasing_threshold
        
        # Calculate aliasing score
        aliasing_energy = np.sum(actual_spectrum[aliasing_mask])
        total_energy = np.sum(spectrum)
        
        if total_energy > 0:
            aliasing_score = min(1.0, aliasing_energy / total_energy * 20.0)  # Scale for better range
        else:
            aliasing_score = 0.0
        
        # Find frequency ranges with aliasing
        freq_ranges = []
        high_freqs = freqs[high_freq_mask]
        
        if np.any(aliasing_mask):
            # Group adjacent bins
            aliasing_bins = np.where(aliasing_mask)[0]
            
            if len(aliasing_bins) > 0:
                groups = []
                current_group = [aliasing_bins[0]]
                
                for i in range(1, len(aliasing_bins)):
                    if aliasing_bins[i] == aliasing_bins[i-1] + 1:
                        current_group.append(aliasing_bins[i])
                    else:
                        groups.append(current_group)
                        current_group = [aliasing_bins[i]]
                
                groups.append(current_group)
                
                # Convert bin groups to frequency ranges
                for group in groups:
                    if len(group) > 2:  # Ignore very small ranges
                        low_freq = high_freqs[group[0]]
                        high_freq = high_freqs[group[-1]]
                        freq_ranges.append((float(low_freq), float(high_freq)))
        
        return aliasing_score, freq_ranges
    
    def _detect_phase_issues_blind(self, audio: AudioSegment) -> float:
        """Detect phase issues in stereo audio without clean reference"""
        # Only works for stereo audio
        if audio.audio.shape[0] < 2:
            return 0.0
        
        # Extract left and right channels
        left = audio.audio[0]
        right = audio.audio[1]
        
        # Compute STFTs
        n_fft = 2048
        hop_length = 512
        
        left_stft = librosa.stft(left, n_fft=n_fft, hop_length=hop_length)
        right_stft = librosa.stft(right, n_fft=n_fft, hop_length=hop_length)
        
        # Calculate phase difference between channels
        left_phase = np.angle(left_stft)
        right_phase = np.angle(right_stft)
        phase_diff = np.abs(left_phase - right_phase)
        
        # Calculate coherence (1 = perfectly aligned, 0 = random)
        coherence = np.cos(phase_diff)
        
        # Normally channels should have similar phase in low-mid frequencies
        # Calculate frequency bins
        freqs = librosa.fft_frequencies(sr=audio.sample_rate, n_fft=n_fft)
        
        # Focus on low-mid frequencies (where phase issues are more noticeable)
        freq_mask = (freqs > 100) & (freqs < 4000)
        
        # Calculate average coherence in this range
        mid_coherence = np.mean(coherence[freq_mask, :])
        
        # Invert to get a score (higher = more phase issues)
        score = 1.0 - mid_coherence
        
        # Scale to make more sensitive (phase issues often subtle)
        score = min(1.0, score * 2.0)
        
        return score


class NeuralArtifactDetector(ArtifactDetector):
    """Neural network-based artifact detector"""
    
    def __init__(self, model_path: Optional[Path] = None, config: Optional[ProcessingConfig] = None):
        """Initialize with an optional pre-trained model"""
        super().__init__(config)
        
        # Placeholder for future neural network implementation
        self.model = None
        
        if model_path is not None:
            self._load_model(model_path)
    
    def _load_model(self, model_path: Path) -> None:
        """Load a pre-trained model"""
        # Placeholder for model loading
        pass
    
    def detect_artifacts_nn(self, audio: AudioSegment) -> ArtifactDetectionResult:
        """Use neural network to detect artifacts without reference"""
        # Placeholder for neural network inference
        return ArtifactDetectionResult(
            artifact_types={},
            time_ranges={ArtifactType.PHASE_INCOHERENCE: []},
            freq_ranges={ArtifactType.PHASE_INCOHERENCE: []},
            overall_severity=0.0
        )