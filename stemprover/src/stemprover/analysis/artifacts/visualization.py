import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import seaborn as sns
from dataclasses import dataclass

from ...core.audio import AudioSegment
from ...core.types import ProcessingConfig
from ...common.math_utils import (
    magnitude, phase_coherence, phase_difference, db_scale
)
from .base import ArtifactType, ArtifactParameters


@dataclass
class ArtifactVisualizationConfig:
    """Configuration for artifact visualization"""
    fig_size: Tuple[int, int] = (15, 10)
    cmap_magnitude: str = "viridis"
    cmap_phase: str = "twilight"
    cmap_diff: str = "RdBu_r"
    db_range: Tuple[float, float] = (-80, 0)
    freq_scale: str = "log"  # 'linear' or 'log'
    time_unit: str = "s"
    fft_size: int = 2048
    hop_length: int = 512
    max_freq: Optional[int] = None  # Hz, maximum frequency to show


class ArtifactVisualizer:
    """Visualization tools for artifact analysis"""
    
    def __init__(self, 
                 output_dir: Path, 
                 config: Optional[ProcessingConfig] = None,
                 viz_config: Optional[ArtifactVisualizationConfig] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.config = config or ProcessingConfig()
        self.viz_config = viz_config or ArtifactVisualizationConfig()
    
    def visualize_artifact(self, 
                          clean: AudioSegment, 
                          artifact: AudioSegment, 
                          artifact_type: ArtifactType,
                          save_path: Optional[Path] = None) -> Path:
        """Generate comprehensive visualization of an artifact"""
        # Create output path
        if save_path is None:
            save_path = self.output_dir / f"artifact_{artifact_type.name.lower()}.png"
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=self.viz_config.fig_size)
        
        # 1. Waveform comparison
        ax1 = plt.subplot(3, 2, 1)
        self._plot_waveform_comparison(clean, artifact, ax1)
        
        # 2. Spectrogram comparison
        ax2 = plt.subplot(3, 2, 2)
        ax3 = plt.subplot(3, 2, 4)
        self._plot_spectrogram_comparison(clean, artifact, ax2, ax3)
        
        # 3. Spectrogram difference
        ax4 = plt.subplot(3, 2, 6)
        self._plot_spectrogram_difference(clean, artifact, ax4)
        
        # 4. Phase coherence
        ax5 = plt.subplot(3, 2, 3)
        self._plot_phase_coherence(clean, artifact, ax5)
        
        # 5. Time-frequency analysis specific to artifact type
        ax6 = plt.subplot(3, 2, 5)
        self._plot_artifact_specific_analysis(clean, artifact, artifact_type, ax6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
        return save_path
    
    def compare_artifact_types(self, 
                              reference: AudioSegment,
                              artifacts: Dict[ArtifactType, AudioSegment],
                              save_path: Optional[Path] = None) -> Path:
        """Compare different types of artifacts on the same audio"""
        if save_path is None:
            save_path = self.output_dir / "artifact_comparison.png"
        
        # Count artifacts for grid layout
        n_artifacts = len(artifacts)
        grid_cols = min(3, n_artifacts)
        grid_rows = (n_artifacts + grid_cols - 1) // grid_cols
        
        # Create large figure for comparison
        fig = plt.figure(figsize=(grid_cols * this.viz_config.fig_size[0]//3, 
                                 grid_rows * this.viz_config.fig_size[1]//2))
        
        # Plot each artifact in a separate subplot
        for i, (artifact_type, audio) in enumerate(artifacts.items()):
            ax = plt.subplot(grid_rows, grid_cols, i + 1)
            
            # Create spectrogram difference
            clean_stft = librosa.stft(reference.audio.mean(axis=0), 
                                     n_fft=self.viz_config.fft_size, 
                                     hop_length=self.viz_config.hop_length)
            artifact_stft = librosa.stft(audio.audio.mean(axis=0), 
                                       n_fft=self.viz_config.fft_size, 
                                       hop_length=self.viz_config.hop_length)
            
            # Calculate magnitude difference in dB
            clean_db = librosa.amplitude_to_db(np.abs(clean_stft), ref=np.max)
            artifact_db = librosa.amplitude_to_db(np.abs(artifact_stft), ref=np.max)
            diff_db = artifact_db - clean_db
            
            # Plot with consistent scale
            img = librosa.display.specshow(
                diff_db,
                sr=self.config.sample_rate,
                hop_length=self.viz_config.hop_length,
                x_axis='time',
                y_axis=self.viz_config.freq_scale,
                ax=ax,
                cmap=self.viz_config.cmap_diff,
                vmin=-20,
                vmax=20
            )
            ax.set_title(f"{artifact_type.name}")
            
            # Set frequency range if specified
            if self.viz_config.max_freq:
                ax.set_ylim(0, self.viz_config.max_freq)
        
        # Add colorbar
        plt.colorbar(img, ax=plt.gcf().axes, format="%+2.0f dB", shrink=0.6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
        return save_path
    
    def visualize_artifact_bands(self,
                                reference: AudioSegment,
                                artifact: AudioSegment,
                                frequency_bands: Dict[str, Tuple[float, float]],
                                save_path: Optional[Path] = None) -> Path:
        """Visualize artifact impact on different frequency bands"""
        if save_path is None:
            save_path = self.output_dir / "artifact_bands.png"
        
        # Create figure
        fig = plt.figure(figsize=self.viz_config.fig_size)
        
        # Calculate STFTs
        ref_stft = librosa.stft(reference.audio.mean(axis=0), 
                              n_fft=self.viz_config.fft_size, 
                              hop_length=self.viz_config.hop_length)
        art_stft = librosa.stft(artifact.audio.mean(axis=0), 
                               n_fft=self.viz_config.fft_size, 
                               hop_length=self.viz_config.hop_length)
        
        # Calculate frequency bins
        freq_bins = librosa.fft_frequencies(sr=self.config.sample_rate, 
                                          n_fft=self.viz_config.fft_size)
        
        # Track band metrics
        band_metrics = {}
        
        # Plot each frequency band
        n_bands = len(frequency_bands)
        grid_cols = min(3, n_bands)
        grid_rows = (n_bands + grid_cols - 1) // grid_cols
        
        for i, (band_name, (low_freq, high_freq)) in enumerate(frequency_bands.items()):
            ax = plt.subplot(grid_rows, grid_cols, i + 1)
            
            # Create mask for this frequency band
            band_mask = np.logical_and(freq_bins >= low_freq, freq_bins <= high_freq)
            
            # Extract band data
            ref_band = ref_stft[band_mask, :]
            art_band = art_stft[band_mask, :]
            
            # Calculate magnitude difference in dB
            ref_mag = np.abs(ref_band)
            art_mag = np.abs(art_band)
            
            # Calculate phase difference
            ref_phase = np.angle(ref_band)
            art_phase = np.angle(art_band)
            phase_diff = np.abs(ref_phase - art_phase)
            
            # Calculate metrics
            mag_ratio = np.mean(art_mag / (ref_mag + 1e-10))
            phase_coh = np.mean(np.cos(phase_diff))
            
            # Store metrics
            band_metrics[band_name] = {
                'magnitude_ratio': float(mag_ratio),
                'phase_coherence': float(phase_coh)
            }
            
            # Plot time-averaged spectrum
            freq_range = freq_bins[band_mask]
            ref_spec = librosa.amplitude_to_db(np.mean(ref_mag, axis=1), ref=np.max)
            art_spec = librosa.amplitude_to_db(np.mean(art_mag, axis=1), ref=np.max)
            
            ax.plot(freq_range, ref_spec, label='Reference', color='blue', alpha=0.8)
            ax.plot(freq_range, art_spec, label='Artifact', color='red', alpha=0.8)
            ax.fill_between(freq_range, ref_spec, art_spec, color='purple', alpha=0.3)
            
            ax.set_title(f"{band_name} ({low_freq}-{high_freq} Hz)")
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude (dB)')
            
            # Add metrics as text
            ax.text(0.05, 0.95, 
                   f"Mag ratio: {mag_ratio:.2f}\nPhase coh: {phase_coh:.2f}", 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
        # Save metrics
        with open(save_path.with_suffix('.json'), 'w') as f:
            json.dump(band_metrics, f, indent=2)
        
        return save_path
    
    def _plot_waveform_comparison(self, 
                                clean: AudioSegment, 
                                artifact: AudioSegment, 
                                ax: plt.Axes) -> None:
        """Plot waveform comparison"""
        # Convert to mono and normalize for comparison
        clean_mono = clean.audio.mean(axis=0) if clean.audio.ndim > 1 else clean.audio.flatten()
        artifact_mono = artifact.audio.mean(axis=0) if artifact.audio.ndim > 1 else artifact.audio.flatten()
        
        # Create time axis
        time = np.arange(len(clean_mono)) / clean.sample_rate
        
        # Plot
        ax.plot(time, clean_mono, label='Clean', color='blue', alpha=0.6)
        ax.plot(time, artifact_mono, label='Artifact', color='red', alpha=0.6)
        
        # Calculate difference
        difference = artifact_mono - clean_mono
        
        # Plot envelope of difference
        from scipy.signal import hilbert
        diff_envelope = np.abs(hilbert(difference))
        ax.plot(time, diff_envelope, label='Diff Envelope', color='green', alpha=0.4)
        
        ax.set_title('Waveform Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        
        # Add RMS difference as text
        rms_diff = np.sqrt(np.mean(difference**2))
        clean_rms = np.sqrt(np.mean(clean_mono**2))
        ax.text(0.05, 0.05, f"RMS diff: {rms_diff:.3f}\nRMS ratio: {rms_diff/clean_rms:.3f}", 
               transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_spectrogram_comparison(self, 
                                   clean: AudioSegment, 
                                   artifact: AudioSegment, 
                                   ax1: plt.Axes, 
                                   ax2: plt.Axes) -> None:
        """Plot spectrogram comparison"""
        # Create spectrograms
        clean_stft = librosa.stft(clean.audio.mean(axis=0), 
                                n_fft=self.viz_config.fft_size, 
                                hop_length=self.viz_config.hop_length)
        artifact_stft = librosa.stft(artifact.audio.mean(axis=0), 
                                   n_fft=self.viz_config.fft_size, 
                                   hop_length=self.viz_config.hop_length)
        
        # Convert to dB
        clean_db = librosa.amplitude_to_db(np.abs(clean_stft), ref=np.max)
        artifact_db = librosa.amplitude_to_db(np.abs(artifact_stft), ref=np.max)
        
        # Plot clean spectrogram
        librosa.display.specshow(
            clean_db,
            sr=self.config.sample_rate,
            hop_length=self.viz_config.hop_length,
            x_axis='time',
            y_axis=self.viz_config.freq_scale,
            ax=ax1,
            cmap=self.viz_config.cmap_magnitude,
            vmin=self.viz_config.db_range[0],
            vmax=self.viz_config.db_range[1]
        )
        
        ax1.set_title('Clean Spectrogram')
        plt.colorbar(ax=ax1, format='%+2.0f dB')
        
        # Plot artifact spectrogram
        librosa.display.specshow(
            artifact_db,
            sr=self.config.sample_rate,
            hop_length=self.viz_config.hop_length,
            x_axis='time',
            y_axis=self.viz_config.freq_scale,
            ax=ax2,
            cmap=self.viz_config.cmap_magnitude,
            vmin=self.viz_config.db_range[0],
            vmax=self.viz_config.db_range[1]
        )
        
        ax2.set_title('Artifact Spectrogram')
        plt.colorbar(ax=ax2, format='%+2.0f dB')
        
        # Set frequency range if specified
        if self.viz_config.max_freq:
            ax1.set_ylim(0, self.viz_config.max_freq)
            ax2.set_ylim(0, self.viz_config.max_freq)
    
    def _plot_spectrogram_difference(self, 
                                   clean: AudioSegment, 
                                   artifact: AudioSegment, 
                                   ax: plt.Axes) -> None:
        """Plot spectrogram difference"""
        # Create spectrograms
        clean_stft = librosa.stft(clean.audio.mean(axis=0), 
                                n_fft=self.viz_config.fft_size, 
                                hop_length=self.viz_config.hop_length)
        artifact_stft = librosa.stft(artifact.audio.mean(axis=0), 
                                   n_fft=self.viz_config.fft_size, 
                                   hop_length=self.viz_config.hop_length)
        
        # Convert to dB and calculate difference
        clean_db = librosa.amplitude_to_db(np.abs(clean_stft), ref=np.max)
        artifact_db = librosa.amplitude_to_db(np.abs(artifact_stft), ref=np.max)
        diff_db = artifact_db - clean_db
        
        # Plot difference
        librosa.display.specshow(
            diff_db,
            sr=self.config.sample_rate,
            hop_length=self.viz_config.hop_length,
            x_axis='time',
            y_axis=self.viz_config.freq_scale,
            ax=ax,
            cmap=self.viz_config.cmap_diff,
            vmin=-20,
            vmax=20
        )
        
        ax.set_title('Spectrogram Difference (Artifact - Clean)')
        plt.colorbar(ax=ax, format='%+2.0f dB')
        
        # Set frequency range if specified
        if self.viz_config.max_freq:
            ax.set_ylim(0, self.viz_config.max_freq)
    
    def _plot_phase_coherence(self, 
                             clean: AudioSegment, 
                             artifact: AudioSegment, 
                             ax: plt.Axes) -> None:
        """Plot phase coherence analysis"""
        # Create STFTs
        clean_stft = librosa.stft(clean.audio.mean(axis=0), 
                                n_fft=self.viz_config.fft_size, 
                                hop_length=self.viz_config.hop_length)
        artifact_stft = librosa.stft(artifact.audio.mean(axis=0), 
                                   n_fft=self.viz_config.fft_size, 
                                   hop_length=self.viz_config.hop_length)
        
        # Extract phase
        clean_phase = np.angle(clean_stft)
        artifact_phase = np.angle(artifact_stft)
        
        # Calculate phase difference
        phase_diff = np.abs(clean_phase - artifact_phase)
        phase_coh = np.cos(phase_diff)  # 1 = coherent, 0 = orthogonal, -1 = opposite
        
        # Plot phase coherence
        librosa.display.specshow(
            phase_coh,
            sr=self.config.sample_rate,
            hop_length=self.viz_config.hop_length,
            x_axis='time',
            y_axis=self.viz_config.freq_scale,
            ax=ax,
            cmap=self.viz_config.cmap_phase,
            vmin=-1,
            vmax=1
        )
        
        ax.set_title('Phase Coherence (Blue=Coherent, Red=Opposite)')
        plt.colorbar(ax=ax, format='%+.1f')
        
        # Set frequency range if specified
        if self.viz_config.max_freq:
            ax.set_ylim(0, self.viz_config.max_freq)
        
        # Add average phase coherence as text
        avg_coh = np.mean(phase_coh)
        ax.text(0.05, 0.05, f"Avg coherence: {avg_coh:.3f}", 
               transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_artifact_specific_analysis(self, 
                                        clean: AudioSegment, 
                                        artifact: AudioSegment, 
                                        artifact_type: ArtifactType, 
                                        ax: plt.Axes) -> None:
        """Plot analysis specific to the artifact type"""
        
        # Each artifact type gets its own specialized visualization
        if artifact_type == ArtifactType.PHASE_INCOHERENCE:
            self._plot_phase_coherence_histogram(clean, artifact, ax)
        
        elif artifact_type == ArtifactType.HIGH_FREQUENCY_RINGING:
            self._plot_high_frequency_analysis(clean, artifact, ax)
        
        elif artifact_type == ArtifactType.TEMPORAL_SMEARING:
            self._plot_temporal_smearing_analysis(clean, artifact, ax)
        
        elif artifact_type == ArtifactType.SPECTRAL_HOLES:
            self._plot_spectral_hole_analysis(clean, artifact, ax)
        
        elif artifact_type == ArtifactType.QUANTIZATION_NOISE:
            self._plot_quantization_analysis(clean, artifact, ax)
        
        elif artifact_type == ArtifactType.ALIASING:
            self._plot_aliasing_analysis(clean, artifact, ax)
        
        elif artifact_type == ArtifactType.BACKGROUND_BLEED:
            self._plot_background_bleed_analysis(clean, artifact, ax)
        
        elif artifact_type == ArtifactType.TRANSIENT_SUPPRESSION:
            self._plot_transient_analysis(clean, artifact, ax)
        
        else:
            ax.text(0.5, 0.5, f"No specific analysis for\n{artifact_type.name}", 
                   transform=ax.transAxes, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=12)
    
    def _plot_phase_coherence_histogram(self, 
                                       clean: AudioSegment, 
                                       artifact: AudioSegment, 
                                       ax: plt.Axes) -> None:
        """Plot phase coherence histogram for phase incoherence artifacts"""
        # Create STFTs
        clean_stft = librosa.stft(clean.audio.mean(axis=0), 
                                n_fft=self.viz_config.fft_size, 
                                hop_length=self.viz_config.hop_length)
        artifact_stft = librosa.stft(artifact.audio.mean(axis=0), 
                                   n_fft=self.viz_config.fft_size, 
                                   hop_length=self.viz_config.hop_length)
        
        # Extract phase
        clean_phase = np.angle(clean_stft)
        artifact_phase = np.angle(artifact_stft)
        
        # Calculate phase difference
        phase_diff = np.abs(clean_phase - artifact_phase)
        
        # Create histogram of phase differences
        hist_data = phase_diff.flatten()
        
        # Plot histogram
        sns.histplot(hist_data, bins=50, ax=ax, kde=True)
        ax.set_xlim(0, np.pi)
        ax.set_title('Phase Difference Histogram')
        ax.set_xlabel('Phase Difference (radians)')
        ax.set_ylabel('Count')
        
        # Add vertical lines for key values
        ax.axvline(x=np.mean(hist_data), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(hist_data):.2f}')
        ax.axvline(x=np.median(hist_data), color='green', linestyle='--', 
                  label=f'Median: {np.median(hist_data):.2f}')
        ax.axvline(x=np.pi/2, color='black', linestyle=':', 
                  label=r'$\pi/2$', alpha=0.5)
        ax.legend()
    
    def _plot_high_frequency_analysis(self, 
                                    clean: AudioSegment, 
                                    artifact: AudioSegment, 
                                    ax: plt.Axes) -> None:
        """Plot high frequency content analysis for ringing artifacts"""
        # Create spectrograms
        clean_stft = librosa.stft(clean.audio.mean(axis=0), 
                                n_fft=self.viz_config.fft_size, 
                                hop_length=self.viz_config.hop_length)
        artifact_stft = librosa.stft(artifact.audio.mean(axis=0), 
                                   n_fft=self.viz_config.fft_size, 
                                   hop_length=self.viz_config.hop_length)
        
        # Get magnitudes
        clean_mag = np.abs(clean_stft)
        artifact_mag = np.abs(artifact_stft)
        
        # Calculate frequency bins
        freq_bins = librosa.fft_frequencies(sr=self.config.sample_rate, 
                                          n_fft=self.viz_config.fft_size)
        
        # Define high frequency range (e.g., above 5000 Hz)
        high_freq_mask = freq_bins >= 5000
        
        # Calculate average spectrum
        clean_spectrum = np.mean(clean_mag, axis=1)
        artifact_spectrum = np.mean(artifact_mag, axis=1)
        
        # Plot spectrum comparison
        ax.plot(freq_bins, 
               librosa.amplitude_to_db(clean_spectrum, ref=np.max),
               label='Clean', 
               color='blue', 
               alpha=0.8)
        ax.plot(freq_bins, 
               librosa.amplitude_to_db(artifact_spectrum, ref=np.max),
               label='Artifact', 
               color='red', 
               alpha=0.8)
        
        # Shade high frequency region
        ax.fill_between(freq_bins[high_freq_mask], 
                       librosa.amplitude_to_db(clean_spectrum[high_freq_mask], ref=np.max),
                       librosa.amplitude_to_db(artifact_spectrum[high_freq_mask], ref=np.max),
                       color='purple', 
                       alpha=0.3)
        
        ax.set_title('High Frequency Content Comparison')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xscale('log')
        ax.set_xlim(100, self.config.sample_rate/2)
        ax.legend()
        
        # Add high frequency energy ratio as text
        high_clean_energy = np.sum(clean_spectrum[high_freq_mask]**2)
        high_artifact_energy = np.sum(artifact_spectrum[high_freq_mask]**2)
        energy_ratio = high_artifact_energy / (high_clean_energy + 1e-10)
        
        ax.text(0.05, 0.05, 
               f"High freq energy ratio: {energy_ratio:.2f}", 
               transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_temporal_smearing_analysis(self, 
                                       clean: AudioSegment, 
                                       artifact: AudioSegment, 
                                       ax: plt.Axes) -> None:
        """Plot temporal characteristics for smearing artifacts"""
        # Create envelope of signals
        from scipy.signal import hilbert
        
        clean_mono = clean.audio.mean(axis=0)
        artifact_mono = artifact.audio.mean(axis=0)
        
        clean_env = np.abs(hilbert(clean_mono))
        artifact_env = np.abs(hilbert(artifact_mono))
        
        # Calculate time axis
        time = np.arange(len(clean_mono)) / clean.sample_rate
        
        # Calculate envelope derivative to identify sharp transients
        clean_env_diff = np.diff(clean_env)
        artifact_env_diff = np.diff(artifact_env)
        
        # Normalize for comparison
        if np.max(np.abs(clean_env_diff)) > 0:
            clean_env_diff = clean_env_diff / np.max(np.abs(clean_env_diff))
        if np.max(np.abs(artifact_env_diff)) > 0:
            artifact_env_diff = artifact_env_diff / np.max(np.abs(artifact_env_diff))
        
        # Plot envelope derivatives
        ax.plot(time[:-1], clean_env_diff, label='Clean Env Derivative', color='blue', alpha=0.6)
        ax.plot(time[:-1], artifact_env_diff, label='Artifact Env Derivative', color='red', alpha=0.6)
        
        ax.set_title('Envelope Derivative (Transient Detection)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Envelope Derivative')
        ax.legend()
        
        # Calculate temporal smearing metric
        # (Sum of absolute differences in derivatives)
        smearing_metric = np.mean(np.abs(clean_env_diff - artifact_env_diff))
        
        ax.text(0.05, 0.05, 
               f"Temporal smearing: {smearing_metric:.3f}", 
               transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_spectral_hole_analysis(self, 
                                   clean: AudioSegment, 
                                   artifact: AudioSegment, 
                                   ax: plt.Axes) -> None:
        """Plot spectral hole analysis"""
        # Create spectrograms
        clean_stft = librosa.stft(clean.audio.mean(axis=0), 
                                n_fft=self.viz_config.fft_size, 
                                hop_length=self.viz_config.hop_length)
        artifact_stft = librosa.stft(artifact.audio.mean(axis=0), 
                                   n_fft=self.viz_config.fft_size, 
                                   hop_length=self.viz_config.hop_length)
        
        # Get magnitudes and calculate frequency bins
        clean_mag = np.abs(clean_stft)
        artifact_mag = np.abs(artifact_stft)
        freq_bins = librosa.fft_frequencies(sr=self.config.sample_rate, 
                                          n_fft=self.viz_config.fft_size)
        
        # Calculate average spectrum
        clean_spectrum = np.mean(clean_mag, axis=1)
        artifact_spectrum = np.mean(artifact_mag, axis=1)
        
        # Calculate spectral ratio
        spectral_ratio = artifact_spectrum / (clean_spectrum + 1e-10)
        
        # Find potential spectral holes
        # (Regions where artifact is significantly lower than clean)
        hole_threshold = 0.5  # Ratio threshold for considering a hole
        hole_mask = spectral_ratio < hole_threshold
        
        # Plot spectrum comparison
        ax.plot(freq_bins, 
               librosa.amplitude_to_db(clean_spectrum, ref=np.max),
               label='Clean', 
               color='blue', 
               alpha=0.8)
        ax.plot(freq_bins, 
               librosa.amplitude_to_db(artifact_spectrum, ref=np.max),
               label='Artifact', 
               color='red', 
               alpha=0.8)
        
        # Highlight spectral holes
        for i in range(len(hole_mask)):
            if hole_mask[i]:
                ax.axvspan(freq_bins[i-1] if i > 0 else 0, 
                          freq_bins[i+1] if i < len(freq_bins)-1 else freq_bins[i],
                          color='yellow', 
                          alpha=0.3)
        
        ax.set_title('Spectral Holes Analysis')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xscale('log')
        ax.set_xlim(50, self.config.sample_rate/2)
        ax.legend()
        
        # Count and report number of holes
        hole_groups = []
        current_hole = []
        for i, is_hole in enumerate(hole_mask):
            if is_hole:
                current_hole.append(i)
            elif current_hole:
                hole_groups.append(current_hole)
                current_hole = []
        
        if current_hole:
            hole_groups.append(current_hole)
        
        # Get frequency range of each hole group
        hole_ranges = []
        for group in hole_groups:
            if group:
                start_freq = freq_bins[group[0]]
                end_freq = freq_bins[group[-1]]
                hole_ranges.append((start_freq, end_freq))
        
        # Add hole statistics as text
        hole_text = f"Holes detected: {len(hole_groups)}"
        if hole_ranges:
            hole_text += "\nRanges (Hz): "
            hole_text += ", ".join([f"{int(start)}-{int(end)}" for start, end in hole_ranges[:3]])
            if len(hole_ranges) > 3:
                hole_text += f", +{len(hole_ranges)-3} more"
        
        ax.text(0.05, 0.05, 
               hole_text, 
               transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_quantization_analysis(self, 
                                  clean: AudioSegment, 
                                  artifact: AudioSegment, 
                                  ax: plt.Axes) -> None:
        """Plot quantization noise analysis"""
        # Get audio samples and calculate difference
        clean_mono = clean.audio.mean(axis=0)
        artifact_mono = artifact.audio.mean(axis=0)
        difference = artifact_mono - clean_mono
        
        # Use histogram to visualize quantization effects
        ax.hist(clean_mono, bins=100, alpha=0.5, label='Clean', color='blue')
        ax.hist(artifact_mono, bins=100, alpha=0.5, label='Artifact', color='red')
        
        ax.set_title('Amplitude Histogram (Quantization)')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Count')
        ax.legend()
        
        # Calculate effective bit depth
        from scipy.stats import iqr
        clean_range = iqr(clean_mono)
        noise_range = iqr(difference)
        
        if noise_range > 0 and clean_range > 0:
            snr = clean_range / noise_range
            effective_bits = np.log2(snr)
        else:
            effective_bits = float('inf')
        
        # Add estimated bit depth as text
        ax.text(0.05, 0.95, 
               f"Est. effective bits: {effective_bits:.1f}", 
               transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_aliasing_analysis(self, 
                              clean: AudioSegment, 
                              artifact: AudioSegment, 
                              ax: plt.Axes) -> None:
        """Plot aliasing analysis"""
        # Create spectrograms
        clean_stft = librosa.stft(clean.audio.mean(axis=0), 
                                n_fft=self.viz_config.fft_size, 
                                hop_length=self.viz_config.hop_length)
        artifact_stft = librosa.stft(artifact.audio.mean(axis=0), 
                                   n_fft=self.viz_config.fft_size, 
                                   hop_length=self.viz_config.hop_length)
        
        # Get magnitudes
        clean_mag = np.abs(clean_stft)
        artifact_mag = np.abs(artifact_stft)
        
        # Calculate frequencies
        freq_bins = librosa.fft_frequencies(sr=self.config.sample_rate, 
                                          n_fft=self.viz_config.fft_size)
        
        # Calculate average magnitude difference
        clean_spectrum = np.mean(clean_mag, axis=1)
        artifact_spectrum = np.mean(artifact_mag, axis=1)
        
        # Plot spectrum with logarithmic x-axis
        ax.plot(freq_bins, 
               librosa.amplitude_to_db(clean_spectrum, ref=np.max),
               label='Clean', 
               color='blue', 
               alpha=0.8)
        ax.plot(freq_bins, 
               librosa.amplitude_to_db(artifact_spectrum, ref=np.max),
               label='Artifact', 
               color='red', 
               alpha=0.8)
        
        # Calculate the difference, highlighting where artifact > clean
        # (potential aliases)
        diff_spectrum = artifact_spectrum - clean_spectrum
        
        # Find regions where artifact has higher energy (possible aliases)
        alias_threshold = 3  # dB threshold
        alias_mask = librosa.amplitude_to_db(artifact_spectrum, ref=np.max) > \
                   (librosa.amplitude_to_db(clean_spectrum, ref=np.max) + alias_threshold)
        
        # Highlight potential aliasing regions
        for i in range(len(alias_mask)):
            if alias_mask[i]:
                ax.axvspan(freq_bins[i-1] if i > 0 else 0, 
                          freq_bins[i+1] if i < len(freq_bins)-1 else freq_bins[i],
                          color='red', 
                          alpha=0.2)
        
        ax.set_title('Potential Aliasing Analysis')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xscale('log')
        ax.set_xlim(50, self.config.sample_rate/2)
        ax.legend()
        
        # Calculate aliasing metric - energy in frequencies that appear in artifact but not in clean
        positive_diff_energy = np.sum(diff_spectrum[diff_spectrum > 0]**2)
        total_clean_energy = np.sum(clean_spectrum**2)
        aliasing_ratio = positive_diff_energy / (total_clean_energy + 1e-10)
        
        # Count continuous aliasing regions
        alias_groups = []
        current_group = []
        for i, is_alias in enumerate(alias_mask):
            if is_alias:
                current_group.append(i)
            elif current_group:
                alias_groups.append(current_group)
                current_group = []
        
        if current_group:
            alias_groups.append(current_group)
        
        ax.text(0.05, 0.05, 
               f"Aliasing ratio: {aliasing_ratio:.4f}\nAlias regions: {len(alias_groups)}", 
               transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_background_bleed_analysis(self, 
                                      clean: AudioSegment, 
                                      artifact: AudioSegment, 
                                      ax: plt.Axes) -> None:
        """Plot analysis for background bleed artifacts"""
        # Create spectrograms
        clean_stft = librosa.stft(clean.audio.mean(axis=0), 
                                n_fft=self.viz_config.fft_size, 
                                hop_length=self.viz_config.hop_length)
        artifact_stft = librosa.stft(artifact.audio.mean(axis=0), 
                                   n_fft=self.viz_config.fft_size, 
                                   hop_length=self.viz_config.hop_length)
        
        # Calculate spectrogram difference
        clean_spec = np.abs(clean_stft)
        artifact_spec = np.abs(artifact_stft)
        
        # Convert to dB
        clean_db = librosa.amplitude_to_db(clean_spec, ref=np.max)
        artifact_db = librosa.amplitude_to_db(artifact_spec, ref=np.max)
        
        # Calculate time-frequency mask indicating potential bleed
        # Areas where artifact has more energy than clean
        bleed_mask = artifact_spec > clean_spec * 1.1  # 10% more energy
        
        # Calculate energy concentration in bleed regions over time
        bleed_energy = np.sum(artifact_spec * bleed_mask, axis=0)
        total_energy = np.sum(artifact_spec, axis=0)
        bleed_ratio = bleed_energy / (total_energy + 1e-10)
        
        # Create time axis for plotting
        frames = bleed_ratio.shape[0]
        time = librosa.frames_to_time(
            np.arange(frames),
            sr=self.config.sample_rate,
            hop_length=self.viz_config.hop_length
        )
        
        # Plot bleed energy ratio over time
        ax.plot(time, bleed_ratio, color='purple', linewidth=2)
        ax.set_ylim(0, min(1.0, np.max(bleed_ratio) * 1.2))
        
        ax.set_title('Background Bleed Analysis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Bleed Energy Ratio')
        
        # Add overall metrics
        avg_bleed = np.mean(bleed_ratio)
        max_bleed = np.max(bleed_ratio)
        
        ax.axhline(y=avg_bleed, color='red', linestyle='--', 
                  label=f'Avg: {avg_bleed:.3f}')
        
        ax.text(0.05, 0.95, 
               f"Avg bleed: {avg_bleed:.3f}\nMax bleed: {max_bleed:.3f}", 
               transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend()
    
    def _plot_transient_analysis(self, 
                                clean: AudioSegment, 
                                artifact: AudioSegment, 
                                ax: plt.Axes) -> None:
        """Plot transient analysis for transient suppression artifacts"""
        # Compute novelty curve for both signals
        # (indicates where there are significant changes - transients)
        from librosa.onset import onset_strength
        
        clean_mono = clean.audio.mean(axis=0)
        artifact_mono = artifact.audio.mean(axis=0)
        
        clean_onset = onset_strength(
            y=clean_mono, 
            sr=clean.sample_rate,
            hop_length=self.viz_config.hop_length
        )
        
        artifact_onset = onset_strength(
            y=artifact_mono, 
            sr=artifact.sample_rate,
            hop_length=self.viz_config.hop_length
        )
        
        # Create time axis
        frames = len(clean_onset)
        time = librosa.frames_to_time(
            np.arange(frames),
            sr=self.config.sample_rate,
            hop_length=self.viz_config.hop_length
        )
        
        # Normalize for comparison
        if np.max(clean_onset) > 0:
            clean_onset = clean_onset / np.max(clean_onset)
        if np.max(artifact_onset) > 0:
            artifact_onset = artifact_onset / np.max(artifact_onset)
        
        # Plot onset strength
        ax.plot(time, clean_onset, label='Clean Onsets', color='blue', alpha=0.8)
        ax.plot(time, artifact_onset, label='Artifact Onsets', color='red', alpha=0.8)
        
        # Highlight differences
        ax.fill_between(time, 
                       clean_onset, 
                       artifact_onset,
                       where=(clean_onset > artifact_onset),
                       color='red', 
                       alpha=0.3, 
                       label='Suppressed')
        
        ax.set_title('Transient Analysis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Onset Strength')
        ax.legend()
        
        # Calculate suppression metrics
        transient_threshold = 0.2  # Threshold for detecting significant transients
        clean_transients = clean_onset > transient_threshold
        artifact_transients = artifact_onset > transient_threshold
        
        # Count preserved vs. suppressed transients
        preserved = np.sum(np.logical_and(clean_transients, artifact_transients))
        suppressed = np.sum(np.logical_and(clean_transients, np.logical_not(artifact_transients)))
        
        if np.sum(clean_transients) > 0:
            preservation_ratio = preserved / np.sum(clean_transients)
        else:
            preservation_ratio = 1.0
        
        # Calculate average strength reduction
        strength_reduction = np.mean((clean_onset - artifact_onset)[clean_transients])
        
        ax.text(0.05, 0.95, 
               f"Preserved: {preserved}/{np.sum(clean_transients)} ({preservation_ratio:.2f})\nAvg strength reduction: {strength_reduction:.3f}", 
               transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))