import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from ...core.audio import AudioSegment


class SpectralAnalyzer:
    """Spectral analysis and visualization"""
    
    def __init__(self, output_dir: Path, config: Optional[ProcessingConfig] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ProcessingConfig()
        self.normalization_params = {}
        
    def _save_comparison(self, 
                        clean_spec: np.ndarray, 
                        separated_spec: np.ndarray, 
                        path: Path):
        """Save visual comparison of spectrograms"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
        
        # Plot clean spectrogram
        self._plot_spectrogram(
            clean_spec, 
            ax1, 
            "Clean Vocal Spectrogram",
            self.config.sample_rate
        )
        
        # Plot separated spectrogram
        self._plot_spectrogram(
            separated_spec, 
            ax2, 
            "Separated Vocal Spectrogram",
            self.config.sample_rate
        )
        
        # Plot difference
        difference = librosa.amplitude_to_db(
            np.abs(separated_spec) - np.abs(clean_spec),
            ref=np.max
        )
        self._plot_spectrogram(
            difference,
            ax3,
            "Difference (Separated - Clean)",
            self.config.sample_rate,
            cmap='RdBu'
        )
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        
    def _plot_spectrogram(self,
                         spec: np.ndarray,
                         ax: plt.Axes,
                         title: str,
                         sr: int,
                         cmap: str = 'magma'):
        """Helper to plot a single spectrogram"""
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(spec), ref=np.max),
            y_axis='mel',
            x_axis='time',
            sr=sr,
            hop_length=self.config.hop_length,
            ax=ax,
            cmap=cmap
        )
        ax.set_title(title, color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Frequency (Hz)', color='white')
        ax.tick_params(colors='white')
        plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    
    def _analyze_differences(self, 
                           clean_spec: np.ndarray, 
                           separated_spec: np.ndarray) -> Dict:
        """Analyze spectral differences between clean and separated audio"""
        # Get magnitude spectrograms
        clean_mag = np.abs(clean_spec)
        sep_mag = np.abs(separated_spec)
        
        # Get phase information
        clean_phase = np.angle(clean_spec)
        sep_phase = np.angle(separated_spec)
        
        # Calculate frequency bands
        freq_bins = librosa.fft_frequencies(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft
        )
        
        bands = {
            "sub_bass": (20, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "high_mid": (2000, 4000),
            "presence": (4000, 6000),
            "brilliance": (6000, 20000)
        }
        
        analysis = {}
        
        # Analyze each frequency band
        for band_name, (low_freq, high_freq) in bands.items():
            # Get band indices
            band_mask = (freq_bins >= low_freq) & (freq_bins <= high_freq)
            
            # Magnitude difference in band
            mag_diff = np.mean(np.abs(sep_mag[band_mask] - clean_mag[band_mask]))
            
            # Phase coherence in band
            phase_diff = np.abs(sep_phase[band_mask] - clean_phase[band_mask])
            phase_coherence = np.mean(np.cos(phase_diff))
            
            # RMS energy ratio
            clean_rms = np.sqrt(np.mean(clean_mag[band_mask] ** 2))
            sep_rms = np.sqrt(np.mean(sep_mag[band_mask] ** 2))
            energy_ratio = sep_rms / clean_rms if clean_rms > 0 else 0
            
            analysis[band_name] = {
                "magnitude_difference": float(mag_diff),
                "phase_coherence": float(phase_coherence),
                "energy_ratio": float(energy_ratio)
            }
        
        # Overall statistics
        analysis["overall"] = {
            "total_magnitude_difference": float(np.mean(np.abs(sep_mag - clean_mag))),
            "average_phase_coherence": float(np.mean(np.cos(np.abs(sep_phase - clean_phase)))),
            "total_energy_ratio": float(np.sum(sep_mag) / np.sum(clean_mag))
        }
        
        return analysis
    
    def _save_analysis(self, analysis: Dict, path: Path):
        """Save analysis results as JSON"""
        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)
