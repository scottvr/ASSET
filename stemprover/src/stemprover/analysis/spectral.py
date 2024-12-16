from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict
from datetime import datetime

from ...common.types import (
    AudioArray, SpectrogramArray, FrequencyBands,
    DEFAULT_FREQUENCY_BANDS
)
from ...common.audio_utils import (
    create_spectrogram, get_frequency_bins, get_band_mask
)
from ...common.math_utils import (
    magnitude, angle, phase_difference, phase_coherence,
    rms, db_scale
)
from ...core.audio import AudioSegment
from ...core.types import ProcessingConfig

class SpectralAnalyzer:
    """Spectral analysis and visualization with standardized types"""
    
    def __init__(self, 
                 output_dir: Path, 
                 config: Optional[ProcessingConfig] = None,
                 frequency_bands: Optional[FrequencyBands] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ProcessingConfig()
        self.frequency_bands = frequency_bands or DEFAULT_FREQUENCY_BANDS
        self.normalization_params = {}
        
    def analyze(self, clean: AudioSegment, separated: AudioSegment) -> Path:
        """Perform spectral analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = self.output_dir / f"analysis_{timestamp}"
        analysis_dir.mkdir(exist_ok=True)
        
        clean_spec = self._create_spectrogram(clean.audio, clean.sample_rate)
        sep_spec = self._create_spectrogram(separated.audio, separated.sample_rate)
        
        self._save_comparison(
            clean_spec, 
            sep_spec,
            analysis_dir / "spectrogram_comparison.png"
        )
        
        diff_analysis = self._analyze_differences(clean_spec, sep_spec)
        self._save_analysis(diff_analysis, analysis_dir / "analysis.json")
        
        return analysis_dir

    def _create_spectrogram(self, audio: AudioArray, sr: int) -> SpectrogramArray:
        """Create spectrogram with phase preservation"""
        return create_spectrogram(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
    def _save_comparison(self, 
                        spec1: SpectrogramArray, 
                        spec2: SpectrogramArray, 
                        path: Path) -> None:
        """Save visual comparison of spectrograms"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
        
        self._plot_spectrogram(
            spec1, 
            ax1, 
            "Clean Vocal Spectrogram",
            self.config.sample_rate
        )
        
        self._plot_spectrogram(
            spec2, 
            ax2, 
            "Separated Vocal Spectrogram",
            self.config.sample_rate
        )
        
        # Plot difference
        difference = db_scale(spec2) - db_scale(spec1)
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
                         spec: SpectrogramArray,
                         ax: plt.Axes,
                         title: str,
                         sr: int,
                         cmap: str = 'magma') -> None:
        """Helper to plot a single spectrogram"""
        librosa.display.specshow(
            db_scale(spec),
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
                           clean_spec: SpectrogramArray, 
                           separated_spec: SpectrogramArray) -> Dict:
        """Analyze spectral differences between clean and separated audio"""
        freq_bins = get_frequency_bins(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft
        )
        
        analysis = {}
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_mask = get_band_mask(freq_bins, low_freq, high_freq)
            
            # Extract band data
            clean_band = clean_spec[band_mask]
            sep_band = separated_spec[band_mask]
            
            # Compute metrics
            mag_diff = magnitude(sep_band - clean_band).mean()
            phase_diff = phase_difference(clean_band, sep_band)
            
            analysis[band_name] = {
                "magnitude_difference": float(mag_diff),
                "phase_coherence": phase_coherence(phase_diff),
                "energy_ratio": rms(sep_band) / rms(clean_band) if rms(clean_band) > 0 else 0
            }
        
        # Overall metrics
        overall_phase_diff = phase_difference(clean_spec, separated_spec)
        analysis["overall"] = {
            "total_magnitude_difference": float(magnitude(separated_spec - clean_spec).mean()),
            "average_phase_coherence": phase_coherence(overall_phase_diff),
            "total_energy_ratio": rms(separated_spec) / rms(clean_spec)
        }
        
        return analysis
    
    def _save_analysis(self, analysis: Dict, path: Path) -> None:
        """Save analysis results as JSON"""
        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)
