from pathlib import Path
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Dict, Optional
from datetime import datetime

from ..types import (
    AudioArray, SpectrogramArray, FrequencyBands,
    DEFAULT_FREQUENCY_BANDS
)
from ..utils import (
    create_spectrogram, get_frequency_bins, get_band_mask,
    rms, db_scale, to_mono
)
from ..core.audio import AudioSegment
from ..core.types import ProcessingConfig
from .phase import PhaseAnalyzer

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
        self.phase_analyzer = PhaseAnalyzer(self.output_dir)
        self.normalization_params = {}

    def analyze(self, clean: AudioSegment, separated: AudioSegment) -> Path:
        """Perform spectral analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = self.output_dir / f"analysis_{timestamp}"
        analysis_dir.mkdir(exist_ok=True)

        clean_spec = self.create_phase_spectrogram(clean.audio, clean.sample_rate)
        sep_spec = self.create_phase_spectrogram(separated.audio, separated.sample_rate)

        self._save_comparison(
            clean_spec,
            sep_spec,
            analysis_dir / "spectrogram_comparison.png"
        )

        diff_analysis = self._analyze_differences(clean_spec, sep_spec)
        self._save_analysis(diff_analysis, analysis_dir / "analysis.json")

        return analysis_dir

    def create_phase_spectrogram(self, audio: AudioArray, sr: int) -> SpectrogramArray:
        """Create spectrogram with phase preservation for analysis"""
        audio_mono = to_mono(audio)
        return create_spectrogram(
            audio_mono,
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
        difference = db_scale(spec1, ref=np.max) - db_scale(spec2, ref=np.max)
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
            db_scale(spec, ref=np.max),
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
            mag_diff = np.abs(sep_band - clean_band).mean()
            phase_diff = np.abs(np.angle(clean_band) - np.angle(sep_band))

            analysis[band_name] = {
                "magnitude_difference": float(mag_diff),
                "phase_coherence": float(np.mean(np.cos(phase_diff))),
                "energy_ratio": rms(sep_band) / rms(clean_band) if rms(clean_band) > 0 else 0
            }

        # Overall metrics
        overall_phase_diff = np.abs(np.angle(clean_spec) - np.angle(separated_spec))
        analysis["overall"] = {
            "total_magnitude_difference": float(np.abs(separated_spec - clean_spec).mean()),
            "average_phase_coherence": float(np.mean(np.cos(overall_phase_diff))),
            "total_energy_ratio": rms(separated_spec) / rms(clean_spec)
        }

        return analysis

    def _save_analysis(self, analysis: Dict, path: Path) -> None:
        """Save analysis results as JSON"""
        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)

    def analyze_frequency_distribution(self, audio_segment: AudioSegment, preserve_phase: bool = False) -> Dict:
        """Analyzes the energy distribution across frequency bands for a single audio segment."""
        if self.output_dir is None:
            # Create a temporary directory if no output dir is specified
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                self.output_dir = Path(tmpdir)
                return self._analyze_frequency_distribution(audio_segment, preserve_phase)
        else:
            return self._analyze_frequency_distribution(audio_segment, preserve_phase)

    def _analyze_frequency_distribution(self, audio_segment: AudioSegment, preserve_phase: bool) -> Dict:
        """Helper for frequency distribution analysis."""
        import time
        start_time = time.time()

        spec = self.create_phase_spectrogram(audio_segment.audio, audio_segment.sample_rate)
        freq_bins = get_frequency_bins(sr=audio_segment.sample_rate, n_fft=self.config.n_fft)

        energy_dist = {}
        total_energy = np.sum(np.abs(spec))

        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_mask = get_band_mask(freq_bins, low_freq, high_freq)
            band_energy = np.sum(np.abs(spec[band_mask]))
            energy_dist[band_name] = (band_energy / total_energy) * 100 if total_energy > 0 else 0

        results = {
            "energy_distribution": energy_dist,
            "processing_time": time.time() - start_time
        }

        if preserve_phase:
            # This is a placeholder for a more meaningful phase coherence calculation
            # on a single spectrogram, which is not well-defined.
            # We'll calculate coherence against a sine wave of the dominant frequency.
            dominant_freq_idx = np.argmax(np.mean(np.abs(spec), axis=1))
            dominant_freq = freq_bins[dominant_freq_idx]
            t = np.arange(len(audio_segment.audio)) / audio_segment.sample_rate
            sine_wave = np.sin(2 * np.pi * dominant_freq * t)
            sine_spec = self.create_phase_spectrogram(sine_wave, audio_segment.sample_rate)
            # Ensure specs have same shape
            min_shape = min(spec.shape[1], sine_spec.shape[1])
            phase_diff = np.abs(np.angle(spec[:,:min_shape]) - np.angle(sine_spec[:,:min_shape]))
            results["phase_coherence"] = float(np.mean(np.cos(phase_diff)))

        return results

    def calculate_band_isolation(self, audio_segment: AudioSegment) -> float:
        """
        Calculates a score representing how well energy is isolated within the defined frequency bands.
        A higher score means more energy is concentrated within the bands and less is spread between them.
        """
        spec = self.create_phase_spectrogram(audio_segment.audio, audio_segment.sample_rate)
        freq_bins = get_frequency_bins(sr=audio_segment.sample_rate, n_fft=self.config.n_fft)

        total_energy = np.sum(np.abs(spec))
        energy_in_bands = 0

        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_mask = get_band_mask(freq_bins, low_freq, high_freq)
            energy_in_bands += np.sum(np.abs(spec[band_mask]))

        return energy_in_bands / total_energy if total_energy > 0 else 0