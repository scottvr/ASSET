from pathlib import Path
from typing import List, Tuple, Dict, Generator
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import numpy as np

from common.types import AudioArray
from common.audio_utils import create_spectrogram
from core.audio import AudioSegment
from core.types import ProcessingConfig

"""
Key features:
1. Configurable segment length and overlap
2. Filters out segments with insufficient vocal content
3. Creates multiple backing track combinations
4. Generates spectrograms for training
5. Preserves timing information for potential time-based analysis
"""

@dataclass
class SegmentConfig:
    """Configuration for segment generation"""
    segment_length: float = 5.0
    overlap: float = 2.5
    min_vocal_energy: float = 0.1  # Threshold for keeping vocal segments
    sample_rate: int = 44100
    
    @property
    def segment_samples(self) -> int:
        return int(self.segment_length * self.sample_rate)
    
    @property
    def hop_samples(self) -> int:
        return int((self.segment_length - self.overlap) * self.sample_rate)

class TrainingSegmentGenerator:
    """Generates training segments from multitrack sources"""
    
    def __init__(self, 
                 config: SegmentConfig,
                 processing_config: ProcessingConfig):
        self.config = config
        self.processing_config = processing_config
        
    def generate_segments(self,
                         vocal_path: Path,
                         backing_paths: List[Path]) -> Generator[Dict, None, None]:
        """Generate training segments from a song's tracks"""
        
        # Load vocal track
        vocal = AudioSegment.from_file(vocal_path)
        
        # Load backing tracks
        backing_tracks = [AudioSegment.from_file(path) for path in backing_paths]
        
        # Generate segments
        for start_idx in range(0, len(vocal.audio), self.config.hop_samples):
            end_idx = start_idx + self.config.segment_samples
            
            if end_idx > len(vocal.audio):
                break
                
            # Extract vocal segment
            vocal_segment = vocal.audio[start_idx:end_idx]
            
            # Check if segment has sufficient vocal content
            if self._has_vocal_content(vocal_segment):
                # Create different backing track combinations
                backing_combinations = self._create_backing_combinations(
                    backing_tracks,
                    start_idx,
                    end_idx
                )
                
                # Generate training pairs for each combination
                for mix_name, backing_mix in backing_combinations.items():
                    # Create mixed audio
                    mixed = vocal_segment + backing_mix
                    
                    # Create spectrograms
                    vocal_spec = create_spectrogram(
                        vocal_segment,
                        n_fft=self.processing_config.n_fft,
                        hop_length=self.processing_config.hop_length
                    )
                    
                    mixed_spec = create_spectrogram(
                        mixed,
                        n_fft=self.processing_config.n_fft,
                        hop_length=self.processing_config.hop_length
                    )
                    
                    yield {
                        'clean': vocal_spec,
                        'mixed': mixed_spec,
                        'source_audio': mixed,
                        'target_audio': vocal_segment,
                        'mix_type': mix_name,
                        'start_time': start_idx / self.config.sample_rate,
                        'duration': self.config.segment_length
                    }
    
    def _has_vocal_content(self, segment: AudioArray) -> bool:
        """Check if segment has sufficient vocal energy"""
        rms = np.sqrt(np.mean(segment ** 2))
        return rms > self.config.min_vocal_energy
    
    def _create_backing_combinations(self,
                                   backing_tracks: List[AudioSegment],
                                   start_idx: int,
                                   end_idx: int) -> Dict[str, AudioArray]:
        """Create different combinations of backing tracks"""
        segments = [track.audio[start_idx:end_idx] for track in backing_tracks]
        
        # Create standard combinations
        combinations = {
            'full_band': sum(segments),
            'no_drums': sum(segments[:-1]) if len(segments) > 1 else segments[0],
        }
        
        # Add pairs of instruments if we have enough tracks
        if len(segments) >= 2:
            combinations['guitar_bass'] = segments[0] + segments[1]
            
        if len(segments) >= 3:
            combinations['rhythm_section'] = segments[1] + segments[2]  # bass + drums
            
        return combinations

class TrainingDataset(Dataset):
    """Dataset for training with segmented audio"""
    
    def __init__(self,
                 song_paths: List[Dict[str, Path]],
                 segment_config: SegmentConfig,
                 processing_config: ProcessingConfig):
        self.generator = TrainingSegmentGenerator(segment_config, processing_config)
        self.segments = []
        
        # Pre-generate all segments
        for song in song_paths:
            segments = list(self.generator.generate_segments(
                song['vocal'],
                [song['guitar'], song['bass'], song['drums']]
            ))
            self.segments.extend(segments)
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.segments[idx]

"""
Usage would look like:

# Configuration
segment_config = SegmentConfig(
    segment_length=5.0,
    overlap=2.5,
    min_vocal_energy=0.1
)

processing_config = ProcessingConfig(
    sample_rate=44100,
    n_fft=2048,
    hop_length=512
)

# Example song paths
songs = [
    {
        'vocal': Path('song1/vocals.wav'),
        'guitar': Path('song1/guitar.wav'),
        'bass': Path('song1/bass.wav'),
        'drums': Path('song1/drums.wav')
    },
    # ... more songs
]

# Create dataset
dataset = TrainingDataset(songs, segment_config, processing_config)

# Use with DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)
"""