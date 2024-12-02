from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Path
from .core.types import ProcessingConfig, SegmentConfig
from .preparation.segments import TrainingSegmentGenerator
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
