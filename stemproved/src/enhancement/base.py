from abc import ABC, abstractmethod
from typing import Optional
from ...core.audio import AudioSegment
from ...core.types import ProcessingConfig

class EnhancementProcessor(ABC):
    """Base class for audio enhancement processors"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    @abstractmethod
    def enhance(self, audio: AudioSegment) -> AudioSegment:
        """Enhance audio segment"""
        pass
    
    @abstractmethod
    def validate(self, audio: AudioSegment) -> dict:
        """Validate enhancement results"""
        pass