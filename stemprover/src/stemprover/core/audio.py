from dataclasses import dataclass
import numpy as np
import librosa
from typing import Optional

@dataclass
class AudioSegment:
    """Data class for audio segments with their metadata"""
    audio: np.ndarray
    sample_rate: int = 44100
    start_time: float = 0.0
    duration: float = 0.0

    @property
    def is_stereo(self) -> bool:
        stereo = len(self.audio.shape) == 2 and (
            self.audio.shape[0] == 2 or self.audio.shape[1] == 2
        )
        print(f"is_stereo check - shape: {self.audio.shape}, result: {stereo}")
        return stereo

    @property
    def is_mono(self) -> bool:
        mono = len(self.audio.shape) == 1 or (
            len(self.audio.shape) == 2 and (
                self.audio.shape[0] == 1 or self.audio.shape[1] == 1
            )
        )
        print(f"is_mono check - shape: {self.audio.shape}, result: {mono}")
        return mono

    def to_mono(self) -> 'AudioSegment':
        """Convert to mono if stereo"""
        print(f"to_mono - input shape: {self.audio.shape}")
        
        if self.is_mono:
            print("Already mono, returning as is")
            return self
            
        # Handle different stereo formats
        if len(self.audio.shape) == 2:
            if self.audio.shape[0] == 2:
                # (2, samples) format
                mono_audio = librosa.to_mono(self.audio)
            elif self.audio.shape[1] == 2:
                # (samples, 2) format
                mono_audio = librosa.to_mono(self.audio.T)
            else:
                raise ValueError(f"Unexpected audio shape: {self.audio.shape}")
        else:
            raise ValueError(f"Cannot convert shape {self.audio.shape} to mono")
            
        print(f"to_mono - output shape: {mono_audio.shape}")
        
        return AudioSegment(
            audio=mono_audio,
            sample_rate=self.sample_rate,
            start_time=self.start_time,
            duration=self.duration
        )

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds based on audio shape and sample rate"""
        if self.is_stereo:
            return self.audio.shape[1] / self.sample_rate
        return len(self.audio) / self.sample_rate

    def slice(self, start_sec: float, end_sec: float) -> 'AudioSegment':
        """Return a new AudioSegment sliced to the given start and end times"""
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)

        if self.is_stereo:
            sliced_audio = self.audio[:, start_sample:end_sample]
        else:
            sliced_audio = self.audio[start_sample:end_sample]

        return AudioSegment(
            audio=sliced_audio,
            sample_rate=self.sample_rate,
            start_time=self.start_time + start_sec,
            duration=end_sec - start_sec
        )
