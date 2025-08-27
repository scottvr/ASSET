from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
from typing import Tuple
from ..core.audio import AudioSegment

def load_audio(path: Path, sr: int = 44100, mono: bool = False) -> torch.Tensor:
    """Load audio file and return as torch tensor."""
    audio, _ = load_audio_file(path, sr, mono)
    return torch.from_numpy(audio).float()

def load_audio_file(path: Path, sr: int = 44100, mono: bool = False) -> Tuple[np.ndarray, int]:
    """Load audio file with error handling and validation"""
    try:
        audio, file_sr = librosa.load(str(path), sr=sr, mono=mono)
        return audio, file_sr
    except Exception as e:
        raise RuntimeError(f"Error loading audio file {path}: {str(e)}")

def save_audio_file(audio: AudioSegment, path: Path) -> None:
    """Save audio file with proper format handling"""
    try:
        # Handle different array shapes
        audio_to_save = audio.audio
        if audio.is_stereo:
            # Convert from (2, samples) to (samples, 2)
            audio_to_save = audio.audio.T
            
        sf.write(path, audio_to_save, audio.sample_rate)
    except Exception as e:
        raise RuntimeError(f"Error saving audio file {path}: {str(e)}")
