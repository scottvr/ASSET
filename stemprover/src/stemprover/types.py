from typing import Union, List, Dict, Optional, Tuple, Any
import numpy as np
import torch
import librosa


# Common type aliases
AudioArray = np.ndarray
SpectrogramArray = np.ndarray
TensorType = Union[torch.Tensor, np.ndarray]
FrequencyBand = Tuple[float, float]
FrequencyBands = Dict[str, FrequencyBand]


# Common constants
DEFAULT_FREQUENCY_BANDS: FrequencyBands = {
    "sub_bass": (20, 60),
    "bass": (60, 250),
    "low_mid": (250, 500),
    "mid": (500, 2000),
    "high_mid": (2000, 4000),
    "presence": (4000, 6000),
    "brilliance": (6000, 20000)
}
