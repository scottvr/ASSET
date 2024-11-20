"""Stemproved - audio stem separation enhancement tools"""
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .core import AudioSegment, SeparationResult
from .separation import VocalSeparator, SpleeterSeparator