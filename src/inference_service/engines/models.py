"""
Data models for the inference layer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Chunk:
    """Represents a text chunk to be synthesized."""

    chunk_id: str
    """Unique identifier for the chunk."""

    text: str
    """The normalized text content to synthesize."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata (language, speaker_id, voice_style, etc.)."""


@dataclass
class AudioResult:
    """Result of inference containing audio and metadata."""

    chunk_id: str
    """Unique identifier for the chunk (matches input chunk)."""

    audio_path: str
    """Path to WAV file containing synthesized audio."""

    duration: float
    """Audio duration in seconds."""

    sample_rate: int
    """Sample rate in Hz (typically 22050 or 44100)."""

    inference_latency: float
    """Time taken for inference in seconds."""

    status: str
    """Status: 'success', 'error', etc."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata (model_version, speaker_id, etc.)."""
