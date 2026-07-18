"""
Inference abstraction layer for model-agnostic TTS inference.

This module provides an abstract interface for inference implementations
and includes a mock engine for testing before real model integration.
"""

from .base import BaseInference
from .exceptions import (
    AudioGenerationError,
    InferenceException,
    InvalidChunkError,
    ResourceLoadError,
)
from .f5_inference import F5Inference
from .interface_validator import InferenceInterfaceValidator
from .mock_inference import MockInference
from .models import AudioResult, Chunk

__all__ = [
    "BaseInference",
    "Chunk",
    "AudioResult",
    "InferenceException",
    "InvalidChunkError",
    "AudioGenerationError",
    "ResourceLoadError",
    "InferenceInterfaceValidator",
    "F5Inference",
    "MockInference",
]
