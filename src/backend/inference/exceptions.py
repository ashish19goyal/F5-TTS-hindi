"""
Exceptions for the inference layer.
"""


class InferenceException(Exception):
    """Base exception for all inference-related errors."""

    pass


class InvalidChunkError(InferenceException):
    """Raised when chunk validation fails."""

    pass


class AudioGenerationError(InferenceException):
    """Raised when audio generation fails."""

    pass


class ResourceLoadError(InferenceException):
    """Raised when resource loading fails."""

    pass
