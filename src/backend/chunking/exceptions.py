"""
Custom exceptions for Hindi chunking.
"""


class ChunkingError(Exception):
    """Base class for all chunking errors."""


class EmptyChunkError(ChunkingError):
    """Raised when output chunk text is empty."""


class InvalidInputTypeError(ChunkingError):
    """Raised when input text is not a string."""
