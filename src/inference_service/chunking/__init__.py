"""
Hindi Text Chunking Package

This package provides a Hindi-specific smart chunking engine for
Text-to-Speech preprocessing.
"""

from .chunk import Chunk
from .chunker import ChunkGenerator

__all__ = [
    "Chunk",
    "ChunkGenerator",
]
