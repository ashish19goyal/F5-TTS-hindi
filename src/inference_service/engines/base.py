"""
Abstract base class for inference implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .models import AudioResult, Chunk


class BaseInference(ABC):
    """
    Abstract base class for all inference implementations.

    Any concrete inference engine must implement this interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the inference engine.

        Args:
            config: Optional configuration dictionary with defaults.
                   Keys may include: model_name, device, batch_size, etc.
        """
        self.config = config or {}

    @abstractmethod
    def infer(self, chunk: Chunk) -> AudioResult:
        """
        Synchronous inference for a single chunk.

        Args:
            chunk: The text chunk to synthesize

        Returns:
            AudioResult object with audio file path and metadata

        Raises:
            InvalidChunkError: If chunk validation fails
            AudioGenerationError: If audio generation fails
        """
        pass

    @abstractmethod
    async def infer_async(self, chunk: Chunk) -> AudioResult:
        """
        Asynchronous inference for distributed execution (e.g., Ray workers).

        Args:
            chunk: The text chunk to synthesize

        Returns:
            AudioResult object with audio file path and metadata

        Raises:
            InvalidChunkError: If chunk validation fails
            AudioGenerationError: If audio generation fails
        """
        pass

    @abstractmethod
    def load_resources(self) -> None:
        """
        Load and initialize resources (e.g., model to GPU).

        Called once during startup.

        Raises:
            ResourceLoadError: If resource loading fails
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Release resources and clean up (e.g., free GPU memory).

        Called during shutdown.
        """
        pass
