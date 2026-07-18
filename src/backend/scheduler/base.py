from abc import ABC, abstractmethod
from typing import List, Callable, Any

from backend.chunking.chunk import Chunk as ChunkingChunk
from backend.inference.models import AudioResult


class BaseScheduler(ABC):
    """Abstract scheduler interface."""

    @abstractmethod
    def schedule(self, chunks: List[ChunkingChunk], infer_callable: Callable[[Any], AudioResult]) -> List[AudioResult]:
        """Schedule synthesis of chunks and return results in original order."""
        raise NotImplementedError()
