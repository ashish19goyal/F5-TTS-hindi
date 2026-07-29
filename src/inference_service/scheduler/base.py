from abc import ABC, abstractmethod
from typing import List, Callable, Any

from inference_service.chunking.chunk import Chunk as ChunkingChunk
from inference_service.engines.models import AudioResult


class BaseScheduler(ABC):
    """Abstract scheduler interface."""

    @abstractmethod
    def schedule(self, chunks: List[ChunkingChunk], infer_callable: Callable[[Any], AudioResult]) -> List[AudioResult]:
        """Schedule synthesis of chunks and return results in original order."""
        raise NotImplementedError()
