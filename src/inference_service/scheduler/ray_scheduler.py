"""
Ray-based scheduler. Uses `ray` if available; otherwise raises ImportError.
This implementation expects `infer` or `infer_async` to be available on the inference
engine. It submits remote tasks to Ray and gathers results.
"""

try:
    import ray
except Exception as e:
    ray = None

from typing import List, Callable

from .base import BaseScheduler
from inference_service.chunking.chunk import Chunk as ChunkingChunk
from inference_service.engines.models import Chunk as InferenceChunk, AudioResult


class RayScheduler(BaseScheduler):
    def __init__(self, address: str = None):
        if ray is None:
            raise ImportError("ray is not installed")

        if address:
            ray.init(address=address)
        else:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

    def _to_inference_chunk(self, chunk: ChunkingChunk) -> InferenceChunk:
        return InferenceChunk(
            chunk_id=f"chunk_{chunk.id:03d}",
            text=chunk.text,
            metadata={"order": chunk.order, "original_length": chunk.length},
        )

    def schedule(self, chunks: List[ChunkingChunk], infer_callable: Callable[[InferenceChunk], AudioResult]) -> List[AudioResult]:
        # Wrap infer_callable as a Ray remote if it's a function
        remote = ray.remote(infer_callable).remote

        # Submit tasks
        refs = []
        for chunk in chunks:
            inf_chunk = self._to_inference_chunk(chunk)
            refs.append(remote(inf_chunk))

        # Fetch results preserving order
        results = ray.get(refs)
        return results
