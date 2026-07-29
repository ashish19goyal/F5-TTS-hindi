"""
Local Thread-based scheduler: dispatches inference calls to a ThreadPoolExecutor.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable
import os

from .base import BaseScheduler
from inference_service.chunking.chunk import Chunk as ChunkingChunk
from inference_service.engines.models import Chunk as InferenceChunk, AudioResult


class LocalThreadScheduler(BaseScheduler):
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or max(1, (os.cpu_count() or 2))

    def _to_inference_chunk(self, chunk: ChunkingChunk) -> InferenceChunk:
        return InferenceChunk(
            chunk_id=f"chunk_{chunk.id:03d}",
            text=chunk.text,
            metadata={"order": chunk.order, "original_length": chunk.length},
        )

    def schedule(self, chunks: List[ChunkingChunk], infer_callable: Callable[[InferenceChunk], AudioResult]) -> List[AudioResult]:
        if not chunks:
            return []

        results_by_order = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {}
            for chunk in chunks:
                inf_chunk = self._to_inference_chunk(chunk)
                future = ex.submit(infer_callable, inf_chunk)
                futures[future] = chunk.order

            for future in as_completed(futures):
                order = futures[future]
                res = future.result()
                results_by_order[order] = res

        # Return results ordered by chunk order
        return [results_by_order[i] for i in sorted(results_by_order.keys())]
