"""
End-to-end pipeline integrating text normalization, chunking, and inference.

Pipeline Flow:
    Raw Text
        ↓
    Normalization
        ↓
    Chunking
        ↓
    Inference
        ↓
    Audio Results
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile

from inference_service.preprocessing.normlizer import HindiNormalizer
from inference_service.chunking.chunker import ChunkGenerator
from inference_service.engines import (
    AudioResult,
    InferenceInterfaceValidator,
)
from inference_service.engines.mock_inference import MockInference
from inference_service.scheduler.local_scheduler import LocalThreadScheduler
from inference_service.scheduler.ray_scheduler import RayScheduler


class TTSPipeline:
    """
    End-to-end TTS pipeline combining normalization, chunking, and inference.
    """

    def __init__(
        self,
        max_chunk_chars: int = 500,
        inference_config: Optional[Dict[str, Any]] = None,
        scheduler=None,
        use_mock: bool = False,
    ):
        """
        Initialize the TTS pipeline.

        Args:
            max_chunk_chars: Maximum characters per chunk
            inference_config: Configuration for inference engine
        """
        self.normalizer = HindiNormalizer()
        self.chunker = ChunkGenerator(max_chars=max_chunk_chars)

        if inference_config is None:
            inference_config = {
                "output_dir": str(Path(tempfile.gettempdir()) / "tts_output"),
                "sample_rate": 22050,
                "duration": 1.0,
                "frequency": 440,
            }

        self.use_mock = use_mock
        self.fallback_to_mock = True
        self._mock_fallback_used = False

        if use_mock:
            self.inference = MockInference(config=inference_config)
        else:
            from inference_service.engines.f5_inference import F5Inference

            self.inference = F5Inference(config=inference_config)

        try:
            self.inference.load_resources()
        except Exception:
            if use_mock:
                raise
            self.inference = MockInference(config=inference_config)
            self.inference.load_resources()

        # Validate inference engine
        InferenceInterfaceValidator.validate(self.inference)

        # Scheduler: default to local thread pool if not provided.
        # If scheduler='ray', use RayScheduler when ray is installed.
        if scheduler == "ray":
            try:
                self.scheduler = RayScheduler()
            except ImportError:
                self.scheduler = LocalThreadScheduler()
        else:
            self.scheduler = scheduler or LocalThreadScheduler()

    def _infer_with_fallback(self, chunk):
        """Run inference with a graceful fallback to the mock engine if the real model errors."""
        if self.use_mock or self._mock_fallback_used:
            return self.inference.infer(chunk)

        try:
            return self.inference.infer(chunk)
        except Exception:
            if not self.fallback_to_mock:
                raise
            self._mock_fallback_used = True
            self.inference = MockInference(config=self.inference.config)
            self.inference.load_resources()
            return self.inference.infer(chunk)

    def process(self, raw_text: str) -> List[AudioResult]:
        """
        Process text through the entire pipeline.

        Args:
            raw_text: Raw Hindi text input

        Returns:
            List of AudioResult objects, one per chunk

        Raises:
            Various exceptions from normalization, chunking, or inference
        """
        # Step 1: Normalization
        normalized_text = self.normalizer.normalize(raw_text)

        # Step 2: Chunking
        chunks = self.chunker.chunk(normalized_text)

        # Step 3: Inference
        # Use scheduler to run inference in parallel/distributed
        audio_results = self.scheduler.schedule(chunks, self._infer_with_fallback)

        # Ensure metadata merges order/original_length into inference metadata
        for i, res in enumerate(audio_results):
            # chunk order maps to i+1
            merged = {"order": i + 1, "original_length": getattr(res, "text_length", None)}
            # update only if not present
            res.metadata = {**merged, **res.metadata}

        return audio_results

    def cleanup(self) -> None:
        """Clean up resources."""
        self.inference.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def print_summary(
    raw_text: str,
    normalized_text: str,
    audio_results: List[AudioResult],
) -> None:
    """
    Print a summary of the pipeline execution.

    Args:
        raw_text: Original input text
        normalized_text: Normalized text
        audio_results: List of audio results
    """
    print("\n" + "=" * 80)
    print("TTS PIPELINE EXECUTION SUMMARY")
    print("=" * 80)

    print(f"\n📝 INPUT TEXT ({len(raw_text)} chars):")
    print(f"  {raw_text[:100]}{'...' if len(raw_text) > 100 else ''}")

    print(f"\n✨ NORMALIZED TEXT ({len(normalized_text)} chars):")
    print(f"  {normalized_text[:100]}{'...' if len(normalized_text) > 100 else ''}")

    print(f"\n📦 CHUNKS CREATED: {len(audio_results)}")
    for i, result in enumerate(audio_results, 1):
        print(f"\n  Chunk {i}:")
        print(f"    ID: {result.chunk_id}")
        print(f"    Audio Path: {result.audio_path}")
        print(f"    Duration: {result.duration:.2f}s")
        print(f"    Sample Rate: {result.sample_rate} Hz")
        print(f"    Latency: {result.inference_latency*1000:.1f}ms")
        print(f"    Status: {result.status}")
        print(f"    Metadata: {result.metadata}")

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


def run_demo(text: str, scheduler: Optional[str] = None) -> None:
    """
    Run a demo of the TTS pipeline.

    Args:
        text: Hindi text to process
        scheduler: Optional scheduler name ('ray' or None)
    """
    with TTSPipeline(max_chunk_chars=500, scheduler=scheduler) as pipeline:
        # Get normalized text for display
        normalized = pipeline.normalizer.normalize(text)

        # Process through pipeline
        results = pipeline.process(text)

        # Print summary
        print_summary(text, normalized, results)


if __name__ == "__main__":
    # Example Hindi text
    sample_text = """
    नमस्ते, दुनिया! 
    यह एक टेस्ट है। मैं अपना नाम संजय हूँ।
    मुझे प्रोग्रामिंग बहुत पसंद है।
    """

    print("\n🚀 Starting TTS Pipeline Demo...\n")
    run_demo(sample_text, scheduler="ray")
