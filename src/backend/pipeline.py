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

from pre_processing.normlizer import HindiNormalizer
from chunking.chunker import ChunkGenerator
from inference import (
    Chunk as InferenceChunk,
    AudioResult,
    MockInference,
    InferenceInterfaceValidator,
)


class TTSPipeline:
    """
    End-to-end TTS pipeline combining normalization, chunking, and inference.
    """

    def __init__(
        self,
        max_chunk_chars: int = 500,
        inference_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the TTS pipeline.

        Args:
            max_chunk_chars: Maximum characters per chunk
            inference_config: Configuration for inference engine
        """
        self.normalizer = HindiNormalizer()
        self.chunker = ChunkGenerator(max_chars=max_chunk_chars)

        # Default inference config
        if inference_config is None:
            inference_config = {
                "output_dir": str(Path(tempfile.gettempdir()) / "tts_output"),
                "sample_rate": 22050,
                "duration": 1.0,
                "frequency": 440,
            }

        self.inference = MockInference(config=inference_config)
        self.inference.load_resources()

        # Validate inference engine
        InferenceInterfaceValidator.validate(self.inference)

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
        audio_results = []
        for chunk in chunks:
            # Convert chunking.Chunk to inference.Chunk
            inference_chunk = InferenceChunk(
                chunk_id=f"chunk_{chunk.id:03d}",
                text=chunk.text,
                metadata={
                    "order": chunk.order,
                    "original_length": chunk.length,
                },
            )

            # Run inference
            result = self.inference.infer(inference_chunk)

            # Merge metadata from both sources
            merged_metadata = {
                "order": chunk.order,
                "original_length": chunk.length,
                **result.metadata,  # Inference metadata takes precedence
            }
            result.metadata = merged_metadata

            audio_results.append(result)

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


def run_demo(text: str) -> None:
    """
    Run a demo of the TTS pipeline.

    Args:
        text: Hindi text to process
    """
    with TTSPipeline(max_chunk_chars=500) as pipeline:
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
    run_demo(sample_text)
