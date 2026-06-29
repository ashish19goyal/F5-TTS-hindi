"""
Mock inference implementation for testing without a real model.
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.io import wavfile

from .base import BaseInference
from .exceptions import AudioGenerationError, InvalidChunkError, ResourceLoadError
from .models import AudioResult, Chunk


class MockInference(BaseInference):
    """
    Mock inference engine that generates predictable placeholder audio.

    This implementation is used for testing and development before the real
    F5 model integration. It produces deterministic sine wave audio.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mock inference engine.

        Args:
            config: Optional configuration. Supported keys:
                   - sample_rate: Output sample rate (default: 22050)
                   - duration: Audio duration in seconds (default: 1.0)
                   - frequency: Sine wave frequency in Hz (default: 440)
                   - output_dir: Directory for audio files (default: './audio_output')
        """
        super().__init__(config)

        self.sample_rate = self.config.get("sample_rate", 22050)
        self.duration = self.config.get("duration", 1.0)
        self.frequency = self.config.get("frequency", 440)
        self.output_dir = Path(self.config.get("output_dir", "./audio_output"))

        self.resources_loaded = False

    def load_resources(self) -> None:
        """
        Load and initialize resources.

        For mock engine, this just creates the output directory.

        Raises:
            ResourceLoadError: If resource loading fails
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.resources_loaded = True
        except Exception as e:
            raise ResourceLoadError(
                f"Failed to initialize mock inference: {str(e)}"
            ) from e

    def cleanup(self) -> None:
        """Release resources and clean up."""
        self.resources_loaded = False

    def _validate_chunk(self, chunk: Chunk) -> None:
        """
        Validate chunk object.

        Args:
            chunk: The chunk to validate

        Raises:
            InvalidChunkError: If validation fails
        """
        if not isinstance(chunk, Chunk):
            raise InvalidChunkError("Input must be a Chunk object")

        if not chunk.chunk_id:
            raise InvalidChunkError("Chunk must have a non-empty chunk_id")

        if not chunk.text:
            raise InvalidChunkError("Chunk must have non-empty text content")

    def _generate_audio(self, text: str) -> np.ndarray:
        """
        Generate deterministic sine wave audio based on text.

        Args:
            text: Input text (used to seed generation for determinism)

        Returns:
            Audio samples as numpy array
        """
        # Use text hash to seed the frequency for determinism
        text_hash = hash(text) % 1000
        adjusted_frequency = self.frequency + (text_hash % 50)

        # Generate sine wave
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        # Scale amplitude to int16 range
        amplitude = 30000
        audio = amplitude * np.sin(2 * np.pi * adjusted_frequency * t)
        audio = audio.astype(np.int16)

        return audio

    def _save_audio(self, audio: np.ndarray, chunk_id: str) -> str:
        """
        Save audio to WAV file.

        Args:
            audio: Audio samples as numpy array
            chunk_id: Chunk ID for filename

        Returns:
            Path to saved WAV file

        Raises:
            AudioGenerationError: If save fails
        """
        try:
            filename = f"{chunk_id}_{uuid.uuid4().hex[:8]}.wav"
            output_path = self.output_dir / filename

            wavfile.write(str(output_path), self.sample_rate, audio)
            return str(output_path)
        except Exception as e:
            raise AudioGenerationError(f"Failed to save audio file: {str(e)}") from e

    def infer(self, chunk: Chunk) -> AudioResult:
        """
        Synchronous inference for a single chunk.

        Generates deterministic sine wave audio.

        Args:
            chunk: The text chunk to synthesize

        Returns:
            AudioResult object with audio file path and metadata

        Raises:
            InvalidChunkError: If chunk validation fails
            AudioGenerationError: If audio generation fails
        """
        if not self.resources_loaded:
            raise ResourceLoadError("Resources not loaded. Call load_resources() first.")

        # Validate input
        self._validate_chunk(chunk)

        # Generate audio
        start_time = time.time()
        audio = self._generate_audio(chunk.text)
        audio_path = self._save_audio(audio, chunk.chunk_id)
        inference_latency = time.time() - start_time

        # Calculate duration
        duration_seconds = len(audio) / self.sample_rate

        return AudioResult(
            chunk_id=chunk.chunk_id,
            audio_path=audio_path,
            duration=duration_seconds,
            sample_rate=self.sample_rate,
            inference_latency=inference_latency,
            status="success",
            metadata={
                "model": "MockInference",
                "frequency": self.frequency,
                "text_length": len(chunk.text),
            },
        )

    async def infer_async(self, chunk: Chunk) -> AudioResult:
        """
        Asynchronous inference for distributed execution.

        For mock engine, simply calls sync version (no actual async work).

        Args:
            chunk: The text chunk to synthesize

        Returns:
            AudioResult object with audio file path and metadata

        Raises:
            InvalidChunkError: If chunk validation fails
            AudioGenerationError: If audio generation fails
        """
        return self.infer(chunk)
