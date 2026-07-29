"""Real F5-TTS inference implementation for the inference service."""

from __future__ import annotations

import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseInference
from .exceptions import AudioGenerationError, InvalidChunkError, ResourceLoadError
from .models import AudioResult, Chunk


class F5Inference(BaseInference):
    """Wraps the real F5-TTS model behind the inference service interface."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        default_output_dir = Path(tempfile.gettempdir()) / "f5_tts_backend"
        self.output_dir = Path(self.config.get("output_dir", str(default_output_dir)))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = self.config.get("model_name", "F5TTS_v1_Base")
        self.device = self.config.get("device")
        self.hf_cache_dir = self.config.get("hf_cache_dir")
        default_ref_audio = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        self.ref_audio = self.config.get("ref_audio") or default_ref_audio
        self.ref_text = self.config.get("ref_text") or "Some call me nature, others call me mother nature."
        self._engine: Optional[Any] = None
        self.resources_loaded = False

    def load_resources(self) -> None:
        if self.resources_loaded:
            return
        try:
            from f5_tts.api import F5TTS

            self._engine = F5TTS(
                model=self.model_name,
                device=self.device,
                hf_cache_dir=self.hf_cache_dir,
            )
            self.resources_loaded = True
        except Exception as exc:  # pragma: no cover - depends on model availability
            raise ResourceLoadError(f"Failed to initialize F5-TTS model: {exc}") from exc

    def cleanup(self) -> None:
        self.resources_loaded = False
        self._engine = None

    def _validate_chunk(self, chunk: Chunk) -> None:
        if not isinstance(chunk, Chunk):
            raise InvalidChunkError("Input must be a Chunk object")
        if not chunk.chunk_id:
            raise InvalidChunkError("Chunk must have a non-empty chunk_id")
        if not chunk.text:
            raise InvalidChunkError("Chunk must have non-empty text content")

    def infer(self, chunk: Chunk) -> AudioResult:
        if not self.resources_loaded or self._engine is None:
            raise ResourceLoadError("Resources not loaded. Call load_resources() first.")

        self._validate_chunk(chunk)

        output_path = self.output_dir / f"{chunk.chunk_id}.wav"
        try:
            ref_audio = self.ref_audio or self.config.get("ref_audio") or ""
            ref_text = self.ref_text or self.config.get("ref_text") or ""
            wav, sr, _ = self._engine.infer(
                ref_file=ref_audio,
                ref_text=ref_text,
                gen_text=chunk.text,
                file_wave=str(output_path),
                seed=self.config.get("seed"),
            )
        except Exception as exc:  # pragma: no cover - depends on model availability
            raise AudioGenerationError(f"Failed to generate audio with F5-TTS: {exc}") from exc

        audio = np.asarray(wav)
        duration_seconds = len(audio) / sr if sr else 0.0

        return AudioResult(
            chunk_id=chunk.chunk_id,
            audio_path=str(output_path),
            duration=float(duration_seconds),
            sample_rate=int(sr),
            inference_latency=float(self.config.get("inference_latency", 0.0)),
            status="success",
            metadata={
                "model": self.model_name,
                "text_length": len(chunk.text),
                "engine": "F5Inference",
            },
        )

    async def infer_async(self, chunk: Chunk) -> AudioResult:
        return self.infer(chunk)
