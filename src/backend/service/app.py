import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from backend.pipeline import TTSPipeline


class InferenceRequest(BaseModel):
    text: str
    max_chunk_chars: int = 500
    use_mock: bool = False
    ref_audio: str | None = None
    ref_text: str | None = None


class InferenceResponse(BaseModel):
    status: str
    chunks: int
    audio_paths: list[str]


def create_app() -> FastAPI:
    app = FastAPI(title="F5-TTS Hindi Inference Service")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/inference", response_model=InferenceResponse)
    async def infer(request: InferenceRequest) -> InferenceResponse:
        inference_config = {}
        if request.ref_audio:
            inference_config["ref_audio"] = request.ref_audio
        if request.ref_text:
            inference_config["ref_text"] = request.ref_text

        with TTSPipeline(
            max_chunk_chars=request.max_chunk_chars,
            use_mock=request.use_mock,
            inference_config=inference_config or None,
        ) as pipeline:
            results = pipeline.process(request.text)

        return InferenceResponse(
            status="success",
            chunks=len(results),
            audio_paths=[result.audio_path for result in results],
        )

    return app


def main() -> None:
    uvicorn.run("backend.service.app:app", host="0.0.0.0", port=8000, reload=False)


app = create_app()
