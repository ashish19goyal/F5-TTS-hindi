from fastapi import FastAPI

from backend.pipeline import TTSPipeline

app = FastAPI(title="F5-TTS Hindi Backend")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/tts")
async def synthesize_text(text: str):
    with TTSPipeline() as pipeline:
        results = pipeline.process(text)
    return {
        "status": "success",
        "chunks": len(results),
        "audio_paths": [result.audio_path for result in results],
    }


def main() -> None:
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)

