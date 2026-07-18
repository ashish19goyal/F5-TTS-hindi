from backend.service.app import app


def main() -> None:
    import uvicorn

    uvicorn.run("backend.service.app:app", host="0.0.0.0", port=8000, reload=False)

