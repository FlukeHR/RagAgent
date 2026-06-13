from __future__ import annotations

from fastapi import FastAPI

from api.routes import router

app = FastAPI(title="Paper RAG Agent", version="0.2.0")
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
