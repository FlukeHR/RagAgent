from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(title="Paper RAG Agent", version="0.2.0")
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


# 前端单页（原生 HTML/JS），访问 http://localhost:8000/ui/
_WEB_DIR = Path(__file__).resolve().parent.parent / "frontend" / "web"
app.mount("/ui", StaticFiles(directory=str(_WEB_DIR), html=True), name="ui")
