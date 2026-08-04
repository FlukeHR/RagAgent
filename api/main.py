from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.app_routes import router as app_router
from api.auth_routes import router as auth_router
from api.dependencies import ingest_manager, settings
from retrieval.search import prewarm_shared_models

app = FastAPI(title="Paper RAG Agent", version="0.2.0")
app.include_router(auth_router)
app.include_router(app_router)


@app.on_event("startup")
def initialize_background_services() -> None:
    """Initialize the unified store and mark interrupted local jobs retryable."""

    ingest_manager()
    if settings().agent.prewarm_on_startup:
        threading.Thread(
            target=prewarm_shared_models,
            args=(settings(),),
            name="rag-model-prewarm",
            daemon=True,
        ).start()


@app.middleware("http")
async def local_security_headers(request: Request, call_next):
    """Apply browser hardening suitable for the same-origin local dashboard."""

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "font-src 'self' data:; connect-src 'self' https: http://127.0.0.1:*; "
        "frame-src 'self' https://arxiv.org https://export.arxiv.org; "
        "object-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'self'"
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


# 前端单页，优先托管 Vite 构建产物；未构建时回退源码目录。
_WEB_SOURCE = Path(__file__).resolve().parent.parent / "frontend" / "web"
_WEB_DIR = _WEB_SOURCE / "dist" if (_WEB_SOURCE / "dist").exists() else _WEB_SOURCE
app.mount("/ui", StaticFiles(directory=str(_WEB_DIR), html=True), name="ui")
