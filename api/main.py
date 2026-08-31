from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.app_routes import router as app_router
from api.auth_routes import router as auth_router
from api.document_routes import router as document_router


app = FastAPI(title="FigureLens", version="0.1.0")
app.include_router(auth_router)
app.include_router(app_router)
app.include_router(document_router)


@app.middleware("http")
async def local_security_headers(request: Request, call_next):
    """Apply browser hardening suitable for a same-origin local UI."""

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; object-src 'none'; base-uri 'self'"
    )
    return response


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse("/ui/")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/ui", StaticFiles(directory=_FRONTEND, html=True), name="ui")
