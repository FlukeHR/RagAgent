from fastapi import FastAPI

from .middleware import add_process_time_header
from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Demo FastAPI Repo")
    app.middleware("http")(add_process_time_header)
    app.include_router(router)
    return app


app = create_app()
