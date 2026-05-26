from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from flexiv_trainer.api.routes import datasets, system, teleop, training
from flexiv_trainer.config import get_settings

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


def create_app() -> FastAPI:
    app = FastAPI(title="Flexiv Trainer API", version="0.1.0")

    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    app.include_router(system.router)
    app.include_router(teleop.router)
    app.include_router(datasets.router)
    app.include_router(training.router)
    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    print(settings.ui_url, flush=True)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="warning",
        access_log=False,
    )
