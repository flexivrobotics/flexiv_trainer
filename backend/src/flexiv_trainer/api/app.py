from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from flexiv_trainer.api.routes import datasets, system, teleop, training
from flexiv_trainer.config import get_settings


def create_app() -> FastAPI:
    app = FastAPI(title="Flexiv Trainer API", version="0.1.0")
    app.include_router(system.router)
    app.include_router(teleop.router)
    app.include_router(datasets.router)
    app.include_router(training.router)
    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
