# Copyright 2026 Flexiv Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from flexivtrainer.api.routes import datasets, system, teleop, training
from flexivtrainer.config import get_settings
from flexivtrainer.observability import (
    banner,
    describe_exception,
    error,
    info,
    install_dependency_log_bridge,
    ok,
    section,
    warn,
)
from flexivtrainer.runtime.manager import get_runtime_manager

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if get_runtime_manager.cache_info().currsize:
            get_runtime_manager().shutdown()
            get_runtime_manager.cache_clear()

    app = FastAPI(title="Flexiv Trainer API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def terminal_request_log(request: Request, call_next):
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            error(
                f"Unhandled request failure for {request.method} {request.url.path}",
                describe_exception(exc),
            )
            raise

        elapsed_ms = (perf_counter() - started) * 1000
        if request.url.path.startswith("/static/"):
            return response

        detail = f"status={response.status_code} duration={elapsed_ms:.1f}ms"
        if response.status_code >= 500:
            error(f"{request.method} {request.url.path}", detail)
        elif response.status_code >= 400:
            warn(f"{request.method} {request.url.path}", detail)
        elif request.method != "GET" or request.url.path == "/":
            info(f"{request.method} {request.url.path}", detail)
        return response

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
    install_dependency_log_bridge()
    settings = get_settings()
    banner(
        "Flexiv Trainer Backend",
        f"UI      {settings.ui_url}",
        f"Docs    {settings.ui_url}docs",
        f"Listen  {settings.host}:{settings.port}",
        f"Data    {settings.storage.root}",
    )
    section(
        "Runtime",
        "Python-first operator UI, typed services, and live console observability",
    )
    info("Backend startup", f"robot_type={settings.robot_type}")
    ok(
        "Console observability ready",
        "live request traces and styled training stream enabled",
    )
    if not settings.teleop_robot_pairs:
        warn(
            "Teleoperation is not configured",
            "Set FLEXIV_TRAINER_TELEOP_ROBOT_PAIRS to enable robot control",
        )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=1,
    )
