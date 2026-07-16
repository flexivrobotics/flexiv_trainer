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

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from flexivtrainer import __version__
from flexivtrainer.api.routes import datasets, rollout, system, teleop, training
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
        # Warm up training device detection in the background so the first visit
        # to the Policy Training page doesn't block on a cold ``import torch`` +
        # CUDA init (which left the device list empty for tens of seconds).
        runtime = get_runtime_manager()
        threading.Thread(
            target=runtime.training.warm_up_devices,
            name="device-warmup",
            daemon=True,
        ).start()
        # Warm the depth-decode imports (lerobot.datasets.depth_utils pulls in
        # torch, ~2.7s cold) so the first depth preview frame isn't stalled.
        threading.Thread(
            target=runtime.warm_up_depth_decode,
            name="depth-decode-warmup",
            daemon=True,
        ).start()
        yield
        if get_runtime_manager.cache_info().currsize:
            get_runtime_manager().shutdown()
            get_runtime_manager.cache_clear()

    app = FastAPI(title="Flexiv Trainer API", version=__version__, lifespan=lifespan)

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
    def root() -> HTMLResponse:
        index_html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        index_html = index_html.replace("__FLEXIV_TRAINER_VERSION__", __version__)
        return HTMLResponse(
            index_html,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    app.include_router(system.router)
    app.include_router(teleop.router)
    app.include_router(datasets.router)
    app.include_router(training.router)
    app.include_router(rollout.router)
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
        # Give in-flight requests (e.g. a frame decode) a moment to finish on
        # Ctrl+C so they aren't force-cancelled into a wall of CancelledError
        # tracebacks. Playback is load-gated, so only a couple are ever active.
        timeout_graceful_shutdown=5,
    )
