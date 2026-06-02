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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from flexivtrainer.observability import info, ok
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/datasets", tags=["datasets"])

# -- In-memory combine job state --
_combine_lock = threading.Lock()
_combine_job: dict[str, Any] | None = None


class CombineRequest(BaseModel):
    episode_paths: list[str] = Field(default_factory=list)
    output_name: str


@router.get("/episodes")
def list_episodes(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return {"episodes": runtime.list_episode_datasets()}


@router.get("/preview")
def preview(path: str, runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    try:
        return runtime.preview_dataset(Path(path))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/browse")
def browse(
    path: str | None = None,
    directories_only: bool = False,
    root_path: str | None = None,
    annotate_episode_dirs: bool = False,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    try:
        return runtime.browse_path(
            Path(path) if path else None,
            directories_only=directories_only,
            root_path=Path(root_path) if root_path else None,
            annotate_episode_dirs=annotate_episode_dirs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/combine")
def combine(
    request: CombineRequest, runtime: RuntimeManager = Depends(get_runtime_manager)
) -> dict:
    global _combine_job
    info(
        "Merging episode datasets",
        f"count={len(request.episode_paths)} output={request.output_name}",
    )

    with _combine_lock:
        if _combine_job and _combine_job["status"] == "running":
            raise HTTPException(
                status_code=409, detail="A merge is already in progress"
            )
        _combine_job = {
            "status": "running",
            "episode_index": 0,
            "total_episodes": len(request.episode_paths),
            "frame_index": 0,
            "total_frames": 0,
            "result": None,
            "error": None,
        }

    def _on_progress(
        ep_idx: int, total_eps: int, frame_idx: int, total_frames: int
    ) -> None:
        assert _combine_job is not None
        _combine_job["episode_index"] = ep_idx
        _combine_job["total_episodes"] = total_eps
        _combine_job["frame_index"] = frame_idx
        _combine_job["total_frames"] = total_frames

    def _run() -> None:
        global _combine_job
        try:
            result = runtime.combine_episodes(
                request.episode_paths, request.output_name, on_progress=_on_progress
            )
            assert _combine_job is not None
            _combine_job["status"] = "done"
            _combine_job["result"] = result
            ok(
                "Merged dataset ready",
                " ".join(
                    [
                        f"root={result.get('root', request.output_name)}",
                        f"episodes={result.get('episodes', len(request.episode_paths))}",
                        f"output={result.get('output_name', request.output_name)}",
                    ]
                ),
            )
        except Exception as exc:
            assert _combine_job is not None
            _combine_job["status"] = "error"
            _combine_job["error"] = str(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started", "total_episodes": len(request.episode_paths)}


@router.get("/combine-progress")
def combine_progress() -> dict:
    """Return the current merge progress."""
    if _combine_job is None:
        raise HTTPException(status_code=404, detail="No merge in progress")
    return _combine_job


@router.get("/series")
def series(
    path: str,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    """Return all numeric time-series data from a dataset for plotting."""
    try:
        return runtime.dataset_series(Path(path))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/frame-image")
def frame_image(
    path: str,
    key: str,
    index: int = 0,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> Response:
    """Return a single frame image as JPEG."""
    try:
        data = runtime.dataset_frame_image(Path(path), key, index)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (RuntimeError, IndexError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=data, media_type="image/jpeg")
