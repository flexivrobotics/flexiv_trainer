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
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from flexivtrainer.observability import info, ok
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/datasets", tags=["datasets"])

# -- In-memory merge job state --
_merge_lock = threading.Lock()
_merge_job: dict[str, Any] | None = None


class MergeRequest(BaseModel):
    episode_paths: list[str] = Field(default_factory=list)
    output_name: str


@router.get("/episodes")
def list_episodes(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return {"episodes": runtime.list_episode_datasets()}


@router.get("/preview")
def preview(
    path: str,
    episode_index: int | None = None,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    try:
        return runtime.preview_dataset(Path(path), episode_index=episode_index)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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


@router.post("/merge")
def merge(
    request: MergeRequest, runtime: RuntimeManager = Depends(get_runtime_manager)
) -> dict:
    global _merge_job
    info(
        "Merging episode datasets",
        f"count={len(request.episode_paths)} output={request.output_name}",
    )

    with _merge_lock:
        if _merge_job and _merge_job["status"] == "running":
            raise HTTPException(
                status_code=409, detail="A merge is already in progress"
            )
        _merge_job = {
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
        assert _merge_job is not None
        _merge_job["episode_index"] = ep_idx
        _merge_job["total_episodes"] = total_eps
        _merge_job["frame_index"] = frame_idx
        _merge_job["total_frames"] = total_frames

    def _run() -> None:
        global _merge_job
        try:
            result = runtime.merge_episodes(
                request.episode_paths, request.output_name, on_progress=_on_progress
            )
            assert _merge_job is not None
            _merge_job["status"] = "done"
            _merge_job["result"] = result
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
            assert _merge_job is not None
            _merge_job["status"] = "error"
            _merge_job["error"] = str(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started", "total_episodes": len(request.episode_paths)}


@router.get("/merge-progress")
def merge_progress() -> dict:
    """Return the current merge progress."""
    if _merge_job is None:
        raise HTTPException(status_code=404, detail="No merge in progress")
    return _merge_job


@router.get("/series")
def series(
    path: str,
    episode_index: int | None = None,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    """Return numeric time-series data from a dataset (or one episode) for plotting."""
    try:
        return runtime.dataset_series(Path(path), episode_index=episode_index)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/frame-image")
def frame_image(
    path: str,
    key: str,
    index: int = 0,
    episode_index: int | None = None,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> Response:
    """Return a single frame image as JPEG."""
    try:
        data = runtime.dataset_frame_image(
            Path(path), key, index, episode_index=episode_index
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (RuntimeError, IndexError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # A frame's pixels never change for a given (path, key, index), so let the
    # browser cache aggressively. Playback prefetches frames ahead and relies on
    # this cache to display them without re-hitting the decoder.
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600, immutable"},
    )


@router.get("/video")
def video(
    path: str,
    key: str,
    chunk_index: int = 0,
    file_index: int = 0,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> FileResponse:
    """Stream a camera feed's MP4 directly (FileResponse supports range/seek)."""
    try:
        video_path = runtime.dataset_video_path(
            Path(path), key, chunk_index=chunk_index, file_index=file_index
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Cache-Control": "private, max-age=3600"},
    )
