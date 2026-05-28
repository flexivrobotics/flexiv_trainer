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

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from flexivtrainer.observability import info, ok
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/datasets", tags=["datasets"])


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
    info(
        "Combining episode datasets",
        f"count={len(request.episode_paths)} output={request.output_name}",
    )
    result = runtime.combine_episodes(request.episode_paths, request.output_name)
    ok(
        "Combined dataset ready",
        " ".join(
            [
                f"root={result.get('root', request.output_name)}",
                f"episodes={result.get('episodes', len(request.episode_paths))}",
                f"output={result.get('output_name', request.output_name)}",
            ]
        ),
    )
    return result
