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

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from flexivtrainer.api.app import create_app
from flexivtrainer.api.routes.teleop import camera_frame
from flexivtrainer.runtime.manager import get_runtime_manager


def test_root_serves_packaged_ui() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/static/app.js?v=20260910-06" in response.text


def test_docs_route_is_available() -> None:
    client = TestClient(create_app())

    response = client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_camera_frame_route_returns_png() -> None:
    app = create_app()
    payload = {
        "image": [[[0, 0, 255], [0, 255, 0]]],
    }

    class FakeCameras:
        def capture_frame(self, camera_name: str):
            assert camera_name == "ego"
            return payload

    class FakeRuntime:
        cameras = FakeCameras()

    app.dependency_overrides[get_runtime_manager] = lambda: FakeRuntime()
    client = TestClient(app)

    response = client.get("/teleop/cameras/ego/frame")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_camera_frame_route_colorizes_depth() -> None:
    class FakeCameras:
        def __init__(self) -> None:
            self.leases: list[str] = []

        def capture_frame(self, camera_name: str):
            return {
                "image": [[[0, 0, 0]]],
                "depth": [[0, 500, 1000]],
            }

        def renew_depth_alignment_lease(self, camera_name: str) -> None:
            self.leases.append(camera_name)

    class FakeRuntime:
        cameras = FakeCameras()
        settings = SimpleNamespace(depth_max_m=2.0)

    runtime = FakeRuntime()
    response = camera_frame("ego", "depth", runtime)

    assert response.media_type == "image/png"
    assert response.body.startswith(b"\x89PNG\r\n\x1a\n")
    # Viewing depth must keep alignment alive, or the next frame has no depth.
    assert runtime.cameras.leases == ["ego"]


def test_camera_frame_depth_blames_the_rollout_when_one_is_running() -> None:
    # Alignment is refused during a rollout, so depth is absent. Say why rather
    # than reporting a missing depth stream the operator cannot act on.
    class FakeCameras:
        def capture_frame(self, camera_name: str):
            return {"image": [[[0, 0, 0]]]}

        def renew_depth_alignment_lease(self, camera_name: str) -> None:
            return None

    def _runtime(status: str):
        return SimpleNamespace(
            cameras=FakeCameras(),
            settings=SimpleNamespace(depth_max_m=2.0),
            rollout=SimpleNamespace(status=lambda: {"status": status}),
        )

    with pytest.raises(HTTPException) as running:
        camera_frame("ego", "depth", _runtime("running"))
    assert running.value.status_code == 409
    assert "rollout is running" in running.value.detail

    with pytest.raises(HTTPException) as idle:
        camera_frame("ego", "depth", _runtime("idle"))
    assert idle.value.status_code == 409
    assert "Depth stream is unavailable" in idle.value.detail


def test_runtime_manager_blocks_alignment_and_fails_closed(tmp_path) -> None:
    from flexivtrainer.config import AppSettings, StorageConfig
    from flexivtrainer.runtime.manager import RuntimeManager

    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    settings.ensure_storage()
    manager = RuntimeManager(settings)

    assert (
        manager.teleop._gripper_initialization_registry
        is manager.gripper_initialization
    )
    assert (
        manager.rollout._gripper_initialization_registry
        is manager.gripper_initialization
    )
    assert manager._rollout_is_running() is False
    manager.cameras.renew_depth_alignment_lease("ego")
    assert manager.cameras.depth_alignment_active("ego") is True

    # A running rollout must refuse viewer-driven alignment server-side, not
    # merely hide the checkbox.
    manager.cameras.clear_depth_alignment_leases()
    manager.rollout.status = lambda: {"status": "running"}
    manager.cameras.renew_depth_alignment_lease("ego")
    assert manager.cameras.depth_alignment_active("ego") is False

    # An unreadable rollout state must block, never permit.
    def boom():
        raise RuntimeError("status unavailable")

    manager.rollout.status = boom
    assert manager._rollout_is_running() is True


def _rollout_client(rollout) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_runtime_manager] = lambda: SimpleNamespace(
        rollout=rollout
    )
    return TestClient(app)


def test_rollout_start_is_one_call_that_runs_the_preflight() -> None:
    class FakeRollout:
        def __init__(self) -> None:
            self.started: list[str] = []

        def start(self, checkpoint_path, **kwargs):
            self.started.append(checkpoint_path)
            return {"status": "running", "checkpoint_path": checkpoint_path}

    rollout = FakeRollout()
    response = _rollout_client(rollout).post(
        "/rollout/start", json={"source": "local", "checkpoint_path": "/ckpt"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert rollout.started == ["/ckpt"]


@pytest.mark.parametrize(
    "message",
    [
        "Failed to disconnect teleoperation: TDK refused",
        "Failed to connect cameras: device busy",
    ],
)
def test_rollout_start_surfaces_preflight_failures(message: str) -> None:
    class FakeRollout:
        def start(self, checkpoint_path, **kwargs):
            raise RuntimeError(message)

    response = _rollout_client(FakeRollout()).post(
        "/rollout/start", json={"source": "local", "checkpoint_path": "/ckpt"}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == message
