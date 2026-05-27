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

from fastapi.testclient import TestClient

from flexivtrainer.api.app import create_app
from flexivtrainer.runtime.manager import get_runtime_manager


def test_root_serves_packaged_ui() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert response.text


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
