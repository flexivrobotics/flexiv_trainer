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
