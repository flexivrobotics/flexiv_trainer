from fastapi.testclient import TestClient

from flexivtrainer.api.app import create_app


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
