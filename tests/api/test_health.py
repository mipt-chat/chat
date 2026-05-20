import sys
import types

from fastapi.testclient import TestClient

from app.api.routes import health
from app.main import create_app


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def test_health_live() -> None:
    response = _client().get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_missing_chroma_dir(monkeypatch, tmp_path) -> None:
    missing_dir = tmp_path / "missing"
    monkeypatch.setattr(health.settings, "chroma_persist_directory", str(missing_dir))

    response = _client().get("/health/ready")

    assert response.status_code == 503
    assert "Run indexing first" in response.json()["detail"]


def test_health_ready_success(monkeypatch, tmp_path) -> None:
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    class FakeCollection:
        def count(self) -> int:
            return 3

    class FakeClient:
        def __init__(self, path: str) -> None:
            self.path = path

        def get_collection(self, name: str) -> FakeCollection:
            return FakeCollection()

    fake_chromadb = types.SimpleNamespace(PersistentClient=FakeClient)

    monkeypatch.setattr(health.settings, "chroma_persist_directory", str(chroma_dir))
    monkeypatch.setattr(health.settings, "chroma_collection_name", "support_knowledge")
    monkeypatch.setattr(health.settings, "active_llm_provider", "giga")
    monkeypatch.setattr(health.settings, "gigachat_api_key", "token")
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    response = _client().get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "collection": "support_knowledge",
        "chunks": 3,
        "llm_provider": "giga",
    }


def test_health_ready_empty_collection(monkeypatch, tmp_path) -> None:
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    class FakeCollection:
        def count(self) -> int:
            return 0

    class FakeClient:
        def __init__(self, path: str) -> None:
            self.path = path

        def get_collection(self, name: str) -> FakeCollection:
            return FakeCollection()

    fake_chromadb = types.SimpleNamespace(PersistentClient=FakeClient)

    monkeypatch.setattr(health.settings, "chroma_persist_directory", str(chroma_dir))
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    response = _client().get("/health/ready")

    assert response.status_code == 503
    assert "collection is empty" in response.json()["detail"]
