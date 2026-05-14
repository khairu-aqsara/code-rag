"""Unit tests for admin API endpoints — uses mocked embedder and vector_store."""
import pytest
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.main import app
from src.api import deps
from src.ingestor.code import IngestResult


@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture
def client(mock_embedder, mock_vector_store):
    app.dependency_overrides[deps.get_embedding_service] = lambda: mock_embedder
    app.dependency_overrides[deps.get_vector_store] = lambda: mock_vector_store
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    from src.api.main import lifespan
    app.router.lifespan_context = lifespan


@pytest.fixture
def sample_ingest_result():
    return IngestResult(
        total_files=3,
        total_chunks=12,
        duration_seconds=1.5,
        errors=[],
    )


class TestHealthEndpoint:
    def test_health_ok(self, client, mock_vector_store):
        mock_vector_store.redis = MagicMock()
        mock_vector_store.redis.ping.return_value = True

        resp = client.get("/api/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["redis_ok"] is True
        assert data["models_loaded"] is True

    def test_health_redis_down(self, client, mock_vector_store):
        mock_vector_store.redis = MagicMock()
        mock_vector_store.redis.ping.side_effect = Exception("Connection refused")

        resp = client.get("/api/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert data["redis_ok"] is False


class TestStatsEndpoint:
    def test_stats_returns_index_info(self, client, mock_vector_store):
        mock_vector_store.get_stats.return_value = {
            "idx:code": {"num_docs": 100, "memory_in_bytes": 1024},
            "idx:docs": {"num_docs": 50, "memory_in_bytes": 512},
        }

        resp = client.get("/api/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert "indices" in data
        assert "idx:code" in data["indices"]


class TestIngestCodeEndpoint:
    def test_ingest_code_success(self, client, tmp_path):
        # Write a real python file so CodeIngestor finds something to ingest
        (tmp_path / "hello.py").write_text("def hello(): pass\n")

        with patch("src.api.routers.admin.settings") as mock_settings:
            mock_settings.BASE_PATH = str(tmp_path)
            resp = client.post("/api/projects/myproj/ingest-code", json={
                "root_path": str(tmp_path),
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["total_files"] >= 1

    def test_ingest_code_path_traversal_rejected(self, client):
        with patch("src.api.routers.admin.settings") as mock_settings:
            mock_settings.BASE_PATH = "/data"
            resp = client.post("/api/projects/proj/ingest-code", json={
                "root_path": "/etc/passwd",
            })
        assert resp.status_code == 400


class TestDeleteProjectEndpoint:
    def test_delete_project_success(self, client, mock_vector_store):
        mock_vector_store.delete_project.return_value = 42

        resp = client.delete("/api/projects/myproj")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["deleted_count"] == 42
        mock_vector_store.delete_project.assert_called_once_with("myproj")

    def test_delete_project_zero_count(self, client, mock_vector_store):
        mock_vector_store.delete_project.return_value = 0

        resp = client.delete("/api/projects/empty_project")

        assert resp.status_code == 200
        assert resp.json()["deleted_count"] == 0
