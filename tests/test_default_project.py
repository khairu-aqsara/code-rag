"""Unit tests for default project configuration endpoint."""
import pytest
from unittest.mock import MagicMock
from contextlib import asynccontextmanager

from src.api.main import app
from src.api import deps


@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture
def client(mock_embedder, mock_vector_store):
    """Test client with mocked dependencies."""
    app.dependency_overrides[deps.get_embedding_service] = lambda: mock_embedder
    app.dependency_overrides[deps.get_vector_store] = lambda: mock_vector_store
    app.router.lifespan_context = _noop_lifespan
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    from src.api.main import lifespan
    app.router.lifespan_context = lifespan


@pytest.mark.unit
class TestDefaultProjectEndpoint:
    def test_set_default_project(self, client, mock_vector_store):
        """Set default project → returns success"""
        mock_vector_store.redis.set.return_value = True
        
        resp = client.put("/api/config/default-project", json={
            "workspace": "my-workspace",
            "project_id": "myproject"
        })
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["workspace"] == "my-workspace"
        assert data["project_id"] == "myproject"
    
    def test_get_default_project_found(self, client, mock_vector_store):
        """Get default project → returns stored value"""
        mock_vector_store.redis.get.return_value = "myproject"
        
        resp = client.get("/api/config/default-project?workspace=my-workspace")
        
        assert resp.status_code == 200
        assert resp.json()["project_id"] == "myproject"
    
    def test_get_default_project_not_found(self, client, mock_vector_store):
        """Get default project → returns null when not set"""
        mock_vector_store.redis.get.return_value = None
        
        resp = client.get("/api/config/default-project?workspace=unknown")
        
        assert resp.status_code == 200
        assert resp.json()["project_id"] is None
