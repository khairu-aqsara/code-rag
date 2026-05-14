"""Unit tests for project info endpoint."""
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
class TestProjectInfoEndpoint:
    def test_project_info_returns_all_stats(self, client, mock_vector_store):
        """Project info → returns code/docs chunks, languages, tags"""
        mock_vector_store.get_project_info.return_value = {
            "project_id": "myproject",
            "code_chunks": 100,
            "doc_chunks": 50,
            "last_indexed": 1715334000,
            "index_age_days": 4,
            "languages": ["python", "typescript"],
            "doc_tags": ["api"],
        }
        
        resp = client.get("/api/projects/myproject/info")
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["code_chunks"] == 100
        assert data["doc_chunks"] == 50
        assert data["languages"] == ["python", "typescript"]
    
    def test_project_info_empty_project(self, client, mock_vector_store):
        """Empty project → returns zeros and empty lists"""
        mock_vector_store.get_project_info.return_value = {
            "project_id": "empty",
            "code_chunks": 0,
            "doc_chunks": 0,
            "last_indexed": None,
            "index_age_days": None,
            "languages": [],
            "doc_tags": [],
        }
        
        resp = client.get("/api/projects/empty/info")
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["code_chunks"] == 0
        assert data["doc_chunks"] == 0
