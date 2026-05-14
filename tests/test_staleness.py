"""Unit tests for staleness detection — X-Index-Age header on search responses."""
import pytest
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient

from src.api.main import app
from src.api import deps
from src.vector_store import SearchResult


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


@pytest.mark.unit
class TestStalenessDetection:
    def test_index_age_header_format_days(self, client, mock_vector_store):
        """Index age > 24h → returns 'Xd' format"""
        mock_vector_store.search_code.return_value = []
        mock_vector_store.get_index_age.return_value = "4d"

        resp = client.post("/api/search-code", json={
            "project_id": "test",
            "query": "test"
        })

        assert resp.status_code == 200
        assert "X-Index-Age" in resp.headers
        assert resp.headers["X-Index-Age"] == "4d"

    def test_index_age_header_format_hours(self, client, mock_vector_store):
        """Index age < 24h → returns 'Xh' format"""
        mock_vector_store.search_code.return_value = []
        mock_vector_store.get_index_age.return_value = "5h"

        resp = client.post("/api/search-code", json={
            "project_id": "test",
            "query": "test"
        })

        assert resp.status_code == 200
        assert "X-Index-Age" in resp.headers
        assert resp.headers["X-Index-Age"] == "5h"

    def test_no_indexed_timestamp_returns_none(self, client, mock_vector_store):
        """No timestamp stored → header not present"""
        mock_vector_store.search_code.return_value = []
        mock_vector_store.get_index_age.return_value = None

        resp = client.post("/api/search-code", json={
            "project_id": "test",
            "query": "test"
        })

        assert resp.status_code == 200
        assert "X-Index-Age" not in resp.headers
