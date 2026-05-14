"""Unit tests for batch search endpoint."""
import pytest
from unittest.mock import MagicMock
from contextlib import asynccontextmanager

from src.api.main import app
from src.api import deps
from src._search import SearchResult


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
class TestBatchSearchEndpoint:
    def test_batch_search_multiple_queries(self, client, mock_embedder, mock_vector_store):
        """Multiple queries → returns results for each"""
        # Setup mock to return different results per query
        mock_vector_store.search_code_exact.return_value = [
            SearchResult(path_or_source="/test.py", content="def func1(): pass", score=0.9)
        ]
        mock_vector_store.search_code_hybrid.return_value = [
            SearchResult(path_or_source="/test.py", content="def func2(): pass", score=0.8)
        ]
        mock_vector_store.search_docs.return_value = [
            SearchResult(path_or_source="README.md", content="docs", score=0.7)
        ]
        
        resp = client.post("/api/search-batch", json={
            "project_id": "test",
            "queries": [
                {"type": "code_exact", "query": "func1"},
                {"type": "code_hybrid", "query": "concept", "semantic_weight": 0.7},
                {"type": "docs", "query": "docs"}
            ],
            "k": 5
        })
        
        assert resp.status_code == 200
        data = resp.json()
        assert "query_0" in data["results"]
        assert "query_1" in data["results"]
        assert "query_2" in data["results"]
    
    def test_batch_search_empty_queries(self, client):
        """Empty queries → returns validation error"""
        resp = client.post("/api/search-batch", json={
            "project_id": "test",
            "queries": [],
            "k": 5
        })
        
        assert resp.status_code == 422  # Validation error
    
    def test_batch_search_single_query(self, client, mock_embedder, mock_vector_store):
        """Single query → returns single result set"""
        mock_vector_store.search_code.return_value = [
            SearchResult(path_or_source="/test.py", content="def test(): pass", score=0.85)
        ]
        
        resp = client.post("/api/search-batch", json={
            "project_id": "test",
            "queries": [
                {"type": "code", "query": "test function"}
            ],
            "k": 5
        })
        
        assert resp.status_code == 200
        data = resp.json()
        assert "query_0" in data["results"]
        assert len(data["results"]["query_0"]) == 1
