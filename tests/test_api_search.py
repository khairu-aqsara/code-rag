"""Unit tests for search API endpoints — uses mocked embedder and vector_store."""
import numpy as np
import pytest
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient

from src.api.main import app
from src.api import deps
from src.vector_store import SearchResult


@asynccontextmanager
async def _noop_lifespan(app):
    """Bypass model loading and Redis connection in unit tests."""
    yield


@pytest.fixture
def client(mock_embedder, mock_vector_store):
    """TestClient with dependency overrides and no-op lifespan."""
    app.dependency_overrides[deps.get_embedding_service] = lambda: mock_embedder
    app.dependency_overrides[deps.get_vector_store] = lambda: mock_vector_store
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()
    # Restore real lifespan (import locally to avoid torch at module level)
    from src.api.main import lifespan
    app.router.lifespan_context = lifespan


@pytest.fixture
def sample_code_result():
    return SearchResult(
        score=0.1,
        path_or_source="/src/utils.py",
        content="def helper(): pass",
        start_line=10,
        end_line=11,
        metadata={"lang": "python", "project_id": "myproj"},
    )


@pytest.fixture
def sample_doc_result():
    return SearchResult(
        score=0.2,
        path_or_source="docs/api.md",
        content="API documentation content here.",
        metadata={"tags": ["api"], "project_id": "myproj"},
    )


class TestSearchCodeEndpoint:
    def test_search_code_success(self, client, mock_vector_store, mock_embedder, sample_code_result):
        mock_vector_store.search_code.return_value = [sample_code_result]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "helper function",
            "k": 5,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["path_or_source"] == "/src/utils.py"

    def test_search_code_with_lang_filter(self, client, mock_vector_store, sample_code_result):
        mock_vector_store.search_code.return_value = [sample_code_result]

        resp = client.post("/api/search-code", json={
            "project_id": "proj",
            "query": "function",
            "k": 5,
            "lang_filter": ["python", "javascript"],
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_code.call_args[1]
        assert call_kwargs["lang_filter"] == ["python", "javascript"]

    def test_search_code_k_validation(self, client):
        resp = client.post("/api/search-code", json={
            "project_id": "proj",
            "query": "test",
            "k": 0,  # below minimum
        })
        assert resp.status_code == 422

        resp2 = client.post("/api/search-code", json={
            "project_id": "proj",
            "query": "test",
            "k": 51,  # above maximum (max is 50)
        })
        assert resp2.status_code == 422

    def test_search_code_empty_results(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = []

        resp = client.post("/api/search-code", json={
            "project_id": "proj",
            "query": "nonexistent",
        })

        assert resp.status_code == 200
        assert resp.json()["results"] == []


class TestSearchDocsEndpoint:
    def test_search_docs_success(self, client, mock_vector_store, sample_doc_result):
        mock_vector_store.search_docs.return_value = [sample_doc_result]

        resp = client.post("/api/search-docs", json={
            "project_id": "myproj",
            "query": "API documentation",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["path_or_source"] == "docs/api.md"

    def test_search_docs_with_tags(self, client, mock_vector_store, sample_doc_result):
        mock_vector_store.search_docs.return_value = [sample_doc_result]

        resp = client.post("/api/search-docs", json={
            "project_id": "proj",
            "query": "auth",
            "tags": ["auth", "security"],
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_docs.call_args[1]
        assert call_kwargs["tags"] == ["auth", "security"]


class TestSearchHybridEndpoint:
    def test_hybrid_merges_results(self, client, mock_vector_store, sample_code_result, sample_doc_result):
        mock_vector_store.search_code.return_value = [sample_code_result]
        mock_vector_store.search_docs.return_value = [sample_doc_result]

        resp = client.post("/api/search-hybrid", json={
            "project_id": "proj",
            "query": "helper function documentation",
            "k": 10,
        })

        assert resp.status_code == 200
        data = resp.json()
        # Should have both code and doc results merged
        assert len(data["results"]) == 2

    def test_hybrid_empty_results(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = []
        mock_vector_store.search_docs.return_value = []

        resp = client.post("/api/search-hybrid", json={
            "project_id": "proj",
            "query": "nothing",
        })

        assert resp.status_code == 200
        assert resp.json()["results"] == []
