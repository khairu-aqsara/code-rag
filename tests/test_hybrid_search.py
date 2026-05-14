"""Tests for Phase 4: Hybrid keyword+semantic search with BM25."""
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
    from src.api.main import lifespan
    app.router.lifespan_context = lifespan


class TestHybridCodeSearch:
    """Tests for keyword+semantic hybrid code search."""

    def test_code_hybrid_search_endpoint_exists(self, client, mock_vector_store):
        """Test that /search-code-hybrid endpoint exists and responds."""
        mock_vector_store.search_code_hybrid.return_value = []

        resp = client.post("/api/search-code-hybrid", json={
            "project_id": "proj",
            "query": "authenticate_user",
            "k": 5,
        })

        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_code_hybrid_search_with_semantic_weight_0(self, client, mock_vector_store):
        """Test pure keyword search (semantic_weight=0.0)."""
        mock_vector_store.search_code_hybrid.return_value = []

        resp = client.post("/api/search-code-hybrid", json={
            "project_id": "proj",
            "query": "def authenticate",
            "k": 5,
            "semantic_weight": 0.0,
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_code_hybrid.call_args[1]
        assert call_kwargs["semantic_weight"] == 0.0

    def test_code_hybrid_search_with_semantic_weight_1(self, client, mock_vector_store):
        """Test pure semantic search (semantic_weight=1.0)."""
        mock_vector_store.search_code_hybrid.return_value = []

        resp = client.post("/api/search-code-hybrid", json={
            "project_id": "proj",
            "query": "user authentication",
            "k": 5,
            "semantic_weight": 1.0,
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_code_hybrid.call_args[1]
        assert call_kwargs["semantic_weight"] == 1.0

    def test_code_hybrid_search_balanced(self, client, mock_vector_store):
        """Test balanced hybrid search (semantic_weight=0.5)."""
        mock_vector_store.search_code_hybrid.return_value = []

        resp = client.post("/api/search-code-hybrid", json={
            "project_id": "proj",
            "query": "login validation",
            "k": 5,
            "semantic_weight": 0.5,
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_code_hybrid.call_args[1]
        assert call_kwargs["semantic_weight"] == 0.5

    def test_code_hybrid_search_with_filters(self, client, mock_vector_store):
        """Test hybrid search passes filters through."""
        mock_vector_store.search_code_hybrid.return_value = []

        resp = client.post("/api/search-code-hybrid", json={
            "project_id": "proj",
            "query": "authenticate",
            "k": 5,
            "exclude_tests": True,
            "exclude_paths": ["migrations/", "vendor/"],
            "min_score": 0.7,
            "semantic_weight": 0.6,
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_code_hybrid.call_args[1]
        assert call_kwargs["exclude_tests"] is True
        assert call_kwargs["exclude_paths"] == ["migrations/", "vendor/"]
        assert call_kwargs["min_score"] == 0.7
        assert call_kwargs["semantic_weight"] == 0.6

    def test_code_hybrid_search_returns_results(self, client, mock_vector_store):
        """Test hybrid search returns formatted results."""
        result = SearchResult(
            score=0.85,
            path_or_source="/src/auth.py",
            content="def authenticate(username: str, password: str): ...",
            start_line=10,
            end_line=20,
            metadata={
                "lang": "python",
                "project_id": "proj",
                "name": "authenticate",
                "kind": "function",
                "docstring": "Authenticate user.",
            },
        )
        mock_vector_store.search_code_hybrid.return_value = [result]

        resp = client.post("/api/search-code-hybrid", json={
            "project_id": "proj",
            "query": "authenticate",
            "k": 5,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        result_item = data["results"][0]
        assert result_item["path_or_source"] == "/src/auth.py"
        assert result_item["name"] == "authenticate"
        assert result_item["kind"] == "function"
        assert result_item["language"] == "python"


class TestHybridDocsSearch:
    """Tests for keyword+semantic hybrid documentation search."""

    def test_docs_hybrid_search_endpoint_exists(self, client, mock_vector_store):
        """Test that /search-docs-hybrid endpoint exists."""
        mock_vector_store.search_docs_hybrid.return_value = []

        resp = client.post("/api/search-docs-hybrid", json={
            "project_id": "proj",
            "query": "api documentation",
            "k": 5,
        })

        assert resp.status_code == 200

    def test_docs_hybrid_search_with_tags(self, client, mock_vector_store):
        """Test hybrid docs search with tag filters."""
        mock_vector_store.search_docs_hybrid.return_value = []

        resp = client.post("/api/search-docs-hybrid", json={
            "project_id": "proj",
            "query": "authentication",
            "k": 5,
            "tags": ["auth", "security"],
            "semantic_weight": 0.6,
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_docs_hybrid.call_args[1]
        assert call_kwargs["tags"] == ["auth", "security"]
        assert call_kwargs["semantic_weight"] == 0.6

    def test_docs_hybrid_search_with_filters(self, client, mock_vector_store):
        """Test hybrid docs search with exclusion filters."""
        mock_vector_store.search_docs_hybrid.return_value = []

        resp = client.post("/api/search-docs-hybrid", json={
            "project_id": "proj",
            "query": "setup",
            "k": 5,
            "exclude_sources": ["CHANGELOG.md", "vendor/"],
            "min_score": 0.65,
            "semantic_weight": 0.4,
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_docs_hybrid.call_args[1]
        assert call_kwargs["exclude_sources"] == ["CHANGELOG.md", "vendor/"]
        assert call_kwargs["min_score"] == 0.65
        assert call_kwargs["semantic_weight"] == 0.4


class TestVectorStoreHybridSearch:
    """Tests for VectorStore hybrid search implementation."""

    def test_search_code_hybrid_merges_results(self, mock_vector_store):
        """Test that hybrid search merges semantic and keyword results."""
        # Mock semantic results
        semantic_result = SearchResult(
            score=0.9,
            path_or_source="/src/auth.py",
            content="def authenticate(): pass",
            start_line=10,
            end_line=12,
            metadata={"lang": "python", "project_id": "proj"},
        )
        # Mock keyword results
        keyword_result = SearchResult(
            score=0.85,
            path_or_source="/src/validate.py",
            content="def validate_token(): pass",
            start_line=5,
            end_line=7,
            metadata={"lang": "python", "project_id": "proj"},
        )

        mock_vector_store.search_code_hybrid.return_value = [semantic_result, keyword_result]

        results = mock_vector_store.search_code_hybrid(
            query_text="authenticate",
            query_embedding=np.array([0.1] * 768, dtype=np.float32),
            project_id="proj",
            k=5,
            semantic_weight=0.6,
        )

        assert len(results) == 2
        assert results[0].path_or_source in ["/src/auth.py", "/src/validate.py"]

    def test_search_docs_hybrid_respects_weight(self, mock_vector_store):
        """Test that hybrid search respects semantic_weight parameter."""
        result = SearchResult(
            score=0.8,
            path_or_source="docs/api.md",
            content="API documentation",
            metadata={"tags": ["api"], "project_id": "proj"},
        )
        mock_vector_store.search_docs_hybrid.return_value = [result]

        # Pure keyword
        results_keyword = mock_vector_store.search_docs_hybrid(
            query_text="api",
            query_embedding=np.array([0.1] * 768, dtype=np.float32),
            project_id="proj",
            k=5,
            semantic_weight=0.0,
        )
        assert len(results_keyword) == 1

        # Pure semantic
        results_semantic = mock_vector_store.search_docs_hybrid(
            query_text="documentation",
            query_embedding=np.array([0.1] * 768, dtype=np.float32),
            project_id="proj",
            k=5,
            semantic_weight=1.0,
        )
        assert len(results_semantic) == 1

        # Balanced
        results_balanced = mock_vector_store.search_docs_hybrid(
            query_text="api documentation",
            query_embedding=np.array([0.1] * 768, dtype=np.float32),
            project_id="proj",
            k=5,
            semantic_weight=0.5,
        )
        assert len(results_balanced) == 1


class TestUseCase:
    """Real-world use case tests for Phase 4."""

    def test_find_function_by_exact_name(self, client, mock_vector_store):
        """Use case: Developer searches for 'authenticate_user' function by name.

        With hybrid search and semantic_weight=0.2 (keyword-heavy),
        should find exact function name match even if semantic meaning is different.
        """
        result = SearchResult(
            score=0.92,
            path_or_source="/src/auth.py",
            content="def authenticate_user(username, password):\n    ...",
            start_line=15,
            end_line=25,
            metadata={
                "lang": "python",
                "project_id": "myproj",
                "name": "authenticate_user",
                "kind": "function",
            },
        )
        mock_vector_store.search_code_hybrid.return_value = [result]

        resp = client.post("/api/search-code-hybrid", json={
            "project_id": "myproj",
            "query": "authenticate_user",
            "k": 5,
            "semantic_weight": 0.2,  # keyword-heavy for exact matches
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["name"] == "authenticate_user"

    def test_find_by_semantic_meaning(self, client, mock_vector_store):
        """Use case: Developer searches for login/auth concept with semantic_weight=0.9.

        Should find semantically related functions even if names differ.
        """
        result = SearchResult(
            score=0.87,
            path_or_source="/src/security.py",
            content="def verify_credentials(user, secret):\n    ...",
            start_line=30,
            end_line=45,
            metadata={
                "lang": "python",
                "project_id": "myproj",
                "name": "verify_credentials",
                "kind": "function",
            },
        )
        mock_vector_store.search_code_hybrid.return_value = [result]

        resp = client.post("/api/search-code-hybrid", json={
            "project_id": "myproj",
            "query": "user login authentication",
            "k": 5,
            "semantic_weight": 0.9,  # semantic-heavy for concept search
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["path_or_source"] == "/src/security.py"
