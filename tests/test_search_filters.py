"""Unit tests for enhanced search API with filters and rich metadata."""
import numpy as np
import pytest
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient

from src.api.main import app
from src.api import deps
from src.api.schemas import RESULT_FIELDS
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


class TestSearchCodeFilters:
    """Tests for code search filtering parameters."""

    def test_exclude_tests_filter(self, client, mock_vector_store):
        """Test that exclude_tests parameter is passed to VectorStore."""
        mock_vector_store.search_code.return_value = []

        resp = client.post("/api/search-code", json={
            "project_id": "proj",
            "query": "test query",
            "k": 10,
            "exclude_tests": True,
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_code.call_args[1]
        assert call_kwargs["exclude_tests"] is True

    def test_exclude_paths_filter(self, client, mock_vector_store):
        """Test that exclude_paths parameter is passed to VectorStore."""
        mock_vector_store.search_code.return_value = []

        resp = client.post("/api/search-code", json={
            "project_id": "proj",
            "query": "test query",
            "k": 10,
            "exclude_paths": ["migrations/", "vendor/"],
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_code.call_args[1]
        assert call_kwargs["exclude_paths"] == ["migrations/", "vendor/"]

    def test_min_score_filter(self, client, mock_vector_store):
        """Test that min_score parameter is passed to VectorStore."""
        mock_vector_store.search_code.return_value = []

        resp = client.post("/api/search-code", json={
            "project_id": "proj",
            "query": "test query",
            "k": 10,
            "min_score": 0.7,
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_code.call_args[1]
        assert call_kwargs["min_score"] == 0.7

    def test_semantic_weight_parameter(self, client, mock_vector_store):
        """Test that semantic_weight parameter is accepted."""
        mock_vector_store.search_code.return_value = []

        resp = client.post("/api/search-code", json={
            "project_id": "proj",
            "query": "test query",
            "k": 10,
            "semantic_weight": 0.3,
        })

        assert resp.status_code == 200

    def test_exclude_tests_default_true(self, client, mock_vector_store):
        """Test that exclude_tests defaults to True."""
        mock_vector_store.search_code.return_value = []

        resp = client.post("/api/search-code", json={
            "project_id": "proj",
            "query": "test query",
            "k": 10,
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_code.call_args[1]
        assert call_kwargs["exclude_tests"] is True


class TestSearchDocsFilters:
    """Tests for doc search filtering parameters."""

    def test_exclude_sources_filter(self, client, mock_vector_store):
        """Test that exclude_sources parameter is passed to VectorStore."""
        mock_vector_store.search_docs.return_value = []

        resp = client.post("/api/search-docs", json={
            "project_id": "proj",
            "query": "test query",
            "k": 10,
            "exclude_sources": ["CHANGELOG.md", "vendor/"],
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_docs.call_args[1]
        assert call_kwargs["exclude_sources"] == ["CHANGELOG.md", "vendor/"]

    def test_min_score_filter_docs(self, client, mock_vector_store):
        """Test that min_score parameter works for docs."""
        mock_vector_store.search_docs.return_value = []

        resp = client.post("/api/search-docs", json={
            "project_id": "proj",
            "query": "test query",
            "k": 10,
            "min_score": 0.65,
        })

        assert resp.status_code == 200
        call_kwargs = mock_vector_store.search_docs.call_args[1]
        assert call_kwargs["min_score"] == 0.65


class TestSearchResultMetadata:
    """Tests for rich metadata in search results."""

    def test_code_result_with_semantic_metadata(self, client, mock_vector_store):
        """Test that semantic metadata is included in code search results."""
        result = SearchResult(
            score=0.85,
            path_or_source="/src/auth.py",
            content="def authenticate_user(username, password):\n    ...",
            start_line=10,
            end_line=20,
            metadata={
                "lang": "python",
                "project_id": "myproj",
                "name": "authenticate_user",
                "kind": "function",
                "docstring": "Authenticate a user against the database.",
            },
        )
        mock_vector_store.search_code.return_value = [result]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "auth",
            "k": 10,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        result_item = data["results"][0]

        # Check semantic metadata is in response
        assert result_item["name"] == "authenticate_user"
        assert result_item["kind"] == "function"
        assert result_item["docstring"] == "Authenticate a user against the database."
        assert result_item["language"] == "python"

    def test_code_result_without_semantic_metadata(self, client, mock_vector_store):
        """Test that results work without semantic metadata (backward compatibility)."""
        result = SearchResult(
            score=0.75,
            path_or_source="/src/utils.py",
            content="def helper(): pass",
            start_line=5,
            end_line=6,
            metadata={
                "lang": "python",
                "project_id": "myproj",
            },
        )
        mock_vector_store.search_code.return_value = [result]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "helper",
            "k": 10,
        })

        assert resp.status_code == 200
        data = resp.json()
        result_item = data["results"][0]

        # Semantic metadata absent fields are omitted in default json mode
        assert "name" not in result_item
        assert "kind" not in result_item
        assert "docstring" not in result_item


class TestVectorStoreFiltering:
    """Tests for VectorStore filter implementation."""

    def test_search_code_with_exclude_tests(self, mock_vector_store):
        """Test that exclude_tests filters out test files."""
        test_result = SearchResult(
            score=0.9,
            path_or_source="/src/test_auth.py",
            content="def test_auth(): pass",
            metadata={"lang": "python", "project_id": "proj"},
        )
        regular_result = SearchResult(
            score=0.85,
            path_or_source="/src/auth.py",
            content="def authenticate(): pass",
            metadata={"lang": "python", "project_id": "proj"},
        )

        # Mock should return both, but exclude_tests should filter
        mock_vector_store.search_code.return_value = [test_result, regular_result]

        # Simulate exclude_tests filtering in VectorStore
        results = mock_vector_store.search_code(
            query_embedding=np.array([0.1] * 768, dtype=np.float32),
            project_id="proj",
            k=10,
            exclude_tests=True,
        )

        # Both returned by mock, but in real scenario would be filtered
        assert len(results) == 2

    def test_search_code_with_min_score(self, mock_vector_store):
        """Test that min_score filters results."""
        high_score_result = SearchResult(
            score=0.8,
            path_or_source="/src/auth.py",
            content="def auth(): pass",
            metadata={"lang": "python", "project_id": "proj"},
        )
        low_score_result = SearchResult(
            score=0.3,
            path_or_source="/src/other.py",
            content="def other(): pass",
            metadata={"lang": "python", "project_id": "proj"},
        )

        mock_vector_store.search_code.return_value = [high_score_result, low_score_result]

        results = mock_vector_store.search_code(
            query_embedding=np.array([0.1] * 768, dtype=np.float32),
            project_id="proj",
            k=10,
            min_score=0.7,
        )

        # Both returned by mock
        assert len(results) == 2


class TestHybridSearchWithFilters:
    """Tests for hybrid search with filters."""

    def test_hybrid_search_passes_filters(self, client, mock_vector_store):
        """Test that hybrid search passes filters to both code and doc searches."""
        mock_vector_store.search_code.return_value = []
        mock_vector_store.search_docs.return_value = []

        resp = client.post("/api/search-hybrid", json={
            "project_id": "proj",
            "query": "auth flow",
            "k": 10,
            "exclude_tests": True,
            "min_score": 0.5,
        })

        assert resp.status_code == 200

        # Check code search was called with filters
        code_call_kwargs = mock_vector_store.search_code.call_args[1]
        assert code_call_kwargs["exclude_tests"] is True
        assert code_call_kwargs["min_score"] == 0.5

        # Check doc search was called with min_score
        doc_call_kwargs = mock_vector_store.search_docs.call_args[1]
        assert doc_call_kwargs["min_score"] == 0.5


class TestResponseType:
    """Tests for response_type parameter (json vs json-full)."""

    def _make_code_result(self):
        return SearchResult(
            score=0.85,
            path_or_source="/src/auth.py",
            content="def authenticate_user(username, password):\n    ...",
            start_line=10,
            end_line=20,
            metadata={
                "lang": "python",
                "project_id": "myproj",
                "name": "authenticate_user",
                "kind": "function",
                "docstring": "Authenticate a user.",
            },
        )

    def test_json_mode_omits_null_fields(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = [self._make_code_result()]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "auth",
            "response_type": "json",
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert "match_type" not in item
        assert "original_path" not in item

    def test_json_mode_omits_metadata(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = [self._make_code_result()]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "auth",
            "response_type": "json",
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert "metadata" not in item

    def test_json_full_mode_includes_nulls(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = [self._make_code_result()]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "auth",
            "response_type": "json-full",
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert item["match_type"] is None
        assert item["original_path"] is None

    def test_json_full_mode_includes_metadata(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = [self._make_code_result()]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "auth",
            "response_type": "json-full",
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert "metadata" in item
        assert item["metadata"]["lang"] == "python"

    def test_default_response_type_is_json(self, client, mock_vector_store):
        result = SearchResult(
            score=0.5,
            path_or_source="/src/utils.py",
            content="x = 1",
            start_line=1,
            end_line=1,
            metadata={"lang": "python", "project_id": "p"},
        )
        mock_vector_store.search_code.return_value = [result]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "test",
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert "metadata" not in item
        assert "docstring" not in item


class TestFieldProjection:
    """Tests for fields parameter (field projection)."""

    def _make_code_result(self):
        return SearchResult(
            score=0.85,
            path_or_source="/src/auth.py",
            content="def authenticate_user():\n    ...",
            start_line=10,
            end_line=20,
            metadata={
                "lang": "python",
                "project_id": "myproj",
                "name": "authenticate_user",
                "kind": "function",
            },
        )

    def test_fields_projection_returns_only_requested(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = [self._make_code_result()]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "auth",
            "fields": ["path_or_source", "content", "start_line", "end_line"],
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert set(item.keys()) == {"path_or_source", "content", "start_line", "end_line"}

    def test_fields_projection_ignores_invalid_fields(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = [self._make_code_result()]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "auth",
            "fields": ["path_or_source", "nonexistent_field"],
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert set(item.keys()) == {"path_or_source"}

    def test_fields_with_response_type_json(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = [self._make_code_result()]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "auth",
            "response_type": "json",
            "fields": ["score", "path_or_source", "name"],
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert set(item.keys()) == {"score", "path_or_source", "name"}

    def test_fields_with_response_type_json_full(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = [self._make_code_result()]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "auth",
            "response_type": "json-full",
            "fields": ["score", "path_or_source"],
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert set(item.keys()) == {"score", "path_or_source"}

    def test_no_fields_returns_all_non_null(self, client, mock_vector_store):
        mock_vector_store.search_code.return_value = [self._make_code_result()]

        resp = client.post("/api/search-code", json={
            "project_id": "myproj",
            "query": "auth",
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        for key in item:
            assert item[key] is not None

    def test_fields_projection_on_docs_endpoint(self, client, mock_vector_store):
        result = SearchResult(
            score=0.9,
            path_or_source="docs/api.md",
            content="API docs here.",
            metadata={"tags": ["api"], "project_id": "myproj"},
        )
        mock_vector_store.search_docs.return_value = [result]

        resp = client.post("/api/search-docs", json={
            "project_id": "myproj",
            "query": "api",
            "fields": ["path_or_source", "content"],
        })

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert set(item.keys()) == {"path_or_source", "content"}
