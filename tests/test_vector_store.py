"""Integration tests for VectorStore — requires Redis Stack via Docker.

Run with: pytest -m integration
Requires: Docker running, testcontainers installed.
"""
import numpy as np
import pytest

from src.chunker import CodeChunk, DocChunk
from src.vector_store import VectorStore

pytestmark = pytest.mark.integration


@pytest.fixture
def vector_store(redis_client, flush_redis):
    return VectorStore(redis_client)


class TestVectorStoreIndexCreation:
    def test_indices_created(self, vector_store):
        """RediSearch indices should exist after VectorStore init."""
        result = vector_store.redis.execute_command("FT.INFO", "idx:code")
        assert result is not None

        result2 = vector_store.redis.execute_command("FT.INFO", "idx:docs")
        assert result2 is not None

    def test_recreate_is_idempotent(self, redis_client, flush_redis):
        """Creating VectorStore twice should not raise."""
        VectorStore(redis_client)
        VectorStore(redis_client)  # second init — no error


class TestCodeChunkInsertSearch:
    def test_insert_and_search(self, vector_store):
        chunks = [
            CodeChunk(code="def add(a, b): return a + b", path="/src/math.py", lang="python", start_line=1, end_line=1),
            CodeChunk(code="def sub(a, b): return a - b", path="/src/math.py", lang="python", start_line=2, end_line=2),
        ]
        embeddings = np.random.rand(2, 768).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        count = vector_store.insert_code_chunks("test_project", chunks, embeddings)
        assert count == 2

        query_vec = embeddings[0]
        results = vector_store.search_code(query_vec, "test_project", k=5)
        assert len(results) > 0
        assert results[0].path_or_source == "/src/math.py"

    def test_stale_chunks_replaced_on_reingest(self, vector_store):
        """Re-ingesting a file with fewer chunks should remove old ones."""
        chunks_v1 = [
            CodeChunk(code=f"line {i}", path="/src/file.py", lang="python", start_line=i, end_line=i)
            for i in range(5)
        ]
        emb_v1 = np.random.rand(5, 768).astype(np.float32)
        emb_v1 /= np.linalg.norm(emb_v1, axis=1, keepdims=True)
        vector_store.insert_code_chunks("proj", chunks_v1, emb_v1)

        # Re-ingest with only 2 chunks
        chunks_v2 = [
            CodeChunk(code=f"new line {i}", path="/src/file.py", lang="python", start_line=i, end_line=i)
            for i in range(2)
        ]
        emb_v2 = np.random.rand(2, 768).astype(np.float32)
        emb_v2 /= np.linalg.norm(emb_v2, axis=1, keepdims=True)
        vector_store.insert_code_chunks("proj", chunks_v2, emb_v2)

        # Only 2 chunks should remain for this file
        query = emb_v2[0]
        results = vector_store.search_code(query, "proj", k=10)
        file_results = [r for r in results if r.path_or_source == "/src/file.py"]
        assert len(file_results) == 2

    def test_project_isolation(self, vector_store):
        """Chunks from project A should not appear in project B searches."""
        chunk = CodeChunk(code="secret code", path="/src/secret.py", lang="python", start_line=1, end_line=1)
        emb = np.random.rand(1, 768).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        vector_store.insert_code_chunks("project_a", [chunk], emb)
        results = vector_store.search_code(emb[0], "project_b", k=10)
        assert len(results) == 0

    def test_lang_filter(self, vector_store):
        chunks = [
            CodeChunk(code="<?php echo 'hello'; ?>", path="/src/a.php", lang="php", start_line=1, end_line=1),
            CodeChunk(code="console.log('hello')", path="/src/b.ts", lang="typescript", start_line=1, end_line=1),
        ]
        emb = np.random.rand(2, 768).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        vector_store.insert_code_chunks("proj", chunks, emb)

        results = vector_store.search_code(emb[0], "proj", k=10, lang_filter=["php"])
        assert all(r.metadata["lang"] == "php" for r in results)


class TestDocChunkInsertSearch:
    def test_insert_and_search(self, vector_store):
        chunks = [
            DocChunk(content="Authentication overview for the API.", source="docs/auth.md", tags=["auth"]),
            DocChunk(content="Getting started with installation.", source="docs/setup.md", tags=["setup"]),
        ]
        embeddings = np.random.rand(2, 768).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        count = vector_store.insert_doc_chunks("proj", chunks, embeddings)
        assert count == 2

        results = vector_store.search_docs(embeddings[0], "proj", k=5)
        assert len(results) > 0

    def test_tags_preserved(self, vector_store):
        chunks = [DocChunk(content="API docs", source="api.md", tags=["api", "v2"])]
        emb = np.random.rand(1, 768).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        vector_store.insert_doc_chunks("proj", chunks, emb)
        results = vector_store.search_docs(emb[0], "proj", k=5)
        assert len(results) == 1
        assert "api" in results[0].metadata["tags"]


class TestDeleteProject:
    def test_delete_removes_all_chunks(self, vector_store):
        chunks = [
            CodeChunk(code="code here", path="/f.py", lang="python", start_line=1, end_line=1)
        ]
        emb = np.random.rand(1, 768).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        vector_store.insert_code_chunks("deleteme", chunks, emb)

        deleted = vector_store.delete_project("deleteme")
        assert deleted >= 1

        results = vector_store.search_code(emb[0], "deleteme", k=10)
        assert len(results) == 0

    def test_delete_nonexistent_project_returns_zero(self, vector_store):
        count = vector_store.delete_project("does_not_exist")
        assert count == 0
