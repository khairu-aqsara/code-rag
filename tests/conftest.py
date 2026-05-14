"""pytest fixtures for unit and integration tests.

Unit tests: use mock_embedder (deterministic fake embeddings) and mock_vector_store.
Integration tests: use redis_client fixture which spawns a real Redis Stack container
via testcontainers (requires Docker).
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.chunker import CodeChunk, DocChunk


# ── Shared data fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_code_chunk() -> CodeChunk:
    return CodeChunk(
        code="def hello():\n    return 'world'",
        path="/project/src/hello.py",
        lang="python",
        start_line=1,
        end_line=2,
    )


@pytest.fixture
def sample_doc_chunk() -> DocChunk:
    return DocChunk(
        content="This is a sample documentation paragraph.",
        source="README.md",
        tags=["docs", "intro"],
    )


# ── Unit test fixtures (no external deps) ────────────────────────────────────

@pytest.fixture
def mock_embedder():
    """Returns deterministic fake embeddings of the correct shape."""
    embedder = MagicMock()

    EMBED_DIM = 768  # gte-modernbert-base

    def _fake_embed(text: str) -> np.ndarray:
        """Single embedding function — same model for code and text."""
        rng = np.random.default_rng(seed=hash(text) % (2**31))
        v = rng.random(EMBED_DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def fake_embed_code(code: str) -> np.ndarray:
        """Alias — same as fake_embed, preserves existing call sites."""
        return _fake_embed(code)

    def fake_embed_text(text: str) -> np.ndarray:
        """Alias — same as fake_embed, preserves existing call sites."""
        return _fake_embed(text)

    def fake_embed(text: str) -> np.ndarray:
        """Unified embed — same as _fake_embed."""
        return _fake_embed(text)

    def fake_embed_code_batch(snippets):
        return np.stack([fake_embed_code(s) for s in snippets])

    def fake_embed_text_batch(texts):
        return np.stack([fake_embed_text(t) for t in texts])

    def fake_embed_batch(texts):
        return np.stack([fake_embed(t) for t in texts])

    embedder.embed.side_effect = fake_embed
    embedder.embed_code.side_effect = fake_embed_code
    embedder.embed_text.side_effect = fake_embed_text
    embedder.embed_code_batch.side_effect = fake_embed_code_batch
    embedder.embed_text_batch.side_effect = fake_embed_text_batch
    embedder.embed_batch.side_effect = fake_embed_batch

    return embedder


@pytest.fixture
def mock_vector_store():
    """VectorStore mock with preset return values."""
    store = MagicMock()
    store.insert_code_chunks.return_value = 5
    store.insert_doc_chunks.return_value = 3
    store.search_code.return_value = []
    store.search_docs.return_value = []
    store.delete_project.return_value = 0
    store.get_stats.return_value = {"idx:code": {}, "idx:docs": {}}
    store.get_index_age.return_value = None
    store.get_project_info.return_value = {"project_id": "test", "code_chunks": 0, "doc_chunks": 0, "last_indexed": None, "index_age_days": None, "languages": [], "doc_tags": []}
    return store


# ── Integration test fixtures (require Docker + Redis Stack) ─────────────────

@pytest.fixture(scope="session")
def redis_client():
    """Spawn a real Redis Stack container for integration tests.

    Requires Docker to be running. Marked with pytest.mark.integration.
    """
    pytest.importorskip("testcontainers", reason="testcontainers not installed")
    from testcontainers.core.container import DockerContainer

    container = DockerContainer("redis/redis-stack:latest")
    container.with_exposed_ports(6379)
    container.start()

    import redis as redis_lib
    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(6379))

    client = redis_lib.Redis(host=host, port=port, decode_responses=False)

    # Wait for Redis to be ready
    import time
    for _ in range(30):
        try:
            client.ping()
            break
        except Exception:
            time.sleep(0.5)

    yield client

    client.close()
    container.stop()


@pytest.fixture(autouse=False)
def flush_redis(redis_client):
    """Flush Redis before each integration test for isolation."""
    redis_client.flushall()
    yield
    redis_client.flushall()
