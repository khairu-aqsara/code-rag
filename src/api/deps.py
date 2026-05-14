"""Dependency injection providers for FastAPI routes.

Exists to break the circular import between api/main.py and router files:
  main.py → deps.py ← routers/search.py
          → routers/admin.py → deps.py

EmbeddingService and VectorStore are NOT imported at module level to avoid
pulling in torch/transformers during test collection.
"""
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..embedder import EmbeddingService
    from ..vector_store import VectorStore

_embedding_service: Any = None
_vector_store: Any = None


def set_services(embedder: "EmbeddingService", store: "VectorStore") -> None:
    """Called once during app startup lifespan."""
    global _embedding_service, _vector_store
    _embedding_service = embedder
    _vector_store = store


def get_embedding_service() -> "EmbeddingService":
    if _embedding_service is None:
        raise RuntimeError("EmbeddingService not initialized — app not started?")
    return _embedding_service


def get_vector_store() -> "VectorStore":
    if _vector_store is None:
        raise RuntimeError("VectorStore not initialized — app not started?")
    return _vector_store
