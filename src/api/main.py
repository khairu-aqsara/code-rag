import logging
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI

from ..config import settings
from .deps import set_services

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, clean up on shutdown."""
    # Imported inside lifespan so torch/transformers are not loaded at module import time
    from ..embedder import EmbeddingService
    from ..vector_store import VectorStore

    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=False,  # Keep binary so embedding vectors come back as bytes
    )
    embedder = EmbeddingService()
    store = VectorStore(redis_client)
    set_services(embedder, store)
    logger.info("Startup complete: Redis connected, models loaded")

    yield

    redis_client.close()
    logger.info("Shutdown: Redis connection closed")


app = FastAPI(
    title="RAG Service",
    description="Local-first RAG service for code and documentation retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

# Import routers AFTER app is created — routers import from deps, not main (no circular import)
from .routers.search import search_router  # noqa: E402
from .routers.admin import admin_router   # noqa: E402

app.include_router(search_router, prefix="/api", tags=["search"])
app.include_router(admin_router, prefix="/api", tags=["admin"])
