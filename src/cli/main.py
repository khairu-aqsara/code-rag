import logging
from contextlib import contextmanager
from typing import List, Optional

import redis
import typer

from ..config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="code-rag: local RAG service for code and documentation")


@contextmanager
def get_redis_and_store():
    """Initialize Redis + VectorStore only (fast, no model loading)."""
    from ..vector_store import VectorStore
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=False,
    )
    vector_store = VectorStore(redis_client)
    try:
        yield redis_client, vector_store
    finally:
        redis_client.close()


@contextmanager
def get_all_services():
    """Initialize Redis + VectorStore + EmbeddingService (slow: loads models ~10-30s)."""
    from ..embedder import EmbeddingService
    with get_redis_and_store() as (redis_client, vector_store):
        embedder = EmbeddingService()
        yield embedder, vector_store


@app.command()
def ingest_code(
    project_id: str = typer.Option(..., "--project", "-p", help="Project ID"),
    root_path: str = typer.Option(..., "--root", "-r", help="Root directory to scan"),
    lang_filter: Optional[List[str]] = typer.Option(None, "--lang", "-l", help="Language filter (e.g. python php)"),
    skip_patterns: Optional[List[str]] = typer.Option(None, "--skip", "-s", help="Glob patterns for dirs to skip"),
    summarize: bool = typer.Option(False, "--summarize", help="Enable AI-powered code summarization (loads codet5-small model)"),
):
    """Ingest code files from a directory into the vector store."""
    from ..ingestor.code import CodeIngestor

    try:
        with get_all_services() as (embedder, vector_store):
            summarizer = None
            if summarize:
                from ..summarizer import CodeSummarizer
                import psutil
                available_gb = psutil.virtual_memory().available / (1024 ** 3)
                if available_gb < 2.0:
                    logger.warning(
                        f"Only {available_gb:.1f}GB RAM available. --summarize loads ~550MB of models. "
                        f"Consider running without --summarize on this machine."
                    )
                summarizer = CodeSummarizer()
            ingestor = CodeIngestor(embedder, vector_store, summarizer=summarizer)
            result = ingestor.ingest(project_id, root_path, lang_filter, skip_patterns)
            typer.echo(f"✓ Ingested {result.total_files} files, {result.total_chunks} chunks in {result.duration_seconds:.2f}s")
            if result.errors:
                typer.echo(f"⚠ {len(result.errors)} errors (see logs for details)")
    except Exception as e:
        logger.error(f"Code ingestion failed: {e}", exc_info=True)
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def ingest_docs(
    project_id: str = typer.Option(..., "--project", "-p", help="Project ID"),
    file_paths: List[str] = typer.Option(..., "--paths", "-f", help="File paths or glob patterns (repeat for multiple)"),
    tags: Optional[List[str]] = typer.Option(None, "--tags", "-t", help="Tags to apply to all chunks"),
):
    """Ingest documentation files into the vector store."""
    from ..ingestor import DocIngestor

    try:
        with get_all_services() as (embedder, vector_store):
            ingestor = DocIngestor(embedder, vector_store)
            result = ingestor.ingest(project_id, file_paths, tags)
            typer.echo(f"✓ Ingested {result.total_files} files, {result.total_chunks} chunks in {result.duration_seconds:.2f}s")
            if result.errors:
                typer.echo(f"⚠ {len(result.errors)} errors (see logs for details)")
    except Exception as e:
        logger.error(f"Doc ingestion failed: {e}", exc_info=True)
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def delete_project(
    project_id: str = typer.Option(..., "--project", "-p", help="Project ID to delete"),
):
    """Delete all indexed data for a project (no model loading needed)."""
    try:
        with get_redis_and_store() as (_, vector_store):
            count = vector_store.delete_project(project_id)
            typer.echo(f"✓ Deleted {count} chunks for project '{project_id}'")
    except Exception as e:
        logger.error(f"Delete failed: {e}", exc_info=True)
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def stats():
    """Show RediSearch index statistics (no model loading needed)."""
    try:
        with get_redis_and_store() as (_, vector_store):
            index_stats = vector_store.get_stats()
            typer.echo(typer.style("Index Statistics", bold=True))
            for index_name, info in index_stats.items():
                typer.echo(f"\n{index_name}:")
                for key, val in info.items():
                    typer.echo(f"  {key}: {val}")
    except Exception as e:
        logger.error(f"Stats failed: {e}", exc_info=True)
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
