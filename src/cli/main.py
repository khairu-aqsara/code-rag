import logging
import os

# Suppress tqdm progress bars from huggingface_hub and transformers during model downloads.
# Must be set before any HF imports so the library picks it up at import time.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import redis
import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from rich.logging import RichHandler

from ..config import settings

_console = Console()

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler(console=_console, show_path=False, show_time=False, show_level=False, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def _build_session_panel(embedder, redis_client, project_id: str, root_path: str, summarizer=None) -> Panel:
    """Build a rich Panel summarising model, Redis, and job info."""
    import torch

    # ── Model info ────────────────────────────────────────────────────────────
    device_label = embedder.device.upper()
    if embedder.device == "mps":
        device_label = "MPS (Apple Silicon)"
    elif embedder.device == "cuda":
        try:
            device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:
            device_label = "CUDA"

    model_table = Table.grid(padding=(0, 2))
    model_table.add_column(style="bold dim")
    model_table.add_column()
    model_table.add_row("Model", embedder.model_name.split("/")[-1])
    model_table.add_row("Full name", embedder.model_name)
    model_table.add_row("Device", device_label)
    model_table.add_row("Size", f"{embedder.model_size_mb:.0f} MB")
    model_table.add_row("Embed dim", str(settings.EMBED_DIM))
    model_table.add_row("Batch size", str(settings.EMBED_BATCH_SIZE))
    if summarizer is not None:
        sum_device = summarizer.device.upper()
        if summarizer.device == "mps":
            sum_device = "MPS (Apple Silicon)"
        elif summarizer.device == "cuda":
            sum_device = "CUDA"
        model_table.add_row("Summarizer", f"codet5-small  [{sum_device}]")

    # ── Redis info ─────────────────────────────────────────────────────────────
    redis_table = Table.grid(padding=(0, 2))
    redis_table.add_column(style="bold dim")
    redis_table.add_column()
    try:
        info = redis_client.info("server")
        used_mem = redis_client.info("memory").get("used_memory_human", "?")
        redis_table.add_row("Host", f"{settings.REDIS_HOST}:{settings.REDIS_PORT}")
        redis_table.add_row("Version", info.get("redis_version", "?"))
        redis_table.add_row("Memory used", used_mem)
        redis_status = Text("● connected", style="bold green")
    except Exception:
        redis_table.add_row("Host", f"{settings.REDIS_HOST}:{settings.REDIS_PORT}")
        redis_status = Text("● unreachable", style="bold red")

    # ── Job info ───────────────────────────────────────────────────────────────
    job_table = Table.grid(padding=(0, 2))
    job_table.add_column(style="bold dim")
    job_table.add_column()
    job_table.add_row("Project", project_id)
    job_table.add_row("Root", root_path)

    redis_title = Text()
    redis_title.append("Redis  ", style="bold")
    redis_title.append_text(redis_status)

    left = Panel(model_table, title="[bold]Embedding Model[/bold]", border_style="blue", expand=True)
    mid = Panel(
        redis_table,
        title=redis_title,
        border_style="blue",
        expand=True,
    )
    right = Panel(job_table, title="[bold]Job[/bold]", border_style="blue", expand=True)

    return Panel(Columns([left, mid, right], equal=True), border_style="dim")

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
                summarizer = CodeSummarizer()

            ingestor = CodeIngestor(embedder, vector_store, summarizer=summarizer)

            # Clear screen, print header + session info panel
            _console.clear()
            _console.rule("[bold]code-rag  ·  ingest-code[/bold]")
            _console.print(_build_session_panel(embedder, vector_store.redis, project_id, root_path, summarizer))

            # Pre-scan to count eligible files for accurate ETA (fast — no file reads)
            skip_pats = skip_patterns or []
            _console.print("Scanning files...", style="dim")
            total_eligible = 0
            for dirpath, dirnames, filenames in os.walk(root_path):
                dirnames[:] = [
                    d for d in dirnames
                    if not ingestor._should_skip_dir(d, skip_pats)
                ]
                for f in filenames:
                    if ingestor._should_skip_file(f):
                        continue
                    lang = ingestor._detect_lang(f)
                    if lang is None:
                        continue
                    if lang_filter and lang not in lang_filter:
                        continue
                    total_eligible += 1
            _console.print(f"Found [bold]{total_eligible}[/bold] files to ingest.\n")

            # Fixed-width progress bar: description column is padded to a constant width
            # so the bar never shifts as filenames change.
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description:<50}", style="cyan"),
                BarColumn(bar_width=30),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("ETA"),
                TimeRemainingColumn(),
                console=_console,
                transient=False,
            ) as progress:
                task = progress.add_task("Waiting...", total=total_eligible)

                def on_file(file_path: str, chunks: int) -> None:
                    name = Path(file_path).name
                    # Truncate to fit the fixed 50-char description column
                    label = name[:47] + "..." if len(name) > 50 else name
                    progress.update(task, advance=1, description=label)

                result = ingestor.ingest(
                    project_id, root_path, lang_filter, skip_patterns,
                    progress_callback=on_file,
                )

            _console.print()
            total_processed = result.total_files + result.skipped_unchanged
            if result.skipped_unchanged and result.total_files == 0:
                summary = f"[bold green]✓[/bold green] Nothing changed — all [bold]{total_processed}[/bold] files up to date  [dim]{result.duration_seconds:.1f}s[/dim]"
            elif result.skipped_unchanged:
                summary = (
                    f"[bold green]✓[/bold green] [bold]{total_processed}[/bold] files checked in {result.duration_seconds:.1f}s  "
                    f"[bold]{result.total_files}[/bold] updated ([bold]{result.total_chunks}[/bold] chunks)  "
                    f"[dim]{result.skipped_unchanged} unchanged[/dim]"
                )
            else:
                summary = (
                    f"[bold green]✓[/bold green] Ingested [bold]{result.total_files}[/bold] files, "
                    f"[bold]{result.total_chunks}[/bold] chunks in {result.duration_seconds:.1f}s"
                )
            if result.duplicate_chunks:
                summary += f"  [dim]{result.duplicate_chunks} duplicates skipped[/dim]"
            _console.print(summary)
            if result.errors:
                _console.print(f"[yellow]⚠ {len(result.errors)} errors[/yellow] (run with --verbose for details)")
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
