import glob
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from ..chunker import DocChunker

if TYPE_CHECKING:
    from ..embedder import EmbeddingService
    from ..vector_store import VectorStore

logger = logging.getLogger(__name__)

DOC_EXTENSIONS = {".md", ".txt", ".rst", ".html"}

# HTML tag stripping pattern
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s{3,}")


@dataclass
class IngestResult:
    total_files: int
    total_chunks: int
    duration_seconds: float
    errors: List[str] = field(default_factory=list)


class DocIngestor:
    """Loads documentation files, chunks, embeds, and inserts into Redis."""

    def __init__(self, embedder: "EmbeddingService", vector_store: "VectorStore"):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = DocChunker()

    def _clean_content(self, content: str, file_path: str) -> str:
        """Strip HTML tags and normalize whitespace."""
        if file_path.endswith(".html"):
            content = _HTML_TAG_RE.sub(" ", content)
        content = _WHITESPACE_RE.sub("\n\n", content)
        return content.strip()

    def _expand_paths(self, file_paths: List[str]) -> List[str]:
        """Expand glob patterns and filter to supported extensions."""
        expanded: List[str] = []
        for pattern in file_paths:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(pattern)  # treat as literal path
        return [
            p for p in expanded
            if any(p.endswith(ext) for ext in DOC_EXTENSIONS)
        ]

    def ingest(
        self,
        project_id: str,
        file_paths: List[str],
        tags: Optional[List[str]] = None,
    ) -> IngestResult:
        """Load, chunk, embed, and insert documentation files.

        Args:
            project_id: Unique project identifier for Redis key namespacing.
            file_paths: List of file paths or glob patterns.
            tags: Optional tags applied to all chunks from this ingest run.
        """
        start_time = time.time()
        total_files = 0
        total_chunks = 0
        errors: List[str] = []

        expanded = self._expand_paths(file_paths)
        logger.info(f"Ingesting {len(expanded)} doc files for project '{project_id}'")

        for file_path in expanded:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_content = f.read()
            except Exception as e:
                msg = f"Cannot read {file_path}: {e}"
                logger.warning(msg)
                errors.append(msg)
                continue

            content = self._clean_content(raw_content, file_path)
            if not content:
                logger.debug(f"Skipping empty file: {file_path}")
                continue

            chunks = self.chunker.chunk_doc(content, source=file_path, tags=tags)
            if not chunks:
                continue

            try:
                texts = [c.content for c in chunks]
                embeddings = self.embedder.embed_text_batch(texts)
                self.vector_store.insert_doc_chunks(project_id, chunks, embeddings)
                total_files += 1
                total_chunks += len(chunks)
            except Exception as e:
                msg = f"Failed to embed/insert {file_path}: {e}"
                logger.error(msg, exc_info=True)
                errors.append(msg)

        duration = time.time() - start_time
        logger.info(
            f"Doc ingestion complete: {total_files} files, {total_chunks} chunks "
            f"in {duration:.2f}s ({len(errors)} errors)"
        )
        return IngestResult(
            total_files=total_files,
            total_chunks=total_chunks,
            duration_seconds=duration,
            errors=errors,
        )
