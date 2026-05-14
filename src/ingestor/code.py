import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Dict, List, Optional

from ..chunker import CodeChunk, CodeChunker
from ..chunker_ast import ASTChunkerFactory
from ..config import settings

if TYPE_CHECKING:
    from ..embedder import EmbeddingService
    from ..vector_store import VectorStore

logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", "node_modules", "vendor", "build", "dist", "__pycache__", ".venv", "venv"}

LANG_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".php": "php",
    ".rb": "ruby",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
}


@dataclass
class IngestResult:
    total_files: int
    total_chunks: int
    duration_seconds: float
    errors: List[str] = field(default_factory=list)
    skipped_files: int = 0  # files skipped by filter patterns
    duplicate_chunks: int = 0  # chunks skipped due to deduplication


class CodeIngestor:
    """Walks a directory tree, chunks code files, embeds, and inserts into Redis."""

    def __init__(self, embedder: "EmbeddingService", vector_store: "VectorStore", summarizer=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.summarizer = summarizer
        self.line_chunker = CodeChunker()  # fallback for languages without AST support

    def _detect_lang(self, file_path: str) -> Optional[str]:
        ext = os.path.splitext(file_path)[1].lower()
        return LANG_EXTENSIONS.get(ext)

    def _should_skip_dir(self, dirname: str, skip_patterns: List[str]) -> bool:
        if dirname in SKIP_DIRS:
            return True
        return any(fnmatch(dirname, pat) for pat in skip_patterns)

    def _should_skip_file(self, filename: str) -> bool:
        """Check if file should be skipped based on SKIP_FILES patterns."""
        return any(fnmatch(filename, pat) for pat in settings.skip_files_patterns)

    def _get_summary(self, chunk: CodeChunk) -> str:
        """Get a summary for a code chunk — AI-generated if summarizer is available, else AST-based fallback."""
        if self.summarizer:
            fallback = " ".join(filter(None, [chunk.name, chunk.kind, chunk.docstring]))
            return self.summarizer.summarize(chunk.code, fallback=fallback or chunk.code[:200])
        parts = [p for p in [chunk.name, chunk.kind, chunk.docstring] if p]
        return " ".join(parts) if parts else chunk.code[:200]

    def _get_existing_summary(self, project_id: str, file_path: str, chunk: CodeChunk) -> str:
        """Check if a chunk already has a summary in Redis (for incremental re-ingestion)."""
        try:
            fhash = hashlib.md5(f"{project_id}:{file_path}".encode()).hexdigest()[:8]
            # We need to find the existing key — scan for it
            pattern = f"code:{project_id}:{fhash}:*"
            cursor = 0
            while True:
                cursor, keys = self.vector_store.redis.scan(cursor, match=pattern, count=50)
                for key in keys:
                    summary = self.vector_store.redis.hget(key, "summary")
                    if summary:
                        code = self.vector_store.redis.hget(key, "code")
                        if code and code.decode("utf-8") == chunk.code:
                            return summary.decode("utf-8")
                if cursor == 0:
                    break
        except Exception:
            pass
        return ""

    @staticmethod
    def _hash_chunk(chunk: CodeChunk) -> str:
        """Hash chunk content for deduplication."""
        return hashlib.md5(chunk.code.encode()).hexdigest()

    def ingest(
        self,
        project_id: str,
        root_path: str,
        lang_filter: Optional[List[str]] = None,
        skip_patterns: Optional[List[str]] = None,
    ) -> IngestResult:
        """Walk root_path, chunk and embed code files, insert into Redis.

        Args:
            project_id: Unique project identifier for Redis key namespacing.
            root_path: Absolute or relative root directory to walk.
            lang_filter: If set, only ingest files of these languages (e.g. ["python", "php"]).
            skip_patterns: Additional glob patterns for directories to skip.
        """
        skip_patterns = skip_patterns or []
        start_time = time.time()
        total_files = 0
        total_chunks = 0
        skipped_files = 0
        duplicate_chunks = 0
        errors: List[str] = []
        # Map chunk hash → canonical (first-seen) file path for original_path tracking
        seen_hashes: Dict[str, str] = {}

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Prune skip dirs in-place (modifies dirnames to prevent recursion)
            dirnames[:] = [
                d for d in dirnames
                if not self._should_skip_dir(d, skip_patterns)
            ]

            for filename in filenames:
                # Skip files matching SKIP_FILES patterns
                if self._should_skip_file(filename):
                    skipped_files += 1
                    continue

                file_path = os.path.join(dirpath, filename)
                lang = self._detect_lang(filename)

                if lang is None:
                    continue
                if lang_filter and lang not in lang_filter:
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception as e:
                    msg = f"Cannot read {file_path}: {e}"
                    logger.warning(msg)
                    errors.append(msg)
                    continue

                # Try AST-aware chunking first, fall back to line-based
                ast_chunker = ASTChunkerFactory.get_chunker(lang)
                if ast_chunker:
                    try:
                        chunks = ast_chunker.chunk_file(content, file_path)
                    except Exception as e:
                        logger.debug(f"AST chunking failed for {file_path}, falling back to line-based: {e}")
                        chunks = self.line_chunker.chunk_file(content, file_path, lang)
                else:
                    chunks = self.line_chunker.chunk_file(content, file_path, lang)

                if not chunks:
                    continue

                # Deduplicate chunks: skip if hash already seen, else record canonical path
                unique_chunks: List[CodeChunk] = []
                for chunk in chunks:
                    chunk_hash = self._hash_chunk(chunk)
                    if chunk_hash in seen_hashes:
                        duplicate_chunks += 1
                    else:
                        seen_hashes[chunk_hash] = chunk.path
                        # Stamp canonical original path on the chunk so VectorStore can persist it
                        chunk.original_path = chunk.path
                        unique_chunks.append(chunk)

                if not unique_chunks:
                    continue

                # Embed in batches
                try:
                    code_texts = [c.code for c in unique_chunks]
                    embeddings = self.embedder.embed_code_batch(code_texts)

                    # Generate summaries if summarizer is available
                    summaries = None
                    summary_embeddings = None
                    if self.summarizer:
                        summary_texts: list[str] = []
                        for chunk in unique_chunks:
                            # Mitigation 7: check if chunk already has a summary
                            existing = self._get_existing_summary(project_id, file_path, chunk)
                            if existing:
                                summary_texts.append(existing)
                            else:
                                summary_texts.append(self._get_summary(chunk))
                        summaries = summary_texts
                        summary_embeddings = self.embedder.embed_batch(summary_texts)

                    self.vector_store.insert_code_chunks(project_id, unique_chunks, embeddings, summaries=summaries, summary_embeddings=summary_embeddings)
                    total_files += 1
                    total_chunks += len(unique_chunks)

                    if total_files % 50 == 0:
                        logger.info(f"Progress: {total_files} files, {total_chunks} chunks, {duplicate_chunks} duplicates skipped")
                except Exception as e:
                    msg = f"Failed to embed/insert {file_path}: {e}"
                    logger.error(msg, exc_info=True)
                    errors.append(msg)

        duration = time.time() - start_time
        logger.info(
            f"Ingestion complete: {total_files} files, {total_chunks} chunks in {duration:.2f}s "
            f"({skipped_files} files skipped, {duplicate_chunks} duplicate chunks skipped, {len(errors)} errors)"
        )
        return IngestResult(
            total_files=total_files,
            total_chunks=total_chunks,
            duration_seconds=duration,
            errors=errors,
            skipped_files=skipped_files,
            duplicate_chunks=duplicate_chunks,
        )
