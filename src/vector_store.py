import logging
import time
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, List, Optional

import numpy as np
import redis

from ._search import SearchResult, parse_ft_search
from .chunker import CodeChunk, DocChunk
from .config import settings
from .ranker import SignalRanker

logger = logging.getLogger(__name__)

CODE_PREFIX = "code"
DOC_PREFIX = "doc"
CODE_INDEX = "idx:code"
DOC_INDEX = "idx:docs"
CODE_BM25_INDEX = "idx:code_bm25"
DOC_BM25_INDEX = "idx:docs_bm25"


class VectorStore:
    """Redis-backed vector store using RediSearch HNSW indices."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._create_indices()
        self.ranker = SignalRanker()

    # ── Index Management ─────────────────────────────────────────────────────

    def _create_code_index(self) -> None:
        try:
            self.redis.execute_command(
                "FT.CREATE", CODE_INDEX,
                "ON", "HASH", "PREFIX", "1", f"{CODE_PREFIX}:",
                "SCHEMA",
                "code", "TEXT", "NOINDEX",
                "path", "TAG",
                "lang", "TAG",
                "project_id", "TAG",
                "start_line", "NUMERIC",
                "end_line", "NUMERIC",
                "embedding", "VECTOR", "HNSW", "10",
                "TYPE", "FLOAT32", "DIM", str(settings.EMBED_DIM), "DISTANCE_METRIC", "COSINE",
                "M", str(settings.HNSW_M),
                "EF_CONSTRUCTION", str(settings.HNSW_EF_CONSTRUCTION),
                "has_summary", "TAG",
                "summary_embedding", "VECTOR", "HNSW", "10",
                "TYPE", "FLOAT32", "DIM", str(settings.EMBED_DIM), "DISTANCE_METRIC", "COSINE",
                "M", str(settings.HNSW_M),
                "EF_CONSTRUCTION", str(settings.HNSW_EF_CONSTRUCTION),
            )
            logger.info(f"Created index: {CODE_INDEX}")
        except Exception as e:
            if "Index already exists" in str(e):
                logger.debug(f"{CODE_INDEX} already exists")
            else:
                raise

    def _create_doc_index(self) -> None:
        try:
            self.redis.execute_command(
                "FT.CREATE", DOC_INDEX,
                "ON", "HASH", "PREFIX", "1", f"{DOC_PREFIX}:",
                "SCHEMA",
                "content", "TEXT", "NOINDEX",
                "source", "TAG",
                "project_id", "TAG",
                "tags", "TAG",
                "embedding", "VECTOR", "HNSW", "10",
                "TYPE", "FLOAT32", "DIM", str(settings.EMBED_DIM), "DISTANCE_METRIC", "COSINE",
                "M", str(settings.HNSW_M),
                "EF_CONSTRUCTION", str(settings.HNSW_EF_CONSTRUCTION),
            )
            logger.info(f"Created index: {DOC_INDEX}")
        except Exception as e:
            if "Index already exists" in str(e):
                logger.debug(f"{DOC_INDEX} already exists")
            else:
                raise

    def _create_code_bm25_index(self) -> None:
        try:
            self.redis.execute_command(
                "FT.CREATE", CODE_BM25_INDEX,
                "ON", "HASH", "PREFIX", "1", f"{CODE_PREFIX}:",
                "SCHEMA",
                "code", "TEXT",
                "path", "TAG",
                "lang", "TAG",
                "project_id", "TAG",
                "name", "TAG",
                "kind", "TAG",
            )
            logger.info(f"Created BM25 index: {CODE_BM25_INDEX}")
        except Exception as e:
            if "Index already exists" in str(e):
                logger.debug(f"{CODE_BM25_INDEX} already exists")
            else:
                raise

    def _create_doc_bm25_index(self) -> None:
        try:
            self.redis.execute_command(
                "FT.CREATE", DOC_BM25_INDEX,
                "ON", "HASH", "PREFIX", "1", f"{DOC_PREFIX}:",
                "SCHEMA",
                "content", "TEXT",
                "source", "TAG",
                "project_id", "TAG",
                "tags", "TAG",
            )
            logger.info(f"Created BM25 index: {DOC_BM25_INDEX}")
        except Exception as e:
            if "Index already exists" in str(e):
                logger.debug(f"{DOC_BM25_INDEX} already exists")
            else:
                raise

    def _create_indices(self) -> None:
        self._create_code_index()
        self._create_doc_index()
        self._create_code_bm25_index()
        self._create_doc_bm25_index()

    # ── Key Helpers ───────────────────────────────────────────────────────────

    def _escape_tag(self, value: str) -> str:
        """Escape hyphens in RediSearch TAG values (hyphens are parsed as subtraction)."""
        return value.replace("-", "\\-")

    @staticmethod
    def _sanitize_bm25_query(text: str) -> str:
        """Strip RediSearch query operators from user-supplied keyword text.

        Without this, a query like 'auth | drop' becomes a syntax error or
        unintended OR/exclusion. We keep letters, digits, underscores, dots,
        and whitespace — enough for function names, identifiers, and natural
        words — and collapse anything else to a space.
        """
        import re
        cleaned = re.sub(r"[^\w\s.]", " ", text)
        # Drop leading dashes per token (would be parsed as negation)
        tokens = [t.lstrip("-").lstrip(".") for t in cleaned.split() if t.strip(".-_")]
        return " ".join(tokens).strip()

    def _file_hash(self, project_id: str, identifier: str) -> str:
        """Deterministic 8-char hex hash — avoids Python's non-deterministic hash()."""
        return md5(f"{project_id}:{identifier}".encode()).hexdigest()[:8]

    def _make_key(self, prefix: str, project_id: str, file_hash: str, chunk_id: int) -> str:
        return f"{prefix}:{project_id}:{file_hash}:{chunk_id}"

    def _delete_file_chunks(self, prefix: str, project_id: str, file_hash: str) -> int:
        """Remove all chunks for a file before re-inserting (prevents stale data)."""
        pattern = f"{prefix}:{project_id}:{file_hash}:*"
        cursor = 0
        keys_to_delete: List[bytes] = []
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            keys_to_delete.extend(keys)
            if cursor == 0:
                break
        if keys_to_delete:
            pipe = self.redis.pipeline()
            for key in keys_to_delete:
                pipe.delete(key)
            pipe.execute()
        return len(keys_to_delete)

    # ── File fingerprinting (incremental ingest) ──────────────────────────────

    def _fp_key(self, project_id: str, file_hash: str) -> str:
        return f"meta:{project_id}:fp:{file_hash}"

    def get_file_fingerprint(self, project_id: str, file_path: str) -> Optional[str]:
        """Return the stored MD5 of a file's content, or None if not indexed."""
        fhash = self._file_hash(project_id, file_path)
        val = self.redis.get(self._fp_key(project_id, fhash))
        return val.decode() if val else None

    def set_file_fingerprint(self, project_id: str, file_path: str, content_md5: str) -> None:
        """Store the MD5 of a file's content after successful ingestion."""
        fhash = self._file_hash(project_id, file_path)
        self.redis.set(self._fp_key(project_id, fhash), content_md5)

    def delete_file_fingerprint(self, project_id: str, file_path: str) -> None:
        """Remove the stored fingerprint for a file (called on project delete)."""
        fhash = self._file_hash(project_id, file_path)
        self.redis.delete(self._fp_key(project_id, fhash))

    # ── Insert ────────────────────────────────────────────────────────────────

    # Flush the Redis pipeline every N inserts to cap the in-memory command buffer.
    _PIPELINE_FLUSH_EVERY = 100

    def insert_code_chunks(
        self, project_id: str, chunks: List[CodeChunk], embeddings: np.ndarray,
        summaries: Optional[List[str]] = None,
        summary_embeddings: Optional[np.ndarray] = None,
    ) -> int:
        """Upsert code chunks — deletes stale chunks per file before inserting."""
        file_groups: Dict[str, List[int]] = {}
        for idx, chunk in enumerate(chunks):
            file_groups.setdefault(chunk.path, []).append(idx)

        pipe = self.redis.pipeline()
        inserted = 0

        for file_path, indices in file_groups.items():
            fhash = self._file_hash(project_id, file_path)
            self._delete_file_chunks(CODE_PREFIX, project_id, fhash)
            for chunk_id, idx in enumerate(indices):
                chunk = chunks[idx]
                key = self._make_key(CODE_PREFIX, project_id, fhash, chunk_id)
                mapping = {
                    "code": chunk.code,
                    "path": chunk.path,
                    "lang": chunk.lang,
                    "project_id": project_id,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "embedding": embeddings[idx].astype(np.float32).tobytes(),
                }
                # Add semantic metadata if available
                if chunk.name:
                    mapping["name"] = chunk.name
                if chunk.kind:
                    mapping["kind"] = chunk.kind
                if chunk.docstring:
                    mapping["docstring"] = chunk.docstring
                if chunk.original_path and chunk.original_path != chunk.path:
                    mapping["original_path"] = chunk.original_path
                # Add summary fields if available
                if summaries is not None and idx < len(summaries) and summaries[idx]:
                    mapping["summary"] = summaries[idx]
                    mapping["has_summary"] = "1"
                if summary_embeddings is not None and idx < len(summary_embeddings):
                    mapping["summary_embedding"] = summary_embeddings[idx].astype(np.float32).tobytes()
                pipe.hset(key, mapping=mapping)
                inserted += 1
                if inserted % self._PIPELINE_FLUSH_EVERY == 0:
                    pipe.execute()
                    pipe = self.redis.pipeline()

        if inserted % self._PIPELINE_FLUSH_EVERY != 0:
            pipe.execute()
        self.redis.set(f"meta:{project_id}:last_indexed", int(time.time()))
        logger.debug(f"Inserted {inserted} code chunks for project '{project_id}'")
        return inserted

    def insert_doc_chunks(
        self, project_id: str, chunks: List[DocChunk], embeddings: np.ndarray
    ) -> int:
        """Upsert doc chunks — deletes stale chunks per source before inserting."""
        file_groups: Dict[str, List[int]] = {}
        for idx, chunk in enumerate(chunks):
            file_groups.setdefault(chunk.source, []).append(idx)

        pipe = self.redis.pipeline()
        inserted = 0

        for source, indices in file_groups.items():
            fhash = self._file_hash(project_id, source)
            self._delete_file_chunks(DOC_PREFIX, project_id, fhash)
            for chunk_id, idx in enumerate(indices):
                chunk = chunks[idx]
                key = self._make_key(DOC_PREFIX, project_id, fhash, chunk_id)
                tags_str = ",".join(chunk.tags) if chunk.tags else ""
                pipe.hset(key, mapping={
                    "content": chunk.content,
                    "source": chunk.source,
                    "project_id": project_id,
                    "tags": tags_str,
                    "embedding": embeddings[idx].astype(np.float32).tobytes(),
                })
                inserted += 1
                if inserted % self._PIPELINE_FLUSH_EVERY == 0:
                    pipe.execute()
                    pipe = self.redis.pipeline()

        if inserted % self._PIPELINE_FLUSH_EVERY != 0:
            pipe.execute()
        self.redis.set(f"meta:{project_id}:last_indexed", int(time.time()))
        logger.debug(f"Inserted {inserted} doc chunks for project '{project_id}'")
        return inserted

    def get_index_age(self, project_id: str) -> Optional[str]:
        """Return human-readable age of the last index update (e.g. '2d', '5h', '30m')."""
        timestamp_bytes = self.redis.get(f"meta:{project_id}:last_indexed")
        if not timestamp_bytes:
            return None
        try:
            ts = int(timestamp_bytes)
            age_seconds = int(time.time()) - ts
            if age_seconds < 60:
                return f"{age_seconds}s"
            elif age_seconds < 3600:
                return f"{age_seconds // 60}m"
            elif age_seconds < 86400:
                return f"{age_seconds // 3600}h"
            else:
                return f"{age_seconds // 86400}d"
        except Exception:
            return None

    # ── Search ────────────────────────────────────────────────────────────────

    def search_code_exact(
        self,
        query_text: str,
        project_id: str,
        k: int = 10,
        lang_filter: Optional[List[str]] = None,
        exclude_tests: bool = False,
        exclude_paths: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Keyword-only search over code using BM25 index — no embedding needed.

        Fast exact match for identifiers, function names, class names, etc.
        """
        fetch_k = k * 3
        pid = self._escape_tag(project_id)
        clean_query = self._sanitize_bm25_query(query_text)
        if not clean_query:
            return []

        conditions = [f"@project_id:{{{pid}}}"]
        if lang_filter:
            langs = "|".join(lang_filter)
            conditions.append(f"@lang:{{{langs}}}")
        query = " ".join(conditions) + f" @code:({clean_query})"

        try:
            raw = self.redis.execute_command(
                "FT.SEARCH", CODE_BM25_INDEX, query,
                "LIMIT", "0", str(fetch_k), "DIALECT", "2",
            )
            results = parse_ft_search(raw, "code")
        except Exception as e:
            logger.warning(f"BM25 code search failed for query {clean_query!r}: {e}")
            return []

        if exclude_tests:
            test_patterns = ["test_", "_test.py", ".test.", "conftest.py"]
            results = [r for r in results if not any(pat in r.path_or_source for pat in test_patterns)]

        if exclude_paths:
            results = [r for r in results if not any(r.path_or_source.startswith(p) for p in exclude_paths)]

        return results[:k]

    def search_code_summary(
        self,
        query_embedding: np.ndarray,
        project_id: str,
        k: int = 10,
        lang_filter: Optional[List[str]] = None,
        exclude_tests: bool = False,
        exclude_paths: Optional[List[str]] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search code using summary_embedding, falling back to embedding for chunks without summaries."""
        fetch_k = k * 3
        pid = self._escape_tag(project_id)

        # Search 1: summary_embedding (preferred — only chunks with has_summary=1)
        summary_query = f"(@project_id:{{{pid}}} @has_summary:{{1}})=>[KNN $k @summary_embedding $vec AS score]"
        if lang_filter:
            langs = "|".join(lang_filter)
            summary_query = f"(@project_id:{{{pid}}} @has_summary:{{1}} @lang:{{{langs}}})=>[KNN $k @summary_embedding $vec AS score]"

        results_summary: List[SearchResult] = []
        try:
            raw_summary = self.redis.execute_command(
                "FT.SEARCH", CODE_INDEX, summary_query,
                "PARAMS", "4", "vec", query_embedding.astype(np.float32).tobytes(), "k", str(fetch_k),
                "SORTBY", "score", "ASC", "DIALECT", "2",
            )
            results_summary = parse_ft_search(raw_summary, "code")
        except Exception as e:
            logger.warning(f"summary search failed: {e}")

        # Search 2: embedding (fallback — all chunks)
        fallback_query = f"(@project_id:{{{pid}}})=>[KNN $k @embedding $vec AS score]"
        if lang_filter:
            langs = "|".join(lang_filter)
            fallback_query = f"(@project_id:{{{pid}}} @lang:{{{langs}}})=>[KNN $k @embedding $vec AS score]"

        raw_fallback = self.redis.execute_command(
            "FT.SEARCH", CODE_INDEX, fallback_query,
            "PARAMS", "4", "vec", query_embedding.astype(np.float32).tobytes(), "k", str(fetch_k),
            "SORTBY", "score", "ASC", "DIALECT", "2",
        )
        results_fallback = parse_ft_search(raw_fallback, "code")

        # Merge: prefer summary results, fill remaining slots with fallback
        seen_keys: set = set()
        merged: List[SearchResult] = []
        for r in results_summary:
            key = f"{r.path_or_source}:{r.start_line}"
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(r)
        for r in results_fallback:
            key = f"{r.path_or_source}:{r.start_line}"
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(r)

        # Apply post-search filters
        if exclude_tests:
            test_patterns = ["test_", "_test.py", ".test.", "conftest.py"]
            merged = [r for r in merged if not any(pat in r.path_or_source for pat in test_patterns)]
        if exclude_paths:
            merged = [r for r in merged if not any(r.path_or_source.startswith(p) for p in exclude_paths)]
        if min_score is not None:
            merged = [r for r in merged if r.score >= min_score]

        return merged[:k]

    def search_code(
        self,
        query_embedding: np.ndarray,
        project_id: str,
        k: int = 10,
        lang_filter: Optional[List[str]] = None,
        path_filter: Optional[str] = None,
        exclude_tests: bool = False,
        exclude_paths: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        query_text: Optional[str] = None,
        rerank: bool = False,
    ) -> List[SearchResult]:
        """KNN search over code chunks with advanced filtering.

        Post-search filters (applied in Python): path_filter, exclude_tests, exclude_paths, min_score.
        Pre-search filters (RediSearch): lang_filter via TAG.
        """
        # Over-fetch to compensate for post-filtering
        post_filters_count = sum([bool(path_filter), exclude_tests, bool(exclude_paths), bool(min_score)])
        fetch_k = k * (3 if post_filters_count > 0 else 1)

        pid = self._escape_tag(project_id)
        if lang_filter:
            langs = "|".join(lang_filter)
            query = f"(@project_id:{{{pid}}} @lang:{{{langs}}})=>[KNN $k @embedding $vec AS score]"
        else:
            query = f"(@project_id:{{{pid}}})=>[KNN $k @embedding $vec AS score]"

        raw = self.redis.execute_command(
            "FT.SEARCH", CODE_INDEX, query,
            "PARAMS", "4", "vec", query_embedding.astype(np.float32).tobytes(), "k", str(fetch_k),
            "SORTBY", "score", "ASC", "DIALECT", "2",
        )
        results = parse_ft_search(raw, "code")

        # Apply post-search filters
        if path_filter:
            results = [r for r in results if r.path_or_source.startswith(path_filter)]

        if exclude_tests:
            # Skip files matching test patterns
            test_patterns = ["test_", "_test.py", ".test.", "conftest.py"]
            results = [r for r in results if not any(pat in r.path_or_source for pat in test_patterns)]

        if exclude_paths:
            # Skip paths in exclude list
            results = [r for r in results if not any(r.path_or_source.startswith(p) for p in exclude_paths)]

        if min_score is not None:
            # Filter by minimum similarity score
            results = [r for r in results if r.score >= min_score]

        # Optional re-ranking using signal-based scoring
        if rerank and query_text:
            results = self.ranker.rerank(results, query_text, embedding_weight=0.7)

        return results[:k]

    def search_docs(
        self,
        query_embedding: np.ndarray,
        project_id: str,
        k: int = 10,
        tags: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        query_text: Optional[str] = None,
        rerank: bool = False,
    ) -> List[SearchResult]:
        """KNN search over doc chunks with advanced filtering.

        Pre-search filters (RediSearch): tags via TAG.
        Post-search filters (applied in Python): exclude_sources, min_score.
        """
        fetch_k = k * (3 if exclude_sources or min_score is not None else 1)

        pid = self._escape_tag(project_id)
        if tags:
            tags_str = "|".join(tags)
            query = f"(@project_id:{{{pid}}} @tags:{{{tags_str}}})=>[KNN $k @embedding $vec AS score]"
        else:
            query = f"(@project_id:{{{pid}}})=>[KNN $k @embedding $vec AS score]"

        raw = self.redis.execute_command(
            "FT.SEARCH", DOC_INDEX, query,
            "PARAMS", "4", "vec", query_embedding.astype(np.float32).tobytes(), "k", str(fetch_k),
            "SORTBY", "score", "ASC", "DIALECT", "2",
        )
        results = parse_ft_search(raw, "doc")

        # Apply post-search filters
        if exclude_sources:
            results = [r for r in results if not any(r.path_or_source == s for s in exclude_sources)]

        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # Optional re-ranking using signal-based scoring
        if rerank and query_text:
            results = self.ranker.rerank(results, query_text, embedding_weight=0.7)

        return results[:k]

    def search_code_hybrid(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        project_id: str,
        k: int = 10,
        lang_filter: Optional[List[str]] = None,
        exclude_tests: bool = False,
        exclude_paths: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        semantic_weight: float = 0.3,
        rerank: bool = False,
    ) -> List[SearchResult]:
        """Hybrid search combining semantic (HNSW) + keyword (BM25) for code.

        semantic_weight (0.0-1.0): how much to weight semantic vs keyword
        - 0.0 = pure keyword
        - 0.5 = balanced
        - 1.0 = pure semantic
        """
        fetch_k = k * 3
        pid = self._escape_tag(project_id)

        # Semantic search
        if lang_filter:
            langs = "|".join(lang_filter)
            query = f"(@project_id:{{{pid}}} @lang:{{{langs}}})=>[KNN $k @embedding $vec AS score]"
        else:
            query = f"(@project_id:{{{pid}}})=>[KNN $k @embedding $vec AS score]"

        raw_semantic = self.redis.execute_command(
            "FT.SEARCH", CODE_INDEX, query,
            "PARAMS", "4", "vec", query_embedding.astype(np.float32).tobytes(), "k", str(fetch_k),
            "SORTBY", "score", "ASC", "DIALECT", "2",
        )
        semantic_results = parse_ft_search(raw_semantic, "code")

        # Keyword search (BM25) — target the code field explicitly and sanitize the
        # user query to avoid RediSearch operator injection (|, -, {, etc.).
        keyword_results: List[SearchResult] = []
        clean_query = self._sanitize_bm25_query(query_text)
        if clean_query and semantic_weight < 1.0:
            bm25_query = f"@project_id:{{{pid}}} @code:({clean_query})"
            try:
                raw_keyword = self.redis.execute_command(
                    "FT.SEARCH", CODE_BM25_INDEX, bm25_query,
                    "LIMIT", "0", str(fetch_k), "DIALECT", "2",
                )
                keyword_results = parse_ft_search(raw_keyword, "code")
            except Exception as e:
                logger.warning(f"BM25 code search failed for query {clean_query!r}: {e}")

        # Merge results by key, weighted by semantic_weight. Track which side(s)
        # contributed so callers can see match_type ("semantic"/"keyword"/"both").
        merged: Dict[str, tuple[float, SearchResult, set]] = {}

        for rank, result in enumerate(semantic_results):
            key = f"{result.path_or_source}:{result.start_line}"
            score = (1.0 - rank / (len(semantic_results) + 1)) * semantic_weight
            merged[key] = (score, result, {"semantic"})

        for rank, result in enumerate(keyword_results):
            key = f"{result.path_or_source}:{result.start_line}"
            keyword_score = (1.0 - rank / (len(keyword_results) + 1)) * (1.0 - semantic_weight)
            if key in merged:
                prev_score, prev_result, sources = merged[key]
                sources.add("keyword")
                merged[key] = (prev_score + keyword_score, prev_result, sources)
            else:
                merged[key] = (keyword_score, result, {"keyword"})

        # Stamp match_type onto each result's metadata for the API layer.
        for _score, result, sources in merged.values():
            if result.metadata is None:
                result.metadata = {}
            if sources == {"semantic"}:
                result.metadata["match_type"] = "semantic"
            elif sources == {"keyword"}:
                result.metadata["match_type"] = "keyword"
            else:
                result.metadata["match_type"] = "both"

        # Sort by combined score and copy that score onto each SearchResult so
        # downstream re-ranking and min_score filters see the fused value.
        sorted_results = sorted(merged.values(), key=lambda x: x[0], reverse=True)
        results: List[SearchResult] = []
        for score, result, _sources in sorted_results[:fetch_k]:
            result.score = score
            results.append(result)

        # Apply post-search filters
        if exclude_tests:
            test_patterns = ["test_", "_test.py", ".test.", "conftest.py"]
            results = [r for r in results if not any(pat in r.path_or_source for pat in test_patterns)]

        if exclude_paths:
            results = [r for r in results if not any(r.path_or_source.startswith(p) for p in exclude_paths)]

        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # Optional re-ranking using signal-based scoring
        if rerank:
            results = self.ranker.rerank(results, query_text, embedding_weight=0.7)

        return results[:k]

    def search_docs_hybrid(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        project_id: str,
        k: int = 10,
        tags: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        semantic_weight: float = 0.6,
        rerank: bool = False,
    ) -> List[SearchResult]:
        """Hybrid search combining semantic (HNSW) + keyword (BM25) for docs."""
        fetch_k = k * 3
        pid = self._escape_tag(project_id)

        # Semantic search
        if tags:
            tags_str = "|".join(tags)
            query = f"(@project_id:{{{pid}}} @tags:{{{tags_str}}})=>[KNN $k @embedding $vec AS score]"
        else:
            query = f"(@project_id:{{{pid}}})=>[KNN $k @embedding $vec AS score]"

        raw_semantic = self.redis.execute_command(
            "FT.SEARCH", DOC_INDEX, query,
            "PARAMS", "4", "vec", query_embedding.astype(np.float32).tobytes(), "k", str(fetch_k),
            "SORTBY", "score", "ASC", "DIALECT", "2",
        )
        semantic_results = parse_ft_search(raw_semantic, "doc")

        # Keyword search (BM25) — target the content field and sanitize input.
        keyword_results: List[SearchResult] = []
        clean_query = self._sanitize_bm25_query(query_text)
        if clean_query and semantic_weight < 1.0:
            bm25_query = f"@project_id:{{{pid}}} @content:({clean_query})"
            try:
                raw_keyword = self.redis.execute_command(
                    "FT.SEARCH", DOC_BM25_INDEX, bm25_query,
                    "LIMIT", "0", str(fetch_k), "DIALECT", "2",
                )
                keyword_results = parse_ft_search(raw_keyword, "doc")
            except Exception as e:
                logger.warning(f"BM25 doc search failed for query {clean_query!r}: {e}")

        # Merge results by source, tracking which side(s) contributed.
        merged: Dict[str, tuple[float, SearchResult, set]] = {}

        for rank, result in enumerate(semantic_results):
            key = result.path_or_source
            score = (1.0 - rank / (len(semantic_results) + 1)) * semantic_weight
            merged[key] = (score, result, {"semantic"})

        for rank, result in enumerate(keyword_results):
            key = result.path_or_source
            keyword_score = (1.0 - rank / (len(keyword_results) + 1)) * (1.0 - semantic_weight)
            if key in merged:
                prev_score, prev_result, sources = merged[key]
                sources.add("keyword")
                merged[key] = (prev_score + keyword_score, prev_result, sources)
            else:
                merged[key] = (keyword_score, result, {"keyword"})

        for _score, result, sources in merged.values():
            if result.metadata is None:
                result.metadata = {}
            if sources == {"semantic"}:
                result.metadata["match_type"] = "semantic"
            elif sources == {"keyword"}:
                result.metadata["match_type"] = "keyword"
            else:
                result.metadata["match_type"] = "both"

        # Sort by combined score and propagate fused score back onto results.
        sorted_results = sorted(merged.values(), key=lambda x: x[0], reverse=True)
        results: List[SearchResult] = []
        for score, result, _sources in sorted_results[:fetch_k]:
            result.score = score
            results.append(result)

        # Apply post-search filters
        if exclude_sources:
            results = [r for r in results if not any(r.path_or_source == s for s in exclude_sources)]

        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # Optional re-ranking using signal-based scoring
        if rerank:
            results = self.ranker.rerank(results, query_text, embedding_weight=0.7)

        return results[:k]

    # ── Delete & Stats ────────────────────────────────────────────────────────

    def delete_project(self, project_id: str) -> int:
        """Delete all code/doc chunks and file fingerprints for a project."""
        total = 0
        # Chunk data + file fingerprints all share the project_id namespace
        for prefix in [CODE_PREFIX, DOC_PREFIX, f"meta:{project_id}"]:
            pattern = f"{prefix}:{project_id}:*" if not prefix.startswith("meta:") else f"{prefix}:*"
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    pipe = self.redis.pipeline()
                    for key in keys:
                        pipe.delete(key)
                    pipe.execute()
                    total += len(keys)
                if cursor == 0:
                    break
        logger.info(f"Deleted {total} keys for project '{project_id}'")
        return total

    def get_stats(self) -> Dict:
        """Return summary statistics from both RediSearch indices."""
        stats: Dict = {}
        for index_name in [CODE_INDEX, DOC_INDEX]:
            try:
                info = self.redis.execute_command("FT.INFO", index_name)
                info_dict = {info[i]: info[i + 1] for i in range(0, len(info), 2)}
                stats[index_name] = {
                    "num_docs": info_dict.get(b"num_docs", 0),
                    "memory_in_bytes": info_dict.get(b"inverted_sz_mb", 0),
                    "indexing": info_dict.get(b"indexing", 0),
                }
            except Exception as e:
                logger.warning(f"Could not get stats for {index_name}: {e}")
                stats[index_name] = {"error": str(e)}
        return stats

    def find_symbols(
        self,
        project_id: str,
        name: Optional[str] = None,
        kind: Optional[str] = None,
        k: int = 10,
    ) -> List[SearchResult]:
        """Find symbol definitions by name/kind using TAG fields on BM25 index."""
        pid = self._escape_tag(project_id)
        conditions = [f"@project_id:{{{pid}}}"]
        if name:
            conditions.append(f"@name:{{{self._escape_tag(name)}}}")
        if kind:
            conditions.append(f"@kind:{{{self._escape_tag(kind)}}}")

        if len(conditions) == 1:
            query = f"{conditions[0]} @kind:{{function|class}}"
        else:
            query = " ".join(conditions)

        try:
            raw = self.redis.execute_command(
                "FT.SEARCH", CODE_BM25_INDEX, query,
                "LIMIT", "0", str(k), "DIALECT", "2",
            )
            results = parse_ft_search(raw, "code")
        except Exception as e:
            logger.warning(f"find_symbols failed for {project_id}: {e}")
            return []

        return results[:k]

    def list_project_files(self, project_id: str, prefix: str = "") -> List[str]:
        """Return unique file paths for a project using FT.AGGREGATE."""
        pid = self._escape_tag(project_id)
        conditions = [f"@project_id:{{{pid}}}"]
        if prefix:
            escaped_prefix = self._escape_tag(prefix)
            conditions.append(f"@path:{{{escaped_prefix}/*}}")
        query = " ".join(conditions)

        try:
            raw = self.redis.execute_command(
                "FT.AGGREGATE", CODE_INDEX, query,
                "GROUPBY", "1", "@path",
                "SORTBY", "1", "@path",
                "DIALECT", "2",
            )
            paths = []
            for row in raw[1:]:
                row_dict = {row[i]: row[i + 1] for i in range(0, len(row), 2)}
                path_bytes = row_dict.get(b"path")
                if path_bytes:
                    paths.append(path_bytes.decode("utf-8") if isinstance(path_bytes, bytes) else path_bytes)
            return paths
        except Exception as e:
            logger.warning(f"list_project_files failed for {project_id}: {e}")
            return []

    def list_projects(self) -> List["ProjectStats"]:
        """Return all projects that have indexed data, with per-project chunk counts.

        Uses FT.AGGREGATE GROUPBY on the TAG-indexed project_id field — one query
        per index (code + docs). Results are merged and sorted alphabetically.
        No keyspace SCAN required.
        """
        code_counts: Dict[str, int] = {}
        doc_counts: Dict[str, int] = {}

        for index_name, target in [(CODE_INDEX, code_counts), (DOC_INDEX, doc_counts)]:
            try:
                raw = self.redis.execute_command(
                    "FT.AGGREGATE", index_name,
                    "*",
                    "GROUPBY", "1", "@project_id",
                    "REDUCE", "COUNT", "0", "AS", "chunk_count",
                )
                # raw[0] is total result count; raw[1:] are result rows.
                # Each row is a flat list: [b"project_id", b"<value>", b"chunk_count", b"<n>"]
                for row in raw[1:]:
                    row_dict = {row[i]: row[i + 1] for i in range(0, len(row), 2)}
                    pid_bytes = row_dict.get(b"project_id")
                    count_bytes = row_dict.get(b"chunk_count", b"0")
                    if pid_bytes:
                        pid = pid_bytes.decode("utf-8") if isinstance(pid_bytes, bytes) else pid_bytes
                        count = int(count_bytes) if count_bytes else 0
                        target[pid] = count
            except Exception as e:
                logger.warning(f"list_projects: FT.AGGREGATE failed for {index_name}: {e}")

        all_ids = sorted(set(code_counts) | set(doc_counts))
        return [
            ProjectStats(
                project_id=pid,
                code_chunks=code_counts.get(pid, 0),
                doc_chunks=doc_counts.get(pid, 0),
            )
            for pid in all_ids
        ]

    def get_project_info(self, project_id: str) -> dict:
        """Return detailed information about a single project.

        Includes chunk counts, last_indexed timestamp, languages, and doc tags.
        """
        code_chunks = 0
        doc_chunks = 0
        languages = set()
        doc_tags = set()
        last_indexed = None

        # Get code chunks count and languages
        try:
            raw = self.redis.execute_command(
                "FT.AGGREGATE", CODE_INDEX,
                f"@project_id:{{{self._escape_tag(project_id)}}}",
                "GROUPBY", "1", "@project_id",
                "REDUCE", "COUNT", "0", "AS", "chunk_count",
            )
            if len(raw) > 1:
                code_chunks = int(raw[1].get(b"chunk_count", b"0"))
            
            # Get unique languages
            raw = self.redis.execute_command(
                "FT.AGGREGATE", CODE_INDEX,
                f"@project_id:{{{self._escape_tag(project_id)}}}",
                "GROUPBY", "1", "@lang",
            )
            for row in raw[1:]:
                row_dict = {row[i]: row[i + 1] for i in range(0, len(row), 2)}
                lang = row_dict.get(b"lang")
                if lang:
                    languages.add(lang.decode("utf-8") if isinstance(lang, bytes) else lang)
        except Exception as e:
            logger.warning(f"get_project_info: code index query failed: {e}")

        # Get doc chunks count and tags
        try:
            raw = self.redis.execute_command(
                "FT.AGGREGATE", DOC_INDEX,
                f"@project_id:{{{self._escape_tag(project_id)}}}",
                "GROUPBY", "1", "@project_id",
                "REDUCE", "COUNT", "0", "AS", "chunk_count",
            )
            if len(raw) > 1:
                doc_chunks = int(raw[1].get(b"chunk_count", b"0"))
            
            # Get unique tags
            raw = self.redis.execute_command(
                "FT.AGGREGATE", DOC_INDEX,
                f"@project_id:{{{self._escape_tag(project_id)}}}",
                "GROUPBY", "1", "@tags",
            )
            for row in raw[1:]:
                row_dict = {row[i]: row[i + 1] for i in range(0, len(row), 2)}
                tags = row_dict.get(b"tags")
                if tags:
                    doc_tags.add(tags.decode("utf-8") if isinstance(tags, bytes) else tags)
        except Exception as e:
            logger.warning(f"get_project_info: docs index query failed: {e}")

        # Get last_indexed timestamp
        try:
            ts = self.redis.get(f"meta:{project_id}:last_indexed")
            if ts:
                last_indexed = int(ts) if isinstance(ts, (int, bytes)) else ts
                if isinstance(last_indexed, bytes):
                    last_indexed = int(last_indexed.decode("utf-8"))
        except Exception as e:
            logger.warning(f"get_project_info: last_indexed query failed: {e}")

        # Calculate index age in days
        index_age_days = None
        if last_indexed:
            import time
            age_seconds = time.time() - last_indexed
            index_age_days = int(age_seconds / 86400)

        return {
            "project_id": project_id,
            "code_chunks": code_chunks,
            "doc_chunks": doc_chunks,
            "last_indexed": last_indexed,
            "index_age_days": index_age_days,
            "languages": sorted(languages),
            "doc_tags": sorted(doc_tags),
        }


@dataclass
class ProjectStats:
    """Per-project chunk counts returned by VectorStore.list_projects()."""

    project_id: str
    code_chunks: int
    doc_chunks: int
