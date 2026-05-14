#!/usr/bin/env python3
"""
MCP Server for code-rag.

Exposes all code-rag search and admin endpoints as MCP tools. Acts as a
stateless HTTP proxy — all ML inference and vector storage remain in the
code-rag app container at APP_URL. This container loads no ML models.

Tools provided:
  Search (read-only):
    - coderag_search_code           Semantic code search (HNSW / gte-modernbert)
    - coderag_search_code_hybrid    Keyword + semantic code search (BM25 + HNSW)
    - coderag_search_code_exact     Fast exact/keyword code search (BM25 only, no embedding)
    - coderag_search_docs           Semantic doc search  (HNSW / gte-modernbert)
    - coderag_search_docs_hybrid    Keyword + semantic doc search  (BM25 + HNSW)
    - coderag_search_hybrid         Cross-modal RRF search (code + docs)
    - coderag_search_batch          Execute multiple searches in one request

  Discovery (read-only):
    - coderag_list_projects         List all ingested projects with chunk counts
    - coderag_list_files            List indexed file paths for a project
    - coderag_find_symbols          Find symbol definitions by name/kind
    - coderag_get_default_project   Get default project_id for a workspace
    - coderag_get_project_info      Get detailed info about a project

  Configuration (write):
    - coderag_set_default_project   Set default project_id for a workspace

  Admin (read-only):
    - coderag_get_health            Health check (Redis + models)
    - coderag_get_stats             RediSearch index statistics
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_URL: str = os.environ.get("APP_URL", "http://localhost:8000")
MCP_PORT: int = int(os.environ.get("MCP_PORT", "8002"))
MCP_HOST: str = os.environ.get("MCP_HOST", "0.0.0.0")

# Valid field names accepted by the app's field-projection parameter.
_VALID_FIELDS = (
    "score, path_or_source, content, start_line, end_line, "
    "name, kind, docstring, language, match_type, original_path, last_indexed"
)

# ---------------------------------------------------------------------------
# Module-level HTTP client — shared across all requests
# ---------------------------------------------------------------------------

_http_client = httpx.AsyncClient(
    base_url=APP_URL,
    timeout=httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0),
    headers={"Content-Type": "application/json"},
)

mcp = FastMCP(
    "code_rag_mcp",
    host=MCP_HOST,
    port=MCP_PORT,
    stateless_http=True,
)

# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------


async def _api_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST to the code-rag app and return parsed JSON."""
    response = await _http_client.post(path, content=json.dumps(body))
    response.raise_for_status()
    return response.json()


async def _api_put(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """PUT to the code-rag app and return parsed JSON."""
    response = await _http_client.put(path, content=json.dumps(body))
    response.raise_for_status()
    return response.json()


async def _api_get(path: str) -> dict[str, Any]:
    """GET from the code-rag app and return parsed JSON."""
    response = await _http_client.get(path)
    response.raise_for_status()
    return response.json()


def handle_error(e: Exception) -> str:
    """Return a clear, actionable error string for any exception."""
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 404:
            return (
                "Error 404: Resource not found. "
                "Check that the project_id is correct and the project has been ingested.\n\n"
                "Suggestion: Call coderag_list_projects to see available projects."
            )
        if code == 422:
            try:
                detail = e.response.json().get("detail", e.response.text)
            except Exception:
                detail = e.response.text
            return f"Error 422: Invalid request parameters — {detail}"
        if code == 503:
            return (
                "Error 503: The code-rag app is unavailable. "
                "Check that the 'app' container is running and models have finished loading "
                "(first startup downloads ~300 MB of HF models — check coderag_get_health)."
            )
        return f"Error {code}: App returned an unexpected status — {e.response.text[:200]}"
    if isinstance(e, httpx.ConnectError):
        return (
            f"Connection error: Cannot reach the code-rag app at {APP_URL}. "
            "Ensure the 'app' container is running."
        )
    if isinstance(e, httpx.TimeoutException):
        return (
            "Timeout: The request took too long. "
            "The app may still be loading ML models (first startup can take 30s+). "
            "Try coderag_get_health to check readiness."
        )
    # Generic error with suggestions for common issues
    error_msg = str(e)
    if "no results" in error_msg.lower() or "empty" in error_msg.lower():
        return (
            f"Error: {error_msg}\n\n"
            "Suggestions:\n"
            "- Try coderag_search_code_exact for exact name matching\n"
            "- Lower min_score threshold (e.g., 0.5)\n"
            "- Check exclude_tests / exclude_paths filters"
        )
    return f"Unexpected error ({type(e).__name__}): {error_msg}"


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    """Remove keys whose value is None so the app uses its own defaults."""
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Annotated type aliases for flat tool kwargs
# (FastMCP generates flat inputSchema from function signatures directly)
# ---------------------------------------------------------------------------

ProjectId = Annotated[str, Field(description="Project identifier used when code/docs were ingested (e.g. 'myproject'). Call coderag_list_projects to find valid values.", min_length=1, max_length=200)]
Query = Annotated[str, Field(description="Search query string.", min_length=1, max_length=1000)]
K = Annotated[Optional[int], Field(default=None, description="Number of results to return (1–20). Default: 10.", ge=1, le=20)]
LangFilter = Annotated[Optional[list[str]], Field(default=None, description="Restrict to these languages, e.g. ['python', 'typescript'].")]
PathPrefix = Annotated[Optional[str], Field(default=None, description="Only return results whose file path starts with this prefix.")]
ExcludeTests = Annotated[Optional[bool], Field(default=None, description="Skip test files (default: true). Set false to include them.")]
ExcludePaths = Annotated[Optional[list[str]], Field(default=None, description="Exclude results whose path contains any of these substrings, e.g. ['migrations/', 'vendor/'].")]
MinScore = Annotated[Optional[float], Field(default=None, description="Minimum relevance score threshold (0.0–1.0). Results below this are dropped.", ge=0.0, le=1.0)]
Rerank = Annotated[Optional[bool], Field(default=None, description="Enable signal-based re-ranking (boosts non-test files, definitions, and query-term matches). Default: false.")]
Fields = Annotated[Optional[list[str]], Field(default=None, description=f"Return only these fields to save tokens. Valid values: {_VALID_FIELDS}. Example: ['path_or_source','start_line','end_line','name'].")]
SemanticWeight = Annotated[Optional[float], Field(default=None, description="Balance between keyword (BM25) and semantic (HNSW) search. 0.0–0.3: keyword-dominant (exact name lookup). 0.5: balanced. 0.7–1.0: semantic-dominant (concept search). Default: 0.3 for code, 0.6 for docs, 0.5 for hybrid.", ge=0.0, le=1.0)]
Tags = Annotated[Optional[list[str]], Field(default=None, description="Filter to docs ingested with all of these tags, e.g. ['api', 'auth'].")]
ExcludeSources = Annotated[Optional[list[str]], Field(default=None, description="Exclude docs whose source path matches any of these strings, e.g. ['CHANGELOG.md'].")]


# ---------------------------------------------------------------------------
# Tools — search
# ---------------------------------------------------------------------------


@mcp.tool(
    name="coderag_search_code",
    annotations={
        "title": "Search Code (Semantic)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_search_code(
    project_id: ProjectId,
    query: Query,
    k: K = None,
    lang_filter: LangFilter = None,
    path_prefix: PathPrefix = None,
    exclude_tests: ExcludeTests = None,
    exclude_paths: ExcludePaths = None,
    min_score: MinScore = None,
    rerank: Rerank = None,
    fields: Fields = None,
    use_summary: bool = None,
) -> str:
    """Search indexed code using semantic (vector) similarity via gte-modernbert embeddings.

    Queries the HNSW vector index for code chunks whose meaning is closest to the
    query. Best for concept/intent queries like "user authentication flow" or
    "database connection pooling". For exact function/class name lookup, prefer
    coderag_search_code_hybrid with a low semantic_weight.

    Args:
        use_summary: When True, searches against AI-generated summary embeddings
            (requires chunks ingested with --summarize). Better for concept queries;
            falls back to raw code embedding for chunks without summaries.

    Returns JSON {"results": [...]} where each result may include:
    score, path_or_source, content, start_line, end_line, name, kind, docstring, language.
    Use the fields parameter to limit which fields are returned and save tokens.
    """
    try:
        body = _drop_none({
            "project_id": project_id,
            "query": query,
            "k": k,
            "lang_filter": lang_filter,
            "path_prefix": path_prefix,
            "exclude_tests": exclude_tests,
            "exclude_paths": exclude_paths,
            "min_score": min_score,
            "rerank": rerank,
            "fields": fields,
            "use_summary": use_summary,
            "response_type": "json",
        })
        result = await _api_post("/api/search-code", body)
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_search_code_hybrid",
    annotations={
        "title": "Search Code (Hybrid: Keyword + Semantic)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_search_code_hybrid(
    project_id: ProjectId,
    query: Query,
    k: K = None,
    semantic_weight: SemanticWeight = None,
    lang_filter: LangFilter = None,
    exclude_tests: ExcludeTests = None,
    exclude_paths: ExcludePaths = None,
    min_score: MinScore = None,
    rerank: Rerank = None,
    fields: Fields = None,
) -> str:
    """Search indexed code using both BM25 keyword and HNSW semantic search, merged by score.

    Superior to pure semantic search when the query contains exact identifiers
    (function names, class names, module paths). Use semantic_weight to tune:
    - 0.0–0.3: keyword-dominant — exact name lookup (e.g. "authenticate_user")
    - 0.5: balanced
    - 0.7–1.0: semantic-dominant — concept/intent queries

    The default semantic_weight for code search is 0.3 (keyword-dominant), which
    is appropriate for most code queries. Pass an explicit value to override.

    Returns JSON {"results": [...]} with optional match_type field ("semantic", "keyword", or "both").
    """
    try:
        body = _drop_none({
            "project_id": project_id,
            "query": query,
            "k": k,
            "semantic_weight": semantic_weight,
            "lang_filter": lang_filter,
            "exclude_tests": exclude_tests,
            "exclude_paths": exclude_paths,
            "min_score": min_score,
            "rerank": rerank,
            "fields": fields,
            "response_type": "json",
        })
        result = await _api_post("/api/search-code-hybrid", body)
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_search_docs",
    annotations={
        "title": "Search Documentation (Semantic)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_search_docs(
    project_id: ProjectId,
    query: Query,
    k: K = None,
    tags: Tags = None,
    exclude_sources: ExcludeSources = None,
    min_score: MinScore = None,
    rerank: Rerank = None,
    fields: Fields = None,
) -> str:
    """Search indexed documentation using semantic similarity via gte-modernbert embeddings.

    Queries the HNSW documentation vector index. Best for conceptual questions
    against prose documentation, READMEs, API references, and markdown files.

    Returns JSON {"results": [...]}. Use tags to filter by ingestion tags,
    exclude_sources to skip specific files, and fields to limit response size.
    """
    try:
        body = _drop_none({
            "project_id": project_id,
            "query": query,
            "k": k,
            "tags": tags,
            "exclude_sources": exclude_sources,
            "min_score": min_score,
            "rerank": rerank,
            "fields": fields,
            "response_type": "json",
        })
        result = await _api_post("/api/search-docs", body)
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_search_docs_hybrid",
    annotations={
        "title": "Search Documentation (Hybrid: Keyword + Semantic)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_search_docs_hybrid(
    project_id: ProjectId,
    query: Query,
    k: K = None,
    semantic_weight: SemanticWeight = None,
    tags: Tags = None,
    exclude_sources: ExcludeSources = None,
    min_score: MinScore = None,
    rerank: Rerank = None,
    fields: Fields = None,
) -> str:
    """Search indexed documentation using both BM25 keyword and HNSW semantic search.

    Combines keyword and semantic search for docs. Useful when the query contains
    specific technical terms, configuration keys, or CLI command names.
    Use semantic_weight to tune: 0.0–0.3 for exact term lookup, 0.7–1.0 for concepts.

    Returns JSON {"results": [...]}.
    """
    try:
        body = _drop_none({
            "project_id": project_id,
            "query": query,
            "k": k,
            "semantic_weight": semantic_weight,
            "tags": tags,
            "exclude_sources": exclude_sources,
            "min_score": min_score,
            "rerank": rerank,
            "fields": fields,
            "response_type": "json",
        })
        result = await _api_post("/api/search-docs-hybrid", body)
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_search_hybrid",
    annotations={
        "title": "Search Code + Docs (Cross-modal RRF)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_search_hybrid(
    project_id: ProjectId,
    query: Query,
    k: K = None,
    exclude_tests: ExcludeTests = None,
    exclude_paths: ExcludePaths = None,
    min_score: MinScore = None,
    tags: Tags = None,
    exclude_sources: ExcludeSources = None,
    rerank: Rerank = None,
    fields: Fields = None,
) -> str:
    """Search across both code and documentation simultaneously using Reciprocal Rank Fusion.

    Queries both indices using gte-modernbert (768-dim) in parallel. Results are merged
    via RRF which ranks by position rather than raw score — this produces better results
    when searching across different content types (code and docs).

    Use when unsure whether the answer lives in code or docs, or when the query
    spans both — e.g. "How is JWT authentication implemented and documented?".

    Returns JSON {"results": [...]} interleaving code and doc results ranked by RRF score.
    Check path_or_source to distinguish file paths (code) from doc source identifiers.
    """
    try:
        body = _drop_none({
            "project_id": project_id,
            "query": query,
            "k": k,
            "exclude_tests": exclude_tests,
            "exclude_paths": exclude_paths,
            "min_score": min_score,
            "tags": tags,
            "exclude_sources": exclude_sources,
            "rerank": rerank,
            "fields": fields,
            "response_type": "json",
        })
        result = await _api_post("/api/search-hybrid", body)
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


# ---------------------------------------------------------------------------
# Tools — project discovery
# ---------------------------------------------------------------------------


@mcp.tool(
    name="coderag_search_code_exact",
    annotations={
        "title": "Search Code (Exact/Keyword)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_search_code_exact(
    project_id: ProjectId,
    query: Query,
    k: K = None,
    lang_filter: LangFilter = None,
    exclude_tests: ExcludeTests = None,
    exclude_paths: ExcludePaths = None,
    fields: Fields = None,
) -> str:
    """Search code by exact keyword match using BM25 — no embedding needed.

    Use this when you know the exact function name, class name, or identifier.
    Faster than semantic search for exact matches. For conceptual queries,
    use coderag_search_code or coderag_search_code_hybrid instead.

    Returns JSON {"results": [...]}.
    """
    try:
        body = _drop_none({
            "project_id": project_id,
            "query": query,
            "k": k,
            "lang_filter": lang_filter,
            "exclude_tests": exclude_tests,
            "exclude_paths": exclude_paths,
            "fields": fields,
            "response_type": "json",
        })
        result = await _api_post("/api/search-code-exact", body)
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_list_files",
    annotations={
        "title": "List Project Files",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_list_files(
    project_id: ProjectId,
    prefix: str = "",
) -> str:
    """List all indexed file paths for a project.

    Returns the file tree of indexed code. Useful for understanding project
    structure. Optional prefix filter narrows results to a directory.

    Returns JSON {"project_id": str, "files": [...], "total": int}.
    """
    try:
        path = f"/api/projects/{project_id}/files"
        if prefix:
            path += f"?prefix={prefix}"
        result = await _api_get(path)
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_find_symbols",
    annotations={
        "title": "Find Symbol Definitions",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_find_symbols(
    project_id: ProjectId,
    name: Optional[str] = None,
    kind: Optional[str] = None,
    k: K = None,
    fields: Fields = None,
) -> str:
    """Find symbol definitions by name and kind.

    Searches the BM25 index for code chunks matching the given name/kind.
    Useful for finding all occurrences of a function or class.

    Args:
        project_id: Project identifier
        name: Symbol name to search for (e.g. "authenticate_user")
        kind: Symbol kind filter ("function", "class", or empty for any)
        k: Number of results to return
        fields: Fields to include in response

    Returns JSON {"results": [...]}.
    """
    try:
        body = _drop_none({
            "project_id": project_id,
            "name": name,
            "kind": kind,
            "k": k,
            "fields": fields,
            "response_type": "json",
        })
        result = await _api_post("/api/find-symbols", body)
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    annotations={
        "title": "List Available Projects",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_list_projects() -> str:
    """List all projects that have indexed data, with per-project chunk counts.

    Call this first to discover which project_ids are available before running
    any search tool. Each entry shows how many code chunks and doc chunks are
    indexed for that project, so you can tell at a glance whether a project has
    code, documentation, or both.

    Returns:
        str: JSON string with the following schema:
            {
                "total": int,          # Number of distinct projects
                "projects": [
                    {
                        "project_id": str,   # Use this value in search tools
                        "code_chunks": int,  # Chunks in the code HNSW index
                        "doc_chunks": int    # Chunks in the docs HNSW index
                    }
                ]
            }

        Projects are sorted alphabetically by project_id.
        An empty list means no projects have been ingested yet.

    Examples:
        - Before searching: call this to find valid project_ids
        - "Which projects have documentation?" → check doc_chunks > 0
        - "What is indexed for project X?" → look up code_chunks + doc_chunks
    """
    try:
        result = await _api_get("/api/projects")
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_set_default_project",
    annotations={
        "title": "Set Default Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_set_default_project(workspace: str, project_id: str) -> str:
    """Set the default project_id for a workspace.

    Use this to configure a default project so you don't need to specify
    project_id on every search call. Useful for single-project setups.

    Args:
        workspace: Workspace identifier (e.g., hostname, workspace name)
        project_id: Project identifier to set as default

    Returns:
        str: JSON confirmation with workspace and project_id
    """
    try:
        body = {"workspace": workspace, "project_id": project_id}
        result = await _api_put("/api/config/default-project", body)
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_get_default_project",
    annotations={
        "title": "Get Default Project",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_get_default_project(workspace: str) -> str:
    """Get the default project_id for a workspace.

    Returns the configured default project_id, or null if not set.
    Use this to check if a default is configured before searching.

    Args:
        workspace: Workspace identifier (e.g., hostname, workspace name)

    Returns:
        str: JSON with workspace and project_id (null if not set)
    """
    try:
        result = await _api_get(f"/api/config/default-project?workspace={workspace}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_get_project_info",
    annotations={
        "title": "Get Project Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_get_project_info(project_id: str) -> str:
    """Get detailed information about a project.

    Returns chunk counts, last_indexed timestamp, programming languages,
    and documentation tags for the specified project.

    Args:
        project_id: Project identifier

    Returns:
        str: JSON with project statistics:
            {
                "project_id": str,
                "code_chunks": int,
                "doc_chunks": int,
                "last_indexed": str | null,
                "index_age_days": int | null,
                "languages": [str],
                "doc_tags": [str]
            }
    """
    try:
        result = await _api_get(f"/api/projects/{project_id}/info")
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


# ---------------------------------------------------------------------------
# Tools — admin / health
# ---------------------------------------------------------------------------


@mcp.tool(
    name="coderag_get_health",
    annotations={
        "title": "Get Health Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_get_health() -> str:
    """Check the health of the code-rag service (Redis connection and ML model status).

    Returns the readiness of the app container. Call this before running searches
    if you suspect the service is not ready — particularly on first startup, where
    downloading and loading the ~300 MB gte-modernbert model can take
    10–60 seconds.

    Returns:
        str: JSON string with the following schema:
            {
                "status": "ok" | "error",
                "redis_ok": bool,     # True if Redis is reachable and responding
                "models_loaded": bool # True if gte-modernbert is loaded
            }

If models_loaded is false, wait and retry — searches will fail until
    the model is ready.
    """
    try:
        result = await _api_get("/api/health")
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_get_stats",
    annotations={
        "title": "Get Index Statistics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_get_stats() -> str:
    """Retrieve RediSearch index statistics for all four indices.

    Returns document counts, memory usage, and index configuration for:
      - idx:code      — HNSW semantic index for code (gte-modernbert, 768-dim, COSINE)
      - idx:docs      — HNSW semantic index for docs  (gte-modernbert, 768-dim, COSINE)
      - idx:code_bm25 — BM25 full-text index for code
      - idx:docs_bm25 — BM25 full-text index for docs

    Use this to verify that a project has been ingested (num_docs > 0) or to
    understand how many chunks are indexed across all projects.

    Returns:
        str: JSON string with {"indices": {"idx:code": {...}, "idx:docs": {...}, ...}}
             Each index entry contains RediSearch FT.INFO fields including
             num_docs, num_terms, max_doc_id, indexing, and memory usage.
    """
    try:
        result = await _api_get("/api/stats")
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="coderag_search_batch",
    annotations={
        "title": "Batch Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def coderag_search_batch(
    project_id: ProjectId,
    queries: List[Dict[str, Any]],
    k: K = None,
    fields: Fields = None,
) -> str:
    """Execute multiple search queries in a single request.

    Useful for AI agents that need to run several related searches efficiently.
    Each query is executed independently and results are returned keyed by query index.

    Args:
        project_id: Project identifier
        queries: List of query objects. Each query must have:
            - type: "code", "code_exact", "code_hybrid", "docs", or "docs_hybrid"
            - query: Search query string
            - semantic_weight: Optional float (0.0-1.0) for hybrid searches
            - lang_filter: Optional list of languages (code searches)
            - tags: Optional list of tags (doc searches)
            - exclude_tests: Optional bool (default: true)
            - min_score: Optional float (0.0-1.0)
        k: Number of results per query (default: 10)
        fields: Optional list of fields to return

    Returns:
        str: JSON with results keyed by query index:
            {
                "results": {
                    "query_0": [...],
                    "query_1": [...],
                    ...
                }
            }

    Example:
        coderag_search_batch(
            project_id="myproject",
            queries=[
                {"type": "code_exact", "query": "authenticate_user"},
                {"type": "code_hybrid", "query": "auth flow", "semantic_weight": 0.7},
                {"type": "docs", "query": "authentication setup"}
            ],
            k=5
        )
    """
    try:
        body = _drop_none({
            "project_id": project_id,
            "queries": queries,
            "k": k,
            "fields": fields,
            "response_type": "json",
        })
        result = await _api_post("/api/search-batch", body)
        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_error(e)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
