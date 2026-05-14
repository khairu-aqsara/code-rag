# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Project Overview

**code-rag** is a local-first Retrieval-Augmented Generation (RAG) service for indexing and searching code and documentation. It uses:
- **Redis Stack** for vector storage (HNSW indices with RediSearch)
- **gte-modernbert-base** (768-dim) for unified code and text embeddings
- **FastAPI** for REST API with Swagger UI at `/docs`
- **Typer** for CLI commands
- No paid APIs — all models are free, open-source Hugging Face models

## Development Commands

### Quick Start

```bash
# Start Redis Stack in Docker
docker-compose up -d redis

# Install dependencies (with test extras)
pip install -e ".[test]"

# Copy and configure environment (defaults work for local dev)
cp .env.example .env

# Start API server (DO NOT use — see CLAUDE.md global instructions)
# Instead, use docker-compose for local testing
docker-compose up
```

### Testing

```bash
# Unit tests only (no external deps — fast)
pytest -m "not integration" -v

# Run specific test file
pytest tests/test_embedder.py -v

# Run single test
pytest tests/test_chunker.py::TestCodeChunker::test_chunk_overlap -v

# Integration tests (requires Docker + Redis Stack running)
pytest -m integration -v

# All tests with coverage
pytest --cov=src
```

### CLI Commands

```bash
# Show all available commands
python -m src.cli --help

# Ingest code from a directory
python -m src.cli ingest-code --project myproject --root ./src --lang python

# Ingest documentation files (single path)
python -m src.cli ingest-docs --project myproject --paths "docs/*.md" --tags api auth

# Ingest documentation files (multiple paths)
python -m src.cli ingest-docs --project myproject --paths "docs/*.md" --paths "README.md" --tags api

# Delete all indexed data for a project
python -m src.cli delete-project --project myproject

# Show index statistics (fast — no model loading)
python -m src.cli stats
```

## Architecture

### High-Level Flow

```
User Request (API or CLI)
    ↓
[VectorStore] ← Redis (RediSearch HNSW indices)
    ↓
[EmbeddingService] ← gte-modernbert-base (768-dim, unified code + text)
    ↓
[Ingestors] ← CodeIngestor | DocIngestor
    ↓
[Chunkers] ← CodeChunker | DocChunker
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/config.py` | Pydantic settings (env vars, defaults) |
| `src/embedder.py` | EmbeddingService: loads gte-modernbert model, converts text to vectors |
| `src/vector_store.py` | VectorStore: manages Redis HNSW + BM25 indices, semantic & hybrid search, re-ranking |
| `src/chunker.py` | CodeChunker, DocChunker: split files into overlapping chunks |
| `src/chunker_ast.py` | **Phase 1**: PythonASTChunker, JavaScriptASTChunker — extract semantic units (functions/classes) |
| `src/ranker.py` | **Phase 5**: SignalRanker — re-rank by non-test files, definitions, query term matches |
| `src/ingestor/code.py` | CodeIngestor: AST chunking, file filtering, MD5 deduplication (**Phases 1-2**) |
| `src/ingestor/docs.py` | DocIngestor: ingests markdown/txt/rst/html files |
| `src/_search.py` | SearchResult, parse_ft_search: parse RediSearch output with semantic metadata |
| `src/api/main.py` | FastAPI app, lifespan context, service initialization |
| `src/api/deps.py` | Dependency injection — stores service singletons (avoids circular imports) |
| `src/api/schemas.py` | Pydantic request/response models with filters and metadata (**Phase 3**) |
| `src/api/routers/search.py` | Search endpoints: semantic, hybrid (keyword+semantic), re-ranking (**Phases 3-5**) |
| `src/api/routers/admin.py` | Admin endpoints: ingest, delete, health, stats |
| `src/cli/main.py` | Typer CLI commands |

### Design Decisions

1. **Phase 1 — AST-aware chunking**: Extract functions/classes as semantic units (not arbitrary lines) to preserve context and enable better search.

2. **Phase 2 — Content-based deduplication**: Hash chunks by MD5 to skip identical code in multiple files; track canonical path.

3. **Phase 3 — Rich metadata in responses**: Extract and return function names, docstrings, kinds from AST so Claude Code has semantic context without extra embedding calls.

4. **Phase 4 — Dual indices (HNSW + BM25)**: Both semantic (vector similarity) and keyword (exact match) searches run in parallel; merge with configurable `semantic_weight`.

5. **Phase 5 — Signal-based re-ranking**: Boost results by relevance signals (non-test file +1, definition +1, query term match +1) combined with embedding score.

6. **`api/deps.py` breaks circular imports**: API routers import service singletons from `deps.py`, not `main.py`. This prevents `main.py` → `routers` → `main.py` cycles.

7. **Redis keys are MD5-deterministic**: Chunks keyed by `hash(project_id + source + chunk_id)` so re-ingestion doesn't create duplicates.

8. **Lazy model loading**: CLI commands like `stats` and `delete` avoid loading embedding models (~5-15s startup) by using only `VectorStore` + Redis.

9. **Binary Redis client** (`decode_responses=False`): Embedding vectors must remain bytes. Fields are decoded explicitly where needed.

10. **Reciprocal Rank Fusion (RRF)**: Since code and docs now share the same 768-dim embedding space, scores are comparable, but RRF is still used for merging different content types.

11. **`SCAN` cursor iteration** (never `KEYS`): Non-blocking Redis iteration for long-running operations.

12. **Fast index creation on startup**: Both HNSW and BM25 indices created if missing; existing indices silently skipped.

## Configuration

All settings come from environment variables or `.env`:

| Variable | Default | Notes |
|----------|---------|-------|
| `REDIS_HOST` | `localhost` | |
| `REDIS_PORT` | `6379` | |
| `REDIS_PASSWORD` | (none) | Optional |
| `EMBED_MODEL` | `Alibaba-NLP/gte-modernbert-base` | gte-modernbert, 768-dim |
| `EMBED_DIM` | `768` | Embedding vector dimension |
| `EMBED_MAX_LENGTH` | `2048` | Max tokens per embedding (model max 8192) |
| `EMBED_BATCH_SIZE` | `32` | Batch size for code and text embeddings |
| `CODE_CHUNK_LINES` | `80` | Lines per code chunk |
| `CODE_OVERLAP_LINES` | `20` | Overlap for code chunks (context preservation) |
| `DOC_CHUNK_SIZE` | `512` | Approx tokens per doc chunk |
| `DEFAULT_TOP_K` | `10` | Default search result count |
| `MAX_TOP_K` | `50` | Maximum allowed K in requests |
| `SKIP_FILES` | `*.test.py,*_test.py,test_*.py,conftest.py,*.min.js,*.min.css,*.map` | **Phase 2**: File patterns to skip during ingestion |
| `HNSW_M` | `16` | HNSW graph branching factor |
| `HNSW_EF_CONSTRUCTION` | `200` | HNSW construction parameter |
| `HNSW_EF_RUNTIME` | `10` | HNSW search parameter |
| `API_HOST` | `0.0.0.0` | |
| `API_PORT` | `8000` | |
| `BASE_PATH` | `/data` | Security: all ingest paths must be under this |

## Testing Strategy

### Unit Tests (no external deps)
- Mock `EmbeddingService` and `VectorStore`
- Located in `tests/test_*.py`
- Marked with `@pytest.mark.unit`
- Fast (~1s total)

### Integration Tests
- Spin up real Redis Stack container via `testcontainers`
- Marked with `@pytest.mark.integration`
- Slower (~30s total) due to container startup
- Test actual Redis operations, embedding pipeline end-to-end

### Conftest
`tests/conftest.py` provides:
- `redis_fixture()`: real Redis container for integration tests
- `mock_embedder()`: embedder stub for unit tests
- `mock_store()`: vector store stub for unit tests

## API Patterns

### Request/Response Models
All Pydantic models are in `src/api/schemas.py`:
- `CodeSearchRequest`, `DocsSearchRequest`, `HybridSearchRequest` → `SearchResponse`
- `IngestCodeRequest`, `IngestDocsRequest` → `IngestResponse`

### Dependency Injection
FastAPI endpoint functions use `get_embedder()` and `get_store()` dependencies from `deps.py`:

```python
@router.post("/search-code")
async def search_code(
    req: CodeSearchRequest,
    embedder: EmbeddingService = Depends(get_embedder),
    store: VectorStore = Depends(get_store),
):
    ...
```

## API Endpoints (Phases 3-5)

### Search Endpoints

| Endpoint | Method | Purpose | Phase |
|----------|--------|---------|-------|
| `/api/search-code` | POST | Semantic code search with filters | Phase 3 |
| `/api/search-docs` | POST | Semantic doc search with filters | Phase 3 |
| `/api/search-hybrid` | POST | Merges code + docs via RRF | Phase 3 |
| `/api/search-code-hybrid` | POST | Keyword + semantic code search (BM25 + HNSW) | Phase 4 |
| `/api/search-docs-hybrid` | POST | Keyword + semantic doc search (BM25 + HNSW) | Phase 4 |

### Request Parameters (Phases 3-5)

All search endpoints accept:

```python
{
  "project_id": str,                    # Required: project identifier
  "query": str,                         # Required: search query
  "k": int,                             # Default: 10, Max: 50
  
  # Phase 3: Filtering
  "exclude_tests": bool,                # Default: True (skip test files)
  "exclude_paths": List[str],           # e.g., ["migrations/", "vendor/"]
  "min_score": float,                   # e.g., 0.7 (relevance threshold)
  
  # Phase 3-4: Search tuning
  "semantic_weight": float,             # Default: 0.6 (0.0 = keyword, 1.0 = semantic)
  
  # Code-specific (optional)
  "lang_filter": List[str],             # e.g., ["python", "typescript"]
  "path_prefix": str,                   # (Phase 3 only)
  
  # Doc-specific (optional)
  "tags": List[str],                    # e.g., ["api", "auth"]
  "exclude_sources": List[str],         # e.g., ["CHANGELOG.md"]
}
```

### Response Format (Phase 3)

All search endpoints return:

```python
{
  "results": [
    {
      "score": float,                   # Similarity or combined score
      "path_or_source": str,            # File path or doc source
      "content": str,                   # Code/doc content
      "start_line": int | null,         # Code only
      "end_line": int | null,           # Code only
      
      # Phase 3: Rich semantic metadata (from AST)
      "name": str | null,               # Function/class name
      "kind": str | null,               # "function", "class", "import"
      "docstring": str | null,          # Extracted docstring
      "language": str | null,           # Programming language
      
      "metadata": dict                  # Full metadata dict
    }
  ]
}
```

## Common Tasks

### Using Phase 4: Hybrid Keyword + Semantic Search
1. Call `/api/search-code-hybrid` instead of `/api/search-code`
2. Set `semantic_weight` based on query type:
   - `0.0-0.3` for exact name matches (e.g., "authenticate_user")
   - `0.5` for balanced search
   - `0.7-1.0` for concept search (e.g., "user authentication flow")
3. Both HNSW (semantic) and BM25 (keyword) indices queried in parallel; results merged

### Using Phase 5: Signal-Based Re-Ranking
Re-ranking is optional and triggered at VectorStore level. Higher-level APIs automatically use it when beneficial. Signals:
- **Non-test file** (+1.0): `path_or_source` doesn't contain test patterns
- **Definition** (+1.0): `kind` is "function" or "class" (not import/comment)
- **Query match** (+1.0): Query terms appear in `name` or `docstring`

Re-ranked score = `normalized_embedding * embedding_weight + normalized_signals * (1 - embedding_weight)`

### Adding a New Search Endpoint
1. Define request/response Pydantic models in `src/api/schemas.py`
2. Add endpoint to `src/api/routers/search.py` (uses `get_embedder()`, `get_store()`)
3. Implement using `store.search_code()`, `store.search_docs()`, `store.search_code_hybrid()`, or `store.search_docs_hybrid()`
4. Add test in `tests/test_api_search.py` or `tests/test_hybrid_search.py`

### Enabling/Disabling Re-Ranking
Re-ranking is integrated into all search methods via optional `rerank=True` parameter:
```python
results = vector_store.search_code(
    query_embedding=embedding,
    project_id="myproj",
    query_text="authenticate",
    rerank=True,  # Enable signal-based re-ranking
)
```

### Adjusting Chunking Strategy
1. For AST-aware chunking (Phase 1): Edit `src/chunker_ast.py` extraction logic
2. For line-based fallback: Modify `CodeChunker.chunk_file()` in `src/chunker.py`
3. Update chunk settings in `src/config.py` (CODE_CHUNK_LINES, CODE_OVERLAP_LINES)
4. Re-ingest: `python -m src.cli delete-project --project test && python -m src.cli ingest-code --project test --root ./src --lang python`

### Adjusting Skip Patterns (Phase 2)
Edit `SKIP_FILES` in `.env` or `src/config.py`:
```ini
SKIP_FILES=*.test.py,*_test.py,test_*.py,conftest.py,*.min.js,*.min.css,*.map
```

### Modifying Embedding Models
1. Update `EMBED_MODEL_TEXT` or `EMBED_MODEL_CODE` in `.env`
2. Update vector dimensions in `src/vector_store.py` (search for `DIM` in index creation)
3. Test with unit tests (mock) first, then integration tests (real models)

### Debugging Search Quality
- Use RedisInsight at `http://localhost:8001` to inspect indices
- Query Redis directly: `redis-cli` → `FT.INFO idx:code` (HNSW) or `FT.INFO idx:code_bm25` (BM25)
- Check semantic metadata: `HGET code:proj:hash:0 name` returns function name (Phase 3)
- Profile signal scores: Enable logging in `src/ranker.py` to see ranking signals (Phase 5)

## AI Coding Agent Integration

### Search Tool Selection Guide

| Query Type | Recommended Tool | Why |
|------------|------------------|-----|
| Exact function/class name | `coderag_search_code_exact` or `coderag_find_symbols` | BM25 is faster and more precise for identifiers |
| Exact variable/module name | `coderag_find_symbols` with `kind=variable` | Uses TAG index on name field |
| Concept/intent ("how does auth work") | `coderag_search_code_hybrid` with `semantic_weight=0.7` | Semantic search finds related code |
| "Where is X implemented?" | `coderag_search_code_exact` | Fast exact match |
| "Explain the auth flow" | `coderag_search_hybrid` | Cross-modal (code + docs) |
| Technical question about docs | `coderag_search_docs` or `coderag_search_docs_hybrid` | Searches documentation index |
| Don't know where answer is | `coderag_search_hybrid` | Searches both code and docs |
| Need file structure | `coderag_list_files` | Returns indexed file paths |

### Query Pattern Examples

- Find exact function: `coderag_find_symbols(project_id="myproject", name="authenticate_user", kind="function")`
- Find all classes in a module: `coderag_find_symbols(project_id="myproject", kind="class", k=20)`
- Concept search: `coderag_search_code_hybrid(project_id="myproject", query="user authentication flow", semantic_weight=0.7)`
- Fast keyword search: `coderag_search_code_exact(project_id="myproject", query="getUserById")`

### Default Project Configuration

For single-project setups, use the default project configuration to skip project confirmation:

- `coderag_set_default_project(workspace, project_id)` — Configure default project
- `coderag_get_default_project(workspace)` — Check current default

---

## Important Notes

- **Model Download**: First run downloads ~300 MB of HF models to `~/.cache/huggingface/`. Subsequent runs use the cache.
- **Redis Persistence**: By default, Redis Stack in Docker is in-memory. For persistence, enable RDB/AOF in `docker-compose.yml`.
- **Security**: `BASE_PATH` setting restricts ingest paths. All paths are validated at ingestion endpoints.
- **Performance**: HNSW parameters (M, EF) tune search speed vs. accuracy. See `src/config.py` for defaults.
