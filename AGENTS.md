# AGENTS.md

Agent guidance for the `code-rag` repository. Every item here answers: "Would an agent miss this without help?"

---

## Critical constraints

- **Never start the dev server** (`uvicorn`, `npm run dev`, `docker-compose up` for the full stack). Use Docker only to spin up Redis.
- Integration tests auto-manage their own Redis container via `testcontainers` — don't start Redis manually for tests.
- The global `CLAUDE.md` instruction "never run development server / never run npm run dev" applies here.
---

## Setup

```bash
# Redis Stack only (required for integration tests and local API use)
docker-compose up -d redis

# Install with test extras
pip install -e ".[test]"

# Copy env (defaults work for local dev)
cp .env.example .env
```

First run downloads ~300 MB of HF models to `~/.cache/huggingface/`.

---

## Running tests

```bash
# Unit tests — fast, no Docker, no model loading (~1s)
pytest -m "not integration" -v

# Single test file
pytest tests/test_embedder.py -v

# Single test
pytest tests/test_chunker.py::TestCodeChunker::test_chunk_overlap -v

# Integration tests — requires Docker running (spawns its own Redis Stack container)
pytest -m integration -v

# All tests with coverage
pytest --cov=src
```

Unit tests mock both the embedder and vector store — they never touch Redis or load torch/transformers.

---

## CLI commands

```bash
python -m src.cli ingest-code --project myproject --root ./src --lang python
python -m src.cli ingest-docs --project myproject --paths "docs/*.md" --tags api auth
python -m src.cli delete-project --project myproject
python -m src.cli stats          # fast — skips model loading
```

`stats` and `delete-project` skip the 10–30s model-load startup.

---

## Architecture notes

**Module boundaries that matter:**

| File | Role |
|------|------|
- `src/api/deps.py` | DI singletons — routers import from here, **not** from `main.py` (breaks circular import) |
| `src/api/main.py` | Imports routers *after* `app = FastAPI(...)` for the same reason |
| `src/embedder.py` | Lazily imported inside lifespan / CLI context managers — never at module level (avoids torch import on test collection); single unified gte-modernbert model with CLS pooling |
| `src/vector_store.py` | `decode_responses=False` on the Redis client — embedding vectors must stay as bytes; fields decoded explicitly |
| `src/chunker_ast.py` | Phase 1 AST chunking (preferred path); `src/chunker.py` is line-based fallback |
| `src/ranker.py` | Signal-based re-ranker (Phase 5) — boost non-test files, definitions, query-term matches |

**Indices (Redis):**
- `idx:code` — HNSW, 768-dim COSINE (gte-modernbert)
- `idx:docs` — HNSW, 768-dim COSINE (gte-modernbert)
- `idx:code_bm25` / `idx:docs_bm25` — BM25 full-text for hybrid search

**Redis key scheme:** `MD5(project_id + source + chunk_id)` — deterministic, re-ingestion is idempotent.

**Hybrid search scoring:** Code (768-dim) and doc (768-dim) scores are comparable via the unified embedding model. `/search-hybrid` uses Reciprocal Rank Fusion (RRF) to merge different content types.

---

## Configuration quirks

- `BASE_PATH` (default `/data`) gates all ingest API endpoints — paths outside it are rejected. In `docker-compose.yml` it's overridden to `/workspace`.
- `SKIP_FILES` is comma-separated glob patterns applied during code ingestion (not at search time).
- `MAX_TOP_K` defaults to `20` in `.env.example` and `config.py`, but README says `50` — **trust `config.py`**.
- `decode_responses=False` is intentional on the Redis client — don't change it.

---

## Adding search endpoints

1. Define Pydantic models in `src/api/schemas.py`
2. Add endpoint to `src/api/routers/search.py` using `get_embedding_service()` and `get_vector_store()` from `deps.py`
3. Endpoints return `{"results": serialized_items}` (dict, not Pydantic response model) so that `response_type` and `fields` filtering can be applied via `_serialize_items()`
4. Add tests in `tests/test_api_search.py` or `tests/test_search_filters.py`

---

## Response optimization

All search endpoints support two token-saving parameters:

- `response_type`: `"json"` (default) omits null fields and `metadata` dict; `"json-full"` includes all fields for backward compatibility
- `fields`: list of field names to project — only requested fields appear in response (valid: `score`, `path_or_source`, `content`, `start_line`, `end_line`, `name`, `kind`, `docstring`, `language`, `match_type`, `original_path`)

Implementation: `_serialize_items()` in `routers/search.py` applies `exclude_none` + field projection after `_to_item()` converts `SearchResult` → `SearchResultItem`.

---

## Testing conventions

- `@pytest.mark.unit` — no external deps, uses `mock_embedder` and `mock_vector_store` fixtures from `conftest.py`
- `@pytest.mark.integration` — requires Docker; uses `redis_client` (session-scoped container) + `flush_redis` (per-test isolation)
- `mock_embedder` returns deterministic seeded vectors: 768-dim unified — shape must match real model
