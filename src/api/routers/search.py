import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response

from ...vector_store import SearchResult
from ..deps import get_embedding_service, get_vector_store
from ..schemas import (
    RESULT_FIELDS,
    ResponseType,
    FindSymbolsRequest,
    SearchBatchRequest,
    SearchBatchResponse,
    SearchCodeExactRequest,
    SearchCodeRequest,
    SearchDocsRequest,
    SearchHybridRequest,
    SearchResultItem,
)

logger = logging.getLogger(__name__)
search_router = APIRouter()

RRF_K = 60


def _inject_index_age(response: Response, project_id: str, vector_store) -> None:
    """Inject X-Index-Age header if index age data is available."""
    try:
        age = vector_store.get_index_age(project_id)
        if age:
            response.headers["X-Index-Age"] = age
    except Exception:
        pass


def _to_item(result: SearchResult, response_type: ResponseType = ResponseType.json) -> SearchResultItem:
    meta = result.metadata or {}
    include_metadata = response_type == ResponseType.json_full
    return SearchResultItem(
        score=result.score,
        path_or_source=result.path_or_source,
        content=result.content,
        start_line=result.start_line,
        end_line=result.end_line,
        name=meta.get("name"),
        kind=meta.get("kind"),
        docstring=meta.get("docstring"),
        language=meta.get("lang"),
        match_type=meta.get("match_type"),
        original_path=meta.get("original_path"),
        metadata=meta if include_metadata and meta else None,
    )


def _serialize_items(
    items: List[SearchResultItem],
    response_type: ResponseType = ResponseType.json,
    fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    exclude_none = response_type == ResponseType.json
    projected_fields = None
    if fields:
        valid = set(fields) & RESULT_FIELDS
        if valid:
            projected_fields = valid

    out: List[Dict[str, Any]] = []
    for item in items:
        d = item.model_dump(exclude_none=exclude_none)
        if projected_fields is not None:
            d = {k: v for k, v in d.items() if k in projected_fields}
        out.append(d)
    return out


def _rrf_score(rank: int) -> float:
    """Reciprocal Rank Fusion score: 1 / (rank + k), rank is 0-indexed."""
    return 1.0 / (rank + RRF_K)


@search_router.post("/search-code")
def search_code(
    req: SearchCodeRequest,
    response: Response,
    embedder=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
):
    try:
        query_embedding = embedder.embed(req.query)

        if req.use_summary:
            results = vector_store.search_code_summary(
                query_embedding=query_embedding,
                project_id=req.project_id,
                k=req.k,
                lang_filter=req.lang_filter,
                exclude_tests=req.exclude_tests,
                exclude_paths=req.exclude_paths,
                min_score=req.min_score,
            )
        else:
            results = vector_store.search_code(
                query_embedding=query_embedding,
                project_id=req.project_id,
                k=req.k,
                lang_filter=req.lang_filter,
                path_filter=req.path_prefix,
                exclude_tests=req.exclude_tests,
                exclude_paths=req.exclude_paths,
                min_score=req.min_score,
                query_text=req.query,
                rerank=req.rerank,
            )
        items = [_to_item(r, req.response_type) for r in results]
        serialized = _serialize_items(items, req.response_type, req.fields)
        _inject_index_age(response, req.project_id, vector_store)
        return {"results": serialized}
    except Exception as e:
        logger.error(f"search_code failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@search_router.post("/search-docs")
def search_docs(
    req: SearchDocsRequest,
    response: Response,
    embedder=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
):
    try:
        query_embedding = embedder.embed(req.query)
        results = vector_store.search_docs(
            query_embedding=query_embedding,
            project_id=req.project_id,
            k=req.k,
            tags=req.tags,
            exclude_sources=req.exclude_sources,
            min_score=req.min_score,
            query_text=req.query,
            rerank=req.rerank,
        )
        items = [_to_item(r, req.response_type) for r in results]
        serialized = _serialize_items(items, req.response_type, req.fields)
        _inject_index_age(response, req.project_id, vector_store)
        return {"results": serialized}
    except Exception as e:
        logger.error(f"search_docs failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@search_router.post("/search-hybrid")
def search_hybrid(
    req: SearchHybridRequest,
    response: Response,
    embedder=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
):
    try:
        # Single embedding call — same model for both code and docs
        query_embedding = embedder.embed(req.query)

        code_results = vector_store.search_code(
            query_embedding=query_embedding,
            project_id=req.project_id,
            k=req.k,
            lang_filter=req.lang_filter,
            exclude_tests=req.exclude_tests,
            exclude_paths=req.exclude_paths,
            min_score=req.min_score,
        )
        doc_results = vector_store.search_docs(
            query_embedding=query_embedding,
            project_id=req.project_id,
            k=req.k,
            tags=req.tags,
            min_score=req.min_score,
        )

        rrf_scores: dict[str, tuple[float, SearchResult]] = {}

        for rank, result in enumerate(code_results):
            key = f"code:{result.path_or_source}:{result.start_line}"
            rrf_scores[key] = (rrf_scores.get(key, (0.0, result))[0] + _rrf_score(rank), result)

        for rank, result in enumerate(doc_results):
            key = f"doc:{result.path_or_source}"
            rrf_scores[key] = (rrf_scores.get(key, (0.0, result))[0] + _rrf_score(rank), result)

        sorted_results = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        top_results = [r for _, r in sorted_results[: req.k]]

        items = [_to_item(r, req.response_type) for r in top_results]
        serialized = _serialize_items(items, req.response_type, req.fields)
        _inject_index_age(response, req.project_id, vector_store)
        return {"results": serialized}
    except Exception as e:
        logger.error(f"search_hybrid failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@search_router.post("/search-code-hybrid")
def search_code_hybrid(
    req: SearchCodeRequest,
    response: Response,
    embedder=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
):
    try:
        query_embedding = embedder.embed(req.query)
        results = vector_store.search_code_hybrid(
            query_text=req.query,
            query_embedding=query_embedding,
            project_id=req.project_id,
            k=req.k,
            lang_filter=req.lang_filter,
            exclude_tests=req.exclude_tests,
            exclude_paths=req.exclude_paths,
            min_score=req.min_score,
            semantic_weight=req.semantic_weight,
            rerank=req.rerank,
        )
        items = [_to_item(r, req.response_type) for r in results]
        serialized = _serialize_items(items, req.response_type, req.fields)
        _inject_index_age(response, req.project_id, vector_store)
        return {"results": serialized}
    except Exception as e:
        logger.error(f"search_code_hybrid failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@search_router.post("/search-docs-hybrid")
def search_docs_hybrid(
    req: SearchDocsRequest,
    response: Response,
    embedder=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
):
    try:
        query_embedding = embedder.embed(req.query)
        results = vector_store.search_docs_hybrid(
            query_text=req.query,
            query_embedding=query_embedding,
            project_id=req.project_id,
            k=req.k,
            tags=req.tags,
            exclude_sources=req.exclude_sources,
            min_score=req.min_score,
            semantic_weight=req.semantic_weight,
            rerank=req.rerank,
        )
        items = [_to_item(r, req.response_type) for r in results]
        serialized = _serialize_items(items, req.response_type, req.fields)
        _inject_index_age(response, req.project_id, vector_store)
        return {"results": serialized}
    except Exception as e:
        logger.error(f"search_docs_hybrid failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@search_router.post("/search-code-exact")
def search_code_exact(
    req: SearchCodeExactRequest,
    response: Response,
    vector_store=Depends(get_vector_store),
):
    """Keyword-only search using BM25 — no embedding needed. Fast exact match."""
    try:
        results = vector_store.search_code_exact(
            query_text=req.query,
            project_id=req.project_id,
            k=req.k,
            lang_filter=req.lang_filter,
            exclude_tests=req.exclude_tests,
            exclude_paths=req.exclude_paths,
        )
        items = [_to_item(r, req.response_type) for r in results]
        serialized = _serialize_items(items, req.response_type, req.fields)
        _inject_index_age(response, req.project_id, vector_store)
        return {"results": serialized}
    except Exception as e:
        logger.error(f"search_code_exact failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@search_router.post("/find-symbols")
def find_symbols(
    req: FindSymbolsRequest,
    response: Response,
    vector_store=Depends(get_vector_store),
):
    """Find symbol definitions (functions, classes) by name or kind."""
    try:
        results = vector_store.find_symbols(
            project_id=req.project_id,
            name=req.name,
            kind=req.kind,
            k=req.k,
        )
        items = [_to_item(r, req.response_type) for r in results]
        serialized = _serialize_items(items, req.response_type, req.fields)
        _inject_index_age(response, req.project_id, vector_store)
        return {"results": serialized}
    except Exception as e:
        logger.error(f"find_symbols failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@search_router.post("/search-batch")
def search_batch(
    req: SearchBatchRequest,
    response: Response,
    embedder=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
) -> Dict[str, Any]:
    """Execute multiple search queries in a single request.
    
    Useful for AI agents that need to run several related searches efficiently.
    Each query is executed independently and results are returned keyed by query index.
    """
    try:
        results: Dict[str, List[SearchResultItem]] = {}
        
        for i, q in enumerate(req.queries):
            query_results: List[SearchResult] = []
            
            if q.type == "code":
                embedding = embedder.embed(q.query)
                query_results = vector_store.search_code(
                    query_embedding=embedding,
                    project_id=req.project_id,
                    k=req.k,
                    lang_filter=q.lang_filter,
                    exclude_tests=q.exclude_tests,
                    exclude_paths=q.exclude_paths,
                    min_score=q.min_score,
                    query_text=q.query,
                )
            elif q.type == "code_exact":
                query_results = vector_store.search_code_exact(
                    query_text=q.query,
                    project_id=req.project_id,
                    k=req.k,
                    lang_filter=q.lang_filter,
                    exclude_tests=q.exclude_tests,
                    exclude_paths=q.exclude_paths,
                )
            elif q.type == "code_hybrid":
                embedding = embedder.embed(q.query)
                query_results = vector_store.search_code_hybrid(
                    query_text=q.query,
                    query_embedding=embedding,
                    project_id=req.project_id,
                    k=req.k,
                    semantic_weight=q.semantic_weight or 0.3,
                    lang_filter=q.lang_filter,
                    exclude_tests=q.exclude_tests,
                    exclude_paths=q.exclude_paths,
                    min_score=q.min_score,
                )
            elif q.type == "docs":
                embedding = embedder.embed(q.query)
                query_results = vector_store.search_docs(
                    query_embedding=embedding,
                    project_id=req.project_id,
                    k=req.k,
                    tags=q.tags,
                    min_score=q.min_score,
                )
            elif q.type == "docs_hybrid":
                embedding = embedder.embed(q.query)
                query_results = vector_store.search_docs_hybrid(
                    query_text=q.query,
                    query_embedding=embedding,
                    project_id=req.project_id,
                    k=req.k,
                    semantic_weight=q.semantic_weight or 0.6,
                    tags=q.tags,
                    min_score=q.min_score,
                )
            else:
                raise ValueError(f"Unknown query type: {q.type}. Valid types: code, code_exact, code_hybrid, docs, docs_hybrid")
            
            items = [_to_item(r, req.response_type) for r in query_results]
            results[f"query_{i}"] = _serialize_items(items, req.response_type, req.fields)
        
        # Inject index age for the first query's project
        if req.queries:
            _inject_index_age(response, req.project_id, vector_store)
        
        return {"results": results}
    except Exception as e:
        logger.error(f"search_batch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
