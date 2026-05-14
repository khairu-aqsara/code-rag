import logging
import os
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_embedding_service, get_vector_store
from ..schemas import (
    DeleteResponse,
    DefaultProjectRequest,
    DefaultProjectResponse,
    HealthResponse,
    IngestCodeRequest,
    IngestDocsRequest,
    IngestResponse,
    ListFilesResponse,
    ListProjectsResponse,
    ProjectInfoResponse,
    ProjectStats,
    StatsResponse,
)
from ...config import settings

logger = logging.getLogger(__name__)
admin_router = APIRouter()


def _validate_path(path: str) -> str:
    """Prevent directory traversal: resolve path and assert it's within BASE_PATH."""
    resolved = os.path.realpath(path)
    base = os.path.realpath(settings.BASE_PATH)
    if not resolved.startswith(base + os.sep) and resolved != base:
        raise HTTPException(
            status_code=400,
            detail=f"Path '{path}' is outside the allowed base directory '{settings.BASE_PATH}'",
        )
    return resolved


@admin_router.post("/projects/{project_id}/ingest-code", response_model=IngestResponse)
def ingest_code(
    project_id: str,
    req: IngestCodeRequest,
    vector_store=Depends(get_vector_store),
    embedder=Depends(get_embedding_service),
) -> IngestResponse:
    """Ingest code files from a directory into the vector store."""
    try:
        _validate_path(req.root_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        from ...ingestor.code import CodeIngestor
        ingestor = CodeIngestor(embedder, vector_store)
        result = ingestor.ingest(
            project_id=project_id,
            root_path=req.root_path,
            lang_filter=req.lang_filter,
            skip_patterns=req.skip_patterns,
        )
        return IngestResponse(
            status="success",
            total_files=result.total_files,
            total_chunks=result.total_chunks,
            duration_seconds=result.duration_seconds,
            errors=result.errors,
        )
    except Exception as e:
        logger.error(f"ingest_code failed for project '{project_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/projects/{project_id}/ingest-docs", response_model=IngestResponse)
def ingest_docs(
    project_id: str,
    req: IngestDocsRequest,
    vector_store=Depends(get_vector_store),
    embedder=Depends(get_embedding_service),
) -> IngestResponse:
    """Ingest documentation files into the vector store."""
    # Validate each path is within BASE_PATH
    for path in req.file_paths:
        # For glob patterns, validate the base prefix before expansion
        base_path = path.split("*")[0].rstrip("/")
        if base_path:
            try:
                _validate_path(base_path)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

    try:
        from ...ingestor.docs import DocIngestor
        ingestor = DocIngestor(embedder, vector_store)
        result = ingestor.ingest(
            project_id=project_id,
            file_paths=req.file_paths,
            tags=req.tags,
        )
        return IngestResponse(
            status="success",
            total_files=result.total_files,
            total_chunks=result.total_chunks,
            duration_seconds=result.duration_seconds,
            errors=result.errors,
        )
    except Exception as e:
        logger.error(f"ingest_docs failed for project '{project_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.delete("/projects/{project_id}", response_model=DeleteResponse)
def delete_project(
    project_id: str,
    vector_store=Depends(get_vector_store),
) -> DeleteResponse:
    """Delete all indexed data for a project."""
    try:
        count = vector_store.delete_project(project_id)
        return DeleteResponse(status="success", deleted_count=count)
    except Exception as e:
        logger.error(f"delete_project failed for '{project_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/projects/{project_id}/files", response_model=ListFilesResponse)
def list_project_files(
    project_id: str,
    prefix: str = "",
    vector_store=Depends(get_vector_store),
) -> ListFilesResponse:
    """Return unique file paths for a project."""
    try:
        files = vector_store.list_project_files(project_id, prefix=prefix)
        return ListFilesResponse(project_id=project_id, files=files, total=len(files))
    except Exception as e:
        logger.error(f"list_project_files failed for '{project_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/projects", response_model=ListProjectsResponse)
def list_projects(
    vector_store=Depends(get_vector_store),
) -> ListProjectsResponse:
    """List all projects that have indexed data, with per-project chunk counts."""
    try:
        project_stats = vector_store.list_projects()
        return ListProjectsResponse(
            projects=[
                ProjectStats(
                    project_id=p.project_id,
                    code_chunks=p.code_chunks,
                    doc_chunks=p.doc_chunks,
                )
                for p in project_stats
            ],
            total=len(project_stats),
        )
    except Exception as e:
        logger.error(f"list_projects failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.put("/config/default-project", response_model=DefaultProjectResponse)
def set_default_project(
    req: DefaultProjectRequest,
    vector_store=Depends(get_vector_store),
) -> DefaultProjectResponse:
    """Set the default project_id for a workspace."""
    try:
        key = f"config:default_project:{req.workspace}"
        vector_store.redis.set(key, req.project_id)
        return DefaultProjectResponse(workspace=req.workspace, project_id=req.project_id)
    except Exception as e:
        logger.error(f"set_default_project failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/config/default-project", response_model=DefaultProjectResponse)
def get_default_project(
    workspace: str,
    vector_store=Depends(get_vector_store),
) -> DefaultProjectResponse:
    """Get the default project_id for a workspace."""
    try:
        key = f"config:default_project:{workspace}"
        project_id = vector_store.redis.get(key)
        if isinstance(project_id, bytes):
            project_id = project_id.decode("utf-8")
        return DefaultProjectResponse(workspace=workspace, project_id=project_id)
    except Exception as e:
        logger.error(f"get_default_project failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/projects/{project_id}/info", response_model=ProjectInfoResponse)
def get_project_info(
    project_id: str,
    vector_store=Depends(get_vector_store),
) -> ProjectInfoResponse:
    """Get detailed information about a project including chunk counts, languages, and tags."""
    try:
        info = vector_store.get_project_info(project_id)
        return ProjectInfoResponse(
            project_id=info["project_id"],
            code_chunks=info["code_chunks"],
            doc_chunks=info["doc_chunks"],
            last_indexed=str(info["last_indexed"]) if info["last_indexed"] else None,
            index_age_days=info["index_age_days"],
            languages=info["languages"],
            doc_tags=info["doc_tags"],
        )
    except Exception as e:
        logger.error(f"get_project_info failed for '{project_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/health", response_model=HealthResponse)
def health(
    vector_store=Depends(get_vector_store),
    embedder=Depends(get_embedding_service),
) -> HealthResponse:
    """Check Redis connectivity and model availability."""
    try:
        vector_store.redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    models_loaded = embedder is not None

    return HealthResponse(
        status="ok" if redis_ok else "error",
        redis_ok=redis_ok,
        models_loaded=models_loaded,
    )


@admin_router.get("/stats", response_model=StatsResponse)
def stats(
    vector_store=Depends(get_vector_store),
) -> StatsResponse:
    """Return statistics for both RediSearch indices."""
    try:
        index_stats = vector_store.get_stats()
        return StatsResponse(indices=index_stats)
    except Exception as e:
        logger.error(f"stats failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
