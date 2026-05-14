from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResponseType(str, Enum):
    json = "json"
    json_full = "json-full"


class ListFilesResponse(BaseModel):
    project_id: str
    files: List[str]
    total: int


class FindSymbolsRequest(BaseModel):
    project_id: str
    name: Optional[str] = None
    kind: Optional[str] = None
    k: int = Field(10, ge=1, le=50)
    response_type: ResponseType = ResponseType.json
    fields: Optional[List[str]] = None


class SearchCodeExactRequest(BaseModel):
    project_id: str
    query: str
    k: int = Field(10, ge=1, le=50)
    lang_filter: Optional[List[str]] = None
    exclude_tests: bool = True
    exclude_paths: Optional[List[str]] = None
    response_type: ResponseType = ResponseType.json
    fields: Optional[List[str]] = None


class SearchCodeRequest(BaseModel):
    project_id: str
    query: str
    k: int = Field(10, ge=1, le=50)
    lang_filter: Optional[List[str]] = None
    path_prefix: Optional[str] = None
    exclude_tests: bool = True
    exclude_paths: Optional[List[str]] = None
    min_score: Optional[float] = None
    semantic_weight: float = Field(0.3, ge=0.0, le=1.0)
    rerank: bool = False
    use_summary: bool = False
    response_type: ResponseType = ResponseType.json
    fields: Optional[List[str]] = None


class SearchDocsRequest(BaseModel):
    project_id: str
    query: str
    k: int = Field(10, ge=1, le=50)
    tags: Optional[List[str]] = None
    min_score: Optional[float] = None
    exclude_sources: Optional[List[str]] = None
    semantic_weight: float = Field(0.6, ge=0.0, le=1.0)
    rerank: bool = False
    response_type: ResponseType = ResponseType.json
    fields: Optional[List[str]] = None


class SearchHybridRequest(BaseModel):
    project_id: str
    query: str
    k: int = Field(10, ge=1, le=50)
    lang_filter: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    exclude_tests: bool = True
    exclude_paths: Optional[List[str]] = None
    min_score: Optional[float] = None
    semantic_weight: float = Field(0.5, ge=0.0, le=1.0)
    rerank: bool = False
    response_type: ResponseType = ResponseType.json
    fields: Optional[List[str]] = None


RESULT_FIELDS = {
    "score",
    "path_or_source",
    "content",
    "start_line",
    "end_line",
    "name",
    "kind",
    "docstring",
    "language",
    "match_type",
    "original_path",
    "last_indexed",
}


class SearchResultItem(BaseModel):
    score: float
    path_or_source: str
    content: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    name: Optional[str] = None
    kind: Optional[str] = None
    docstring: Optional[str] = None
    language: Optional[str] = None
    match_type: Optional[str] = None
    original_path: Optional[str] = None
    last_indexed: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchCodeResponse(BaseModel):
    results: List[SearchResultItem]


class SearchDocsResponse(BaseModel):
    results: List[SearchResultItem]


class SearchHybridResponse(BaseModel):
    results: List[SearchResultItem]


class IngestCodeRequest(BaseModel):
    root_path: str
    lang_filter: Optional[List[str]] = None
    skip_patterns: Optional[List[str]] = None


class IngestDocsRequest(BaseModel):
    file_paths: List[str]
    tags: Optional[List[str]] = None


class IngestResponse(BaseModel):
    status: str
    total_files: int
    total_chunks: int
    duration_seconds: float
    errors: List[str] = []


class HealthResponse(BaseModel):
    status: str
    redis_ok: bool
    models_loaded: bool


class DeleteResponse(BaseModel):
    status: str
    deleted_count: int


class StatsResponse(BaseModel):
    indices: Dict[str, Any]


class ProjectStats(BaseModel):
    """Chunk counts for a single project."""

    project_id: str
    code_chunks: int
    doc_chunks: int


class ListProjectsResponse(BaseModel):
    """Response for GET /api/projects."""

    projects: List[ProjectStats]
    total: int


class DefaultProjectRequest(BaseModel):
    """Request for setting default project."""

    workspace: str
    project_id: str


class DefaultProjectResponse(BaseModel):
    """Response for getting default project."""

    status: str = "success"
    workspace: str
    project_id: Optional[str] = None


class ProjectInfoResponse(BaseModel):
    """Response for GET /api/projects/{project_id}/info."""

    project_id: str
    code_chunks: int
    doc_chunks: int
    last_indexed: Optional[str] = None
    index_age_days: Optional[int] = None
    languages: List[str] = []
    doc_tags: List[str] = []


class BatchSearchQuery(BaseModel):
    """Single query within a batch search request."""

    type: str  # "code", "code_exact", "code_hybrid", "docs", "docs_hybrid", "hybrid"
    query: str
    semantic_weight: Optional[float] = None
    lang_filter: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    exclude_tests: bool = True
    exclude_paths: Optional[List[str]] = None
    min_score: Optional[float] = None


class SearchBatchRequest(BaseModel):
    """Request for batch search endpoint."""

    project_id: str
    queries: List[BatchSearchQuery] = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=50)
    response_type: ResponseType = ResponseType.json
    fields: Optional[List[str]] = None


class SearchBatchResponse(BaseModel):
    """Response for batch search endpoint."""

    results: Dict[str, List[SearchResultItem]]
