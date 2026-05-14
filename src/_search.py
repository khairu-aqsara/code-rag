"""FT.SEARCH result parsing helpers for VectorStore.

Kept in a separate module so vector_store.py stays under 300 lines.
All fields from Redis come back as bytes (decode_responses=False).
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    score: float
    path_or_source: str
    content: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: Optional[Dict] = None


def parse_code_result(doc: dict, score: float) -> SearchResult:
    """Parse a raw Redis hash (bytes keys/values) into a SearchResult."""
    metadata = {
        "lang": doc[b"lang"].decode("utf-8"),
        "project_id": doc[b"project_id"].decode("utf-8"),
    }
    # Add optional semantic metadata
    if b"name" in doc:
        metadata["name"] = doc[b"name"].decode("utf-8")
    if b"kind" in doc:
        metadata["kind"] = doc[b"kind"].decode("utf-8")
    if b"docstring" in doc:
        metadata["docstring"] = doc[b"docstring"].decode("utf-8")
    if b"original_path" in doc:
        metadata["original_path"] = doc[b"original_path"].decode("utf-8")

    return SearchResult(
        score=score,
        path_or_source=doc[b"path"].decode("utf-8"),
        content=doc[b"code"].decode("utf-8"),
        start_line=int(doc[b"start_line"]),
        end_line=int(doc[b"end_line"]),
        metadata=metadata,
    )


def parse_doc_result(doc: dict, score: float) -> SearchResult:
    """Parse a raw Redis hash (bytes keys/values) into a SearchResult."""
    tags_raw = doc.get(b"tags", b"").decode("utf-8")
    tags = [t for t in tags_raw.split(",") if t] if tags_raw else []
    return SearchResult(
        score=score,
        path_or_source=doc[b"source"].decode("utf-8"),
        content=doc[b"content"].decode("utf-8"),
        metadata={
            "tags": tags,
            "project_id": doc[b"project_id"].decode("utf-8"),
        },
    )


def parse_ft_search(raw: list, result_type: str) -> List[SearchResult]:
    """Parse a raw FT.SEARCH response list into SearchResult objects.

    FT.SEARCH response format: [total_count, key1, [f1,v1,f2,v2,...], key2, ...]
    """
    if not raw or raw[0] == 0:
        return []

    results: List[SearchResult] = []
    items = raw[1:]

    for i in range(0, len(items), 2):
        fields_list = items[i + 1]
        doc: dict = {}
        for j in range(0, len(fields_list), 2):
            doc[fields_list[j]] = fields_list[j + 1]

        score_raw = doc.get(b"score", b"1.0")
        score = float(score_raw) if score_raw else 1.0

        try:
            if result_type == "code":
                results.append(parse_code_result(doc, score))
            else:
                results.append(parse_doc_result(doc, score))
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping malformed search result: {e}")

    return results
