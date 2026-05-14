"""Result re-ranking based on relevance signals for Phase 5."""
import logging
from typing import List

from ._search import SearchResult

logger = logging.getLogger(__name__)


class SignalRanker:
    """Re-rank search results using relevance signals.

    Signals:
    - is_semantic_match: Result name/docstring contains query terms
    - is_non_test: Not a test file
    - is_definition: Function/class definition, not comment/log
    - is_canonical: First occurrence (not duplicate)
    """

    def __init__(self):
        self.test_patterns = ["test_", "_test.py", ".test.", "conftest.py"]
        self.definition_kinds = {"function", "class"}

    def _is_test_file(self, path: str) -> bool:
        """Check if path looks like a test file."""
        return any(pat in path for pat in self.test_patterns)

    def _is_definition(self, result: SearchResult) -> bool:
        """Check if result is a definition (function/class) vs comment/log."""
        metadata = result.metadata or {}
        kind = metadata.get("kind", "")
        return kind in self.definition_kinds

    def _contains_query_terms(self, text: str, query: str) -> bool:
        """Check if text contains significant query terms.

        Returns True if at least 30% of query words appear in text.
        """
        if not text or not query:
            return False
        query_words = set(query.lower().split())
        text_lower = text.lower()
        # Count how many query words appear in text (simple substring match)
        matches = sum(1 for word in query_words if word in text_lower)
        return matches / len(query_words) >= 0.3 if query_words else False

    def compute_signal_score(self, result: SearchResult, query: str = "") -> float:
        """Compute combined signal score for a result (0.0 - 3.0).

        Higher is better. Additive signals:
        - +1.0 for non-test file
        - +1.0 for being a definition (function/class)
        - +1.0 for containing query keywords in name/docstring
        """
        score = 0.0
        metadata = result.metadata or {}

        # Signal 1: Non-test file
        if not self._is_test_file(result.path_or_source):
            score += 1.0

        # Signal 2: Is a definition (function/class, not boilerplate)
        if self._is_definition(result):
            score += 1.0

        # Signal 3: Query terms in name or docstring
        name = metadata.get("name", "")
        docstring = metadata.get("docstring", "")
        combined = f"{name} {docstring}"
        if self._contains_query_terms(combined, query):
            score += 1.0

        return score

    def rerank(
        self,
        results: List[SearchResult],
        query: str = "",
        embedding_weight: float = 0.7,
    ) -> List[SearchResult]:
        """Re-rank results using embedding score + signal score.

        Formula: final_score = embedding_score * embedding_weight + signal_score * (1 - embedding_weight)

        embedding_weight: How much to trust the embedding model vs signals
        - 0.5 = balanced (signals matter as much as embeddings)
        - 0.7 = embedding-primary (default, embeddings dominate)
        - 0.9 = embedding-heavy (signals have minimal impact)
        """
        if not results:
            return results

        # Compute signal scores
        signal_scores = []
        for result in results:
            signal_score = self.compute_signal_score(result, query)
            signal_scores.append(signal_score)

        # Normalize scores to [0, 1]
        min_embedding = min(r.score for r in results)
        max_embedding = max(r.score for r in results)
        embedding_range = max_embedding - min_embedding if max_embedding > min_embedding else 1

        min_signal = min(signal_scores) if signal_scores else 0
        max_signal = max(signal_scores) if signal_scores else 3
        signal_range = max_signal - min_signal if max_signal > min_signal else 1

        # Compute combined scores
        reranked = []
        for result, signal_score in zip(results, signal_scores):
            # Normalize scores to [0, 1]
            norm_embedding = (result.score - min_embedding) / embedding_range if embedding_range > 0 else 0.5
            norm_signal = (signal_score - min_signal) / signal_range if signal_range > 0 else 0.5

            # Combine with weight
            combined_score = (
                norm_embedding * embedding_weight
                + norm_signal * (1.0 - embedding_weight)
            )

            result.score = combined_score
            reranked.append(result)

        # Sort by combined score
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked
