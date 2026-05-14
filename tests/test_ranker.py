"""Tests for Phase 5: Signal-based result re-ranking."""
import pytest

from src.ranker import SignalRanker
from src.vector_store import SearchResult


class TestSignalRankerBasics:
    """Tests for SignalRanker signal computation."""

    @pytest.fixture
    def ranker(self):
        return SignalRanker()

    def test_non_test_file_signal(self, ranker):
        """Test detection of non-test files."""
        non_test = SearchResult(
            score=0.8,
            path_or_source="/src/auth.py",
            content="def authenticate(): pass",
            metadata={"kind": "function"},
        )
        assert ranker._is_test_file(non_test.path_or_source) is False

    def test_test_file_signal(self, ranker):
        """Test detection of test files."""
        test_patterns = [
            "/src/test_auth.py",
            "/src/auth_test.py",
            "/src/auth.test.js",
            "/tests/conftest.py",
        ]
        for test_path in test_patterns:
            assert ranker._is_test_file(test_path) is True

    def test_definition_signal(self, ranker):
        """Test detection of definitions (function/class)."""
        definition = SearchResult(
            score=0.8,
            path_or_source="/src/auth.py",
            content="def authenticate(): pass",
            metadata={"kind": "function"},
        )
        assert ranker._is_definition(definition) is True

        definition_class = SearchResult(
            score=0.8,
            path_or_source="/src/auth.py",
            content="class AuthManager: pass",
            metadata={"kind": "class"},
        )
        assert ranker._is_definition(definition_class) is True

    def test_non_definition_signal(self, ranker):
        """Test detection of non-definitions (comments, imports)."""
        comment = SearchResult(
            score=0.8,
            path_or_source="/src/auth.py",
            content="# TODO: fix auth",
            metadata={"kind": "comment"},
        )
        assert ranker._is_definition(comment) is False

    def test_query_terms_in_name(self, ranker):
        """Test detection of query terms in result name."""
        result = SearchResult(
            score=0.8,
            path_or_source="/src/auth.py",
            content="def authenticate_user(): pass",
            metadata={
                "kind": "function",
                "name": "authenticate_user",
                "docstring": "User authentication.",
            },
        )
        # Query "authenticate" appears in name
        assert ranker._contains_query_terms("authenticate_user User authentication", "authenticate") is True

    def test_query_terms_threshold(self, ranker):
        """Test that query terms need 30% overlap."""
        # Query with 3 words
        combined = "authenticate verify user session token"
        # "authenticate" and "user" match = 2/3 = 67% (above 30%)
        assert ranker._contains_query_terms(combined, "authenticate find user") is True
        # Only "user" matches = 1/3 = 33% (above 30%)
        assert ranker._contains_query_terms(combined, "user session verify") is True
        # No matches = 0/2 = 0% (below 30%)
        assert ranker._contains_query_terms(combined, "login password") is False


class TestComputeSignalScore:
    """Tests for signal score computation."""

    @pytest.fixture
    def ranker(self):
        return SignalRanker()

    def test_perfect_score(self, ranker):
        """Test result with all signals (non-test, definition, query match)."""
        result = SearchResult(
            score=0.8,
            path_or_source="/src/auth.py",
            content="def authenticate_user(): pass",
            metadata={
                "kind": "function",
                "name": "authenticate_user",
                "docstring": "Authenticate user credentials.",
            },
        )
        score = ranker.compute_signal_score(result, "authenticate user")
        assert score == 3.0  # All 3 signals: non-test + definition + query match

    def test_partial_score(self, ranker):
        """Test result with some signals."""
        # Non-test + definition, but no query match
        result = SearchResult(
            score=0.8,
            path_or_source="/src/utils.py",
            content="def helper(): pass",
            metadata={"kind": "function", "name": "helper"},
        )
        score = ranker.compute_signal_score(result, "authenticate")
        assert score == 2.0

    def test_test_file_penalty(self, ranker):
        """Test that test files lose the non-test signal."""
        result = SearchResult(
            score=0.8,
            path_or_source="/src/test_auth.py",
            content="def test_authenticate(): pass",
            metadata={"kind": "function", "name": "test_authenticate"},
        )
        score = ranker.compute_signal_score(result, "authenticate")
        # No non-test signal (it's a test file), but has definition + query match
        assert score == 2.0

    def test_minimum_score(self, ranker):
        """Test result with no signals (test file, import statement, no query match)."""
        result = SearchResult(
            score=0.8,
            path_or_source="/src/test_helpers.py",
            content="from auth import authenticate",
            metadata={"kind": "import"},
        )
        score = ranker.compute_signal_score(result, "validate")
        assert score == 0.0


class TestReranking:
    """Tests for result re-ranking logic."""

    @pytest.fixture
    def ranker(self):
        return SignalRanker()

    def test_rerank_sorts_by_combined_score(self, ranker):
        """Test that re-ranking sorts by combined embedding + signal score."""
        results = [
            # High embedding score but test file (no signals)
            SearchResult(
                score=0.95,
                path_or_source="/src/test_auth.py",
                content="def test_auth(): pass",
                metadata={"kind": "function", "name": "test_auth"},
            ),
            # Lower embedding score but non-test definition with query match (all signals)
            SearchResult(
                score=0.70,
                path_or_source="/src/auth.py",
                content="def authenticate_user(): pass",
                metadata={
                    "kind": "function",
                    "name": "authenticate_user",
                    "docstring": "Authenticate user.",
                },
            ),
        ]

        # With embedding_weight=0.7, signals still dominate enough to flip order
        # due to 3x difference in signal scores (1.0 vs 3.0)
        reranked = ranker.rerank(results, "authenticate user", embedding_weight=0.3)

        # With lower embedding weight, signals dominate
        assert reranked[0].path_or_source == "/src/auth.py"
        assert reranked[1].path_or_source == "/src/test_auth.py"

    def test_rerank_respects_embedding_weight(self, ranker):
        """Test that embedding_weight controls signal influence."""
        result1 = SearchResult(
            score=0.90,
            path_or_source="/src/test_util.py",
            content="def test_helper(): pass",
            metadata={"kind": "function"},
        )
        result2 = SearchResult(
            score=0.80,
            path_or_source="/src/util.py",
            content="def helper(): pass",
            metadata={"kind": "function"},
        )

        # With high embedding_weight (0.9), embeddings dominate
        reranked_heavy = ranker.rerank([result1, result2], "", embedding_weight=0.9)
        # result1 should still rank first (higher embedding score overwhelms signal)
        assert reranked_heavy[0].path_or_source == "/src/test_util.py"

        # With low embedding_weight (0.3), signals dominate
        reranked_light = ranker.rerank([result1, result2], "", embedding_weight=0.3)
        # result2 should rank first (signals outweigh embedding score)
        assert reranked_light[0].path_or_source == "/src/util.py"

    def test_rerank_normalized_scores(self, ranker):
        """Test that re-ranked scores are normalized to [0, 1] range."""
        results = [
            SearchResult(
                score=0.95,
                path_or_source="/src/a.py",
                content="a",
                metadata={"kind": "function"},
            ),
            SearchResult(
                score=0.50,
                path_or_source="/src/b.py",
                content="b",
                metadata={"kind": "function"},
            ),
        ]

        reranked = ranker.rerank(results, "", embedding_weight=0.7)

        # All scores should be in [0, 1] range
        for result in reranked:
            assert 0.0 <= result.score <= 1.0


class TestUseCase:
    """Real-world use case tests for Phase 5 re-ranking."""

    @pytest.fixture
    def ranker(self):
        return SignalRanker()

    def test_boost_canonical_functions(self, ranker):
        """Use case: Find canonical version of function among duplicates.

        Original authenticate() in auth.py (non-test, definition, matches query)
        vs Test mock in test_auth.py (test file, has definition and query match)
        → Canonical should rank first with signal-driven weight.
        """
        canonical = SearchResult(
            score=0.82,  # slightly lower embedding
            path_or_source="/src/auth.py",
            content="def authenticate(user, pwd):\n    \"\"\"Authenticate user.\"\"\"",
            metadata={
                "kind": "function",
                "name": "authenticate",
                "docstring": "Authenticate user.",
            },
        )
        test_mock = SearchResult(
            score=0.88,  # higher embedding (test-optimized)
            path_or_source="/tests/test_auth.py",
            content="def test_authenticate():\n    assert authenticate('user', 'pass')",
            metadata={"kind": "function", "name": "test_authenticate"},
        )

        # With embedding_weight=0.3 (signal-driven), canonical wins
        reranked = ranker.rerank([test_mock, canonical], "authenticate user", embedding_weight=0.3)

        # Canonical should rank first due to signal advantage (non-test)
        assert reranked[0].path_or_source == "/src/auth.py"

    def test_query_specific_boost(self, ranker):
        """Use case: Boost results where query terms appear in function names.

        Search: "authenticate"
        Result A: validate_session() in util.py (high embedding, no name match, 2 signals)
        Result B: authenticate() in auth.py (lower embedding, exact name match, 3 signals)
        → Should boost Result B to rank first with signal-driven weighting.
        """
        generic = SearchResult(
            score=0.90,  # high embedding
            path_or_source="/src/util.py",
            content="def validate_session(token):\n    \"\"\"Validate session token.\"\"\"",
            metadata={
                "kind": "function",
                "name": "validate_session",
                "docstring": "Validate session.",
            },
        )
        exact = SearchResult(
            score=0.72,  # lower embedding
            path_or_source="/src/auth.py",
            content="def authenticate(user, pwd):\n    \"\"\"Authenticate user.\"\"\"",
            metadata={
                "kind": "function",
                "name": "authenticate",
                "docstring": "Authenticate.",
            },
        )

        # With embedding_weight=0.4 (signal-driven), exact match wins
        reranked = ranker.rerank([generic, exact], "authenticate", embedding_weight=0.4)

        # Exact match should rank first due to signal advantage (query in name)
        assert reranked[0].path_or_source == "/src/auth.py"
