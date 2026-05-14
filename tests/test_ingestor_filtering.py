import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest

from src.config import settings
from src.ingestor.code import CodeIngestor, IngestResult


class TestCodeIngestorFiltering:
    """Tests for file filtering in code ingestion."""

    @pytest.fixture
    def temp_code_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create regular Python files
            Path(tmpdir, "main.py").write_text("def main():\n    pass")
            Path(tmpdir, "utils.py").write_text("def helper():\n    pass")

            # Create test files (should be skipped)
            Path(tmpdir, "test_main.py").write_text("def test_something():\n    pass")
            Path(tmpdir, "main.test.py").write_text("def test_something():\n    pass")
            Path(tmpdir, "conftest.py").write_text("import pytest")

            # Create minified JS (should be skipped)
            Path(tmpdir, "app.min.js").write_text("const app={run:function(){console.log('hi')}};")
            Path(tmpdir, "app.js").write_text("const app = {\n  run: function() {\n    console.log('hi');\n  }\n};")

            yield tmpdir

    @pytest.fixture
    def mock_embedder_and_store(self):
        """Create mocked embedder and store."""
        def mock_embed_batch(texts):
            # Return proper numpy array of embeddings
            return np.random.rand(len(texts), 768).astype(np.float32)

        embedder = Mock()
        embedder.embed_code_batch = Mock(side_effect=mock_embed_batch)

        store = Mock()
        store.insert_code_chunks = Mock(return_value=len)

        return embedder, store

    def test_skip_test_files(self, temp_code_dir, mock_embedder_and_store):
        """Test that test files are skipped based on SKIP_FILES patterns."""
        embedder, store = mock_embedder_and_store
        ingestor = CodeIngestor(embedder, store)

        result = ingestor.ingest("test_project", temp_code_dir, lang_filter=["python"])

        # Should skip test_main.py, main.test.py, conftest.py, and app.min.js
        assert result.skipped_files >= 3
        # Only main.py and utils.py should be processed (python files)
        assert result.total_files >= 2

    def test_skip_minified_files(self, temp_code_dir, mock_embedder_and_store):
        """Test that minified files are skipped."""
        embedder, store = mock_embedder_and_store
        ingestor = CodeIngestor(embedder, store)

        result = ingestor.ingest("test_project", temp_code_dir, lang_filter=["javascript"])

        # Should skip app.min.js
        assert result.skipped_files >= 1
        # Only app.js should be processed
        assert result.total_files == 1

    def test_should_skip_file_matching(self):
        """Test file matching logic against SKIP_FILES patterns."""
        ingestor = CodeIngestor(Mock(), Mock())

        # Test files should be skipped
        assert ingestor._should_skip_file("test_main.py")
        assert ingestor._should_skip_file("main.test.py")
        assert ingestor._should_skip_file("conftest.py")

        # Minified files should be skipped
        assert ingestor._should_skip_file("app.min.js")
        assert ingestor._should_skip_file("styles.min.css")

        # Source maps should be skipped
        assert ingestor._should_skip_file("app.js.map")

        # Regular files should NOT be skipped
        assert not ingestor._should_skip_file("main.py")
        assert not ingestor._should_skip_file("utils.js")
        assert not ingestor._should_skip_file("auth.ts")

    def test_skip_files_config_property(self):
        """Test that skip_files_patterns property parses correctly."""
        patterns = settings.skip_files_patterns

        # Should be a list
        assert isinstance(patterns, list)

        # Should contain expected patterns
        assert "*.test.py" in patterns
        assert "*_test.py" in patterns
        assert "test_*.py" in patterns
        assert "conftest.py" in patterns
        assert "*.min.js" in patterns
        assert "*.min.css" in patterns
        assert "*.map" in patterns


class TestCodeIngestorDeduplication:
    """Tests for chunk deduplication in code ingestion."""

    @pytest.fixture
    def mock_embedder_and_store(self):
        """Create mocked embedder and store."""
        def mock_embed_batch(texts):
            # Return proper numpy array of embeddings
            return np.random.rand(len(texts), 768).astype(np.float32)

        embedder = Mock()
        embedder.embed_code_batch = Mock(side_effect=mock_embed_batch)

        store = Mock()
        store.insert_code_chunks = Mock(return_value=len)

        return embedder, store

    def test_deduplicate_identical_chunks(self, mock_embedder_and_store):
        """Test that identical chunks are deduplicated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two files with identical code
            identical_code = "def greet(name):\n    return f'Hello {name}'"
            Path(tmpdir, "file1.py").write_text(identical_code)
            Path(tmpdir, "file2.py").write_text(identical_code)

            embedder, store = mock_embedder_and_store
            ingestor = CodeIngestor(embedder, store)

            result = ingestor.ingest("test_project", tmpdir)

            # Two files processed
            assert result.total_files >= 1
            # Duplicate chunks should be tracked
            assert result.duplicate_chunks >= 1

    def test_hash_chunk(self, mock_embedder_and_store):
        """Test chunk hashing for deduplication."""
        from src.chunker import CodeChunk

        ingestor = CodeIngestor(mock_embedder_and_store[0], mock_embedder_and_store[1])

        chunk1 = CodeChunk(
            code="def hello():\n    pass",
            path="file1.py",
            lang="python",
            start_line=1,
            end_line=2,
        )

        chunk2 = CodeChunk(
            code="def hello():\n    pass",
            path="file2.py",
            lang="python",
            start_line=1,
            end_line=2,
        )

        chunk3 = CodeChunk(
            code="def goodbye():\n    pass",
            path="file1.py",
            lang="python",
            start_line=3,
            end_line=4,
        )

        # Identical code should produce identical hash
        hash1 = ingestor._hash_chunk(chunk1)
        hash2 = ingestor._hash_chunk(chunk2)
        assert hash1 == hash2

        # Different code should produce different hash
        hash3 = ingestor._hash_chunk(chunk3)
        assert hash1 != hash3

    def test_ingest_result_includes_skip_stats(self, mock_embedder_and_store):
        """Test that IngestResult includes skipped_files and duplicate_chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.py").write_text("def main():\n    pass")
            Path(tmpdir, "test.py").write_text("def test():\n    pass")

            embedder, store = mock_embedder_and_store
            ingestor = CodeIngestor(embedder, store)

            result = ingestor.ingest("test_project", tmpdir)

            # Result should have skip statistics
            assert hasattr(result, "skipped_files")
            assert hasattr(result, "duplicate_chunks")
            assert isinstance(result.skipped_files, int)
            assert isinstance(result.duplicate_chunks, int)

    def test_no_duplicate_chunks_for_different_code(self, mock_embedder_and_store):
        """Test that different code produces different hashes and no deduplication."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.py").write_text("def func1():\n    pass")
            Path(tmpdir, "file2.py").write_text("def func2():\n    pass")

            embedder, store = mock_embedder_and_store
            ingestor = CodeIngestor(embedder, store)

            result = ingestor.ingest("test_project", tmpdir)

            # Two unique functions, no duplicates
            assert result.duplicate_chunks == 0
            assert result.total_files == 2


class TestIntegrationFilteringAndDedup:
    """Integration tests for filtering and deduplication together."""

    def test_filtering_then_deduplication(self):
        """Test that files are filtered first, then chunks are deduplicated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create regular files with identical code
            identical_code = "def util():\n    return 42"
            Path(tmpdir, "utils1.py").write_text(identical_code)
            Path(tmpdir, "utils2.py").write_text(identical_code)

            # Create test file with same code (should be filtered out first)
            Path(tmpdir, "test_utils.py").write_text(identical_code)

            def mock_embed_batch(texts):
                return np.random.rand(len(texts), 768).astype(np.float32)

            embedder = Mock()
            embedder.embed_code_batch = Mock(side_effect=mock_embed_batch)
            store = Mock()
            store.insert_code_chunks = Mock()

            ingestor = CodeIngestor(embedder, store)
            result = ingestor.ingest("test_project", tmpdir)

            # test_utils.py is skipped due to filtering
            assert result.skipped_files >= 1
            # At least one file processed (utils1.py)
            assert result.total_files >= 1
            # Duplicate chunk from utils2.py is skipped
            assert result.duplicate_chunks >= 1
