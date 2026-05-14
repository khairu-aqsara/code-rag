"""Unit tests for DocIngestor — uses mock embedder and vector_store."""
import pytest

from src.ingestor.docs import DocIngestor


class TestDocIngestor:
    @pytest.fixture
    def ingestor(self, mock_embedder, mock_vector_store):
        return DocIngestor(mock_embedder, mock_vector_store)

    def test_ingest_markdown_file(self, ingestor, mock_vector_store, tmp_path):
        md = tmp_path / "readme.md"
        md.write_text("# Hello\n\nThis is documentation.\n\nAnother paragraph here.")

        result = ingestor.ingest("proj", [str(md)])

        assert result.total_files == 1
        assert result.total_chunks >= 1
        assert result.errors == []
        mock_vector_store.insert_doc_chunks.assert_called_once()

    def test_ingest_txt_file(self, ingestor, mock_vector_store, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("Some plain text documentation.\n\nAnother section.")

        result = ingestor.ingest("proj", [str(txt)])

        assert result.total_files == 1

    def test_ingest_html_strips_tags(self, ingestor, mock_vector_store, tmp_path):
        html = tmp_path / "page.html"
        html.write_text("<html><body><p>Hello world</p><script>evil()</script></body></html>")

        result = ingestor.ingest("proj", [str(html)])

        assert result.total_files == 1
        # The chunk content should not contain HTML tags
        call_args = mock_vector_store.insert_doc_chunks.call_args
        chunks = call_args[0][1]
        for chunk in chunks:
            assert "<" not in chunk.content

    def test_ignores_non_doc_extensions(self, ingestor, mock_vector_store, tmp_path):
        py_file = tmp_path / "script.py"
        py_file.write_text("x = 1")

        result = ingestor.ingest("proj", [str(py_file)])

        assert result.total_files == 0
        mock_vector_store.insert_doc_chunks.assert_not_called()

    def test_tags_passed_to_chunks(self, ingestor, mock_vector_store, tmp_path):
        md = tmp_path / "api.md"
        md.write_text("API documentation.\n\nEndpoints are listed here.")

        ingestor.ingest("proj", [str(md)], tags=["api", "v2"])

        call_args = mock_vector_store.insert_doc_chunks.call_args
        chunks = call_args[0][1]
        for chunk in chunks:
            assert chunk.tags == ["api", "v2"]

    def test_glob_pattern_expansion(self, ingestor, mock_vector_store, tmp_path):
        for i in range(3):
            (tmp_path / f"doc{i}.md").write_text(f"Document {i} content here.\n\nMore text.")

        pattern = str(tmp_path / "*.md")
        result = ingestor.ingest("proj", [pattern])

        assert result.total_files == 3

    def test_unreadable_file_logged_not_crashed(self, ingestor, mock_vector_store, tmp_path):
        good = tmp_path / "good.md"
        good.write_text("Good content here.")
        bad_path = str(tmp_path / "nonexistent.md")

        result = ingestor.ingest("proj", [str(good), bad_path])

        assert result.total_files == 1
        assert len(result.errors) == 1

    def test_empty_file_skipped(self, ingestor, mock_vector_store, tmp_path):
        empty = tmp_path / "empty.md"
        empty.write_text("")

        result = ingestor.ingest("proj", [str(empty)])

        assert result.total_files == 0
        mock_vector_store.insert_doc_chunks.assert_not_called()
