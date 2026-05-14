"""Unit tests for CodeIngestor — uses mock embedder and vector_store."""
import os
import pytest
import tempfile

from src.ingestor.code import CodeIngestor


class TestCodeIngestor:
    @pytest.fixture
    def ingestor(self, mock_embedder, mock_vector_store):
        return CodeIngestor(mock_embedder, mock_vector_store)

    def test_ingest_python_file(self, ingestor, mock_vector_store, tmp_path):
        src = tmp_path / "hello.py"
        src.write_text("def hello():\n    return 'world'\n")

        result = ingestor.ingest("proj", str(tmp_path))

        assert result.total_files == 1
        assert result.total_chunks >= 1
        assert result.errors == []
        mock_vector_store.insert_code_chunks.assert_called_once()

    def test_ingest_ignores_unknown_extensions(self, ingestor, mock_vector_store, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n")

        result = ingestor.ingest("proj", str(tmp_path))

        assert result.total_files == 0
        mock_vector_store.insert_code_chunks.assert_not_called()

    def test_ingest_skips_git_dir(self, ingestor, mock_vector_store, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]\nrepositoryformatversion = 0\n")
        (tmp_path / "app.py").write_text("x = 1\n")

        result = ingestor.ingest("proj", str(tmp_path))

        # Only app.py should be ingested, not .git/config
        assert result.total_files == 1

    def test_lang_filter_restricts_files(self, ingestor, mock_vector_store, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n")
        (tmp_path / "app.js").write_text("const x = 1;\n")

        result = ingestor.ingest("proj", str(tmp_path), lang_filter=["python"])

        assert result.total_files == 1
        # Only the python file
        call_args = mock_vector_store.insert_code_chunks.call_args_list
        langs = []
        for call in call_args:
            chunks = call[0][1]  # second positional arg
            langs.extend([c.lang for c in chunks])
        assert all(lang == "python" for lang in langs)

    def test_ingest_multiple_files(self, ingestor, mock_vector_store, tmp_path):
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"x = {i}\n")

        result = ingestor.ingest("proj", str(tmp_path))

        assert result.total_files == 5
        assert result.errors == []

    def test_unreadable_file_logged_not_crashed(self, ingestor, mock_vector_store, tmp_path):
        import stat
        (tmp_path / "good.py").write_text("x = 1\n")
        bad = tmp_path / "bad.py"
        bad.write_text("x = 2\n")
        # Remove read permission so open() raises PermissionError
        bad.chmod(0o000)

        try:
            result = ingestor.ingest("proj", str(tmp_path))
            # Should still process good.py and log error for bad.py
            assert result.total_files == 1
            assert len(result.errors) == 1
        finally:
            bad.chmod(0o644)  # restore for cleanup

    def test_result_includes_duration(self, ingestor, tmp_path):
        (tmp_path / "a.py").write_text("pass\n")
        result = ingestor.ingest("proj", str(tmp_path))
        assert result.duration_seconds >= 0
