"""Unit tests for CodeChunker and DocChunker."""
import pytest

from src.chunker import CodeChunk, CodeChunker, DocChunk, DocChunker


@pytest.fixture
def code_chunker():
    return CodeChunker()


@pytest.fixture
def doc_chunker():
    return DocChunker()


class TestCodeChunker:
    def test_single_chunk_short_file(self, code_chunker):
        content = "\n".join(f"line {i}" for i in range(10))
        chunks = code_chunker.chunk_file(content, "test.py", "python")
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 10
        assert chunks[0].lang == "python"
        assert chunks[0].path == "test.py"

    def test_multiple_chunks_large_file(self, code_chunker):
        content = "\n".join(f"line {i}" for i in range(200))
        chunks = code_chunker.chunk_file(content, "big.py", "python")
        assert len(chunks) > 1
        # First chunk starts at line 1
        assert chunks[0].start_line == 1

    def test_chunk_overlap(self, code_chunker):
        """Chunks should overlap: chunk N end_line > chunk N+1 start_line."""
        content = "\n".join(f"line {i}" for i in range(200))
        chunks = code_chunker.chunk_file(content, "test.py", "python")
        for i in range(len(chunks) - 1):
            # With 80-line chunks and 20-line overlap, next chunk starts 60 after current
            assert chunks[i + 1].start_line > chunks[i].start_line
            assert chunks[i + 1].start_line <= chunks[i].end_line  # overlap exists

    def test_empty_file_returns_no_chunks(self, code_chunker):
        chunks = code_chunker.chunk_file("", "empty.py", "python")
        assert chunks == []

    def test_whitespace_only_skipped(self, code_chunker):
        content = "\n\n   \n\t\n"
        chunks = code_chunker.chunk_file(content, "blank.py", "python")
        assert chunks == []

    def test_line_numbers_are_1_indexed(self, code_chunker):
        content = "\n".join(f"line {i}" for i in range(5))
        chunks = code_chunker.chunk_file(content, "test.py", "python")
        assert chunks[0].start_line == 1  # not 0

    def test_chunk_contains_correct_code(self, code_chunker):
        lines = [f"line {i}" for i in range(5)]
        content = "\n".join(lines)
        chunks = code_chunker.chunk_file(content, "test.py", "python")
        assert "line 0" in chunks[0].code
        assert "line 4" in chunks[0].code


class TestDocChunker:
    def test_single_paragraph(self, doc_chunker):
        content = "This is a simple sentence. Another sentence follows. End."
        chunks = doc_chunker.chunk_doc(content, "test.md")
        assert len(chunks) >= 1
        assert chunks[0].source == "test.md"

    def test_multiple_paragraphs(self, doc_chunker):
        content = "Para one content here.\n\nPara two content here.\n\nPara three here."
        chunks = doc_chunker.chunk_doc(content, "multi.md")
        assert len(chunks) >= 1

    def test_tags_preserved(self, doc_chunker):
        content = "Some documentation text with enough content to form a chunk."
        chunks = doc_chunker.chunk_doc(content, "api.md", tags=["api", "auth"])
        for chunk in chunks:
            assert chunk.tags == ["api", "auth"]

    def test_no_tags_is_none(self, doc_chunker):
        content = "Content without tags."
        chunks = doc_chunker.chunk_doc(content, "readme.md")
        for chunk in chunks:
            assert chunk.tags is None

    def test_empty_content_returns_no_chunks(self, doc_chunker):
        chunks = doc_chunker.chunk_doc("", "empty.md")
        assert chunks == []

    def test_whitespace_only_returns_no_chunks(self, doc_chunker):
        chunks = doc_chunker.chunk_doc("   \n\n   ", "blank.md")
        assert chunks == []

    def test_long_content_splits_into_multiple_chunks(self, doc_chunker):
        # ~600 words should exceed DOC_CHUNK_SIZE=512 and split
        sentences = ["This is sentence number {i} with some extra words for padding." for i in range(80)]
        content = " ".join(sentences)
        chunks = doc_chunker.chunk_doc(content, "long.md")
        assert len(chunks) > 1

    def test_source_preserved_across_chunks(self, doc_chunker):
        sentences = [f"Sentence {i} with enough words to build up tokens." for i in range(100)]
        content = " ".join(sentences)
        chunks = doc_chunker.chunk_doc(content, "api.md")
        for chunk in chunks:
            assert chunk.source == "api.md"
