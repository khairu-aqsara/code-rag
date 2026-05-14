import re
from dataclasses import dataclass, field
from typing import List, Optional

from .config import settings


@dataclass
class CodeChunk:
    code: str
    path: str
    lang: str
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed, inclusive
    name: str = ""  # function/class name, empty for non-semantic chunks
    kind: str = "code"  # "function", "class", "import", "code"
    docstring: Optional[str] = None  # docstring if available
    # Canonical path where this chunk's content first appeared (set by ingestor on dedup).
    # When None, this chunk is itself canonical (same as `path`).
    original_path: Optional[str] = None


@dataclass
class DocChunk:
    content: str
    source: str           # e.g., "README.md", "docs/api.md"
    tags: Optional[List[str]] = None  # None means no tags


class CodeChunker:
    """Splits source files into overlapping line-based chunks."""

    def chunk_file(self, content: str, file_path: str, lang: str) -> List[CodeChunk]:
        """Split file content into overlapping chunks of CODE_CHUNK_LINES lines.

        Chunks overlap by CODE_OVERLAP_LINES to preserve context at boundaries.
        Line numbers are 1-indexed to match editor conventions.
        """
        lines = content.split("\n")
        chunk_size = settings.CODE_CHUNK_LINES
        overlap = settings.CODE_OVERLAP_LINES
        step = chunk_size - overlap

        chunks: List[CodeChunk] = []
        i = 0
        while i < len(lines):
            chunk_lines = lines[i : i + chunk_size]
            # Skip chunks that are entirely whitespace/empty
            if not any(line.strip() for line in chunk_lines):
                i += step
                continue

            start_line = i + 1  # convert to 1-indexed
            end_line = min(i + chunk_size, len(lines))  # 1-indexed inclusive
            chunks.append(
                CodeChunk(
                    code="\n".join(chunk_lines),
                    path=file_path,
                    lang=lang,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            i += step

        return chunks


class DocChunker:
    """Splits documents into paragraph/sentence-based chunks."""

    def chunk_doc(self, content: str, source: str, tags: List[str] | None = None) -> List[DocChunk]:
        """Split document content into chunks of approximately DOC_CHUNK_SIZE tokens.

        Splits on paragraph boundaries first, then groups sentences until the
        approximate token limit is reached. Uses regex — no NLTK dependency.
        """
        chunks: List[DocChunk] = []
        paragraphs = re.split(r"\n\n+", content.strip())

        current_sentences: List[str] = []
        current_token_count = 0

        def flush() -> None:
            text = " ".join(current_sentences).strip()
            if text:
                chunks.append(DocChunk(content=text, source=source, tags=tags))
            current_sentences.clear()
            nonlocal current_token_count
            current_token_count = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Split paragraph into sentences on .  !  ? boundaries
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Approximate token count: words (whitespace-split)
                token_est = len(sentence.split())

                if current_token_count + token_est > settings.DOC_CHUNK_SIZE and current_sentences:
                    flush()

                current_sentences.append(sentence)
                current_token_count += token_est

        # Flush remaining sentences
        flush()

        return chunks
