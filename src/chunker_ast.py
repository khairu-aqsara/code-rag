import ast
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from .chunker import CodeChunk
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ASTItem:
    """Represents a top-level AST item (function, class, etc.)"""

    name: str
    kind: str  # "function", "class", "import", "constant", "other"
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive
    docstring: Optional[str] = None


class PythonASTChunker:
    """AST-aware chunker for Python files.

    Extracts top-level functions, classes, and imports as individual chunks
    to preserve semantic context. Falls back to line-based chunking for large
    items or parse errors.
    """

    def chunk_file(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse Python file and extract semantic chunks.

        Returns individual functions/classes as chunks. Falls back to
        line-based chunking if AST parsing fails.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Failed to parse {file_path}: {e}. Using line-based chunking.")
            return self._chunk_lines_fallback(content, file_path)

        lines = content.split("\n")
        chunks: List[CodeChunk] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._extract_function(node, lines, file_path)
                if chunk:
                    chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._extract_class(node, lines, file_path)
                if chunk:
                    chunks.append(chunk)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                chunk = self._extract_import(node, lines, file_path)
                if chunk:
                    chunks.append(chunk)

        # If no chunks extracted (empty file or only comments), use line-based
        if not chunks:
            return self._chunk_lines_fallback(content, file_path)

        return chunks

    def _extract_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, lines: List[str], file_path: str
    ) -> Optional[CodeChunk]:
        """Extract a function as a single chunk."""
        start_line = node.lineno - 1  # ast is 0-indexed internally
        end_line = node.end_lineno or len(lines)  # end_lineno is 1-indexed

        code_lines = lines[start_line:end_line]
        code = "\n".join(code_lines)

        # Skip if too large (likely multi-method class or complex function)
        if len(code_lines) > settings.CODE_CHUNK_LINES * 2:
            logger.debug(f"Function {node.name} in {file_path} too large ({len(code_lines)} lines), skipping semantic chunk")
            return None

        docstring = ast.get_docstring(node)

        return CodeChunk(
            code=code,
            path=file_path,
            lang="python",
            start_line=start_line + 1,
            end_line=end_line,
            name=node.name,
            kind="function",
            docstring=docstring,
        )

    def _extract_class(
        self, node: ast.ClassDef, lines: List[str], file_path: str
    ) -> Optional[CodeChunk]:
        """Extract a class as a single chunk."""
        start_line = node.lineno - 1
        end_line = node.end_lineno or len(lines)

        code_lines = lines[start_line:end_line]
        code = "\n".join(code_lines)

        # Skip if too large
        if len(code_lines) > settings.CODE_CHUNK_LINES * 3:
            logger.debug(f"Class {node.name} in {file_path} too large ({len(code_lines)} lines), skipping semantic chunk")
            return None

        docstring = ast.get_docstring(node)

        return CodeChunk(
            code=code,
            path=file_path,
            lang="python",
            start_line=start_line + 1,
            end_line=end_line,
            name=node.name,
            kind="class",
            docstring=docstring,
        )

    def _extract_import(self, node: ast.Import | ast.ImportFrom, lines: List[str], file_path: str) -> Optional[CodeChunk]:
        """Extract import statements as a chunk."""
        start_line = node.lineno - 1
        end_line = node.end_lineno or len(lines)

        code_lines = lines[start_line:end_line]
        code = "\n".join(code_lines)

        return CodeChunk(
            code=code,
            path=file_path,
            lang="python",
            start_line=start_line + 1,
            end_line=end_line,
            name="",
            kind="import",
            docstring=None,
        )

    def _chunk_lines_fallback(self, content: str, file_path: str) -> List[CodeChunk]:
        """Fallback to line-based chunking if AST extraction fails."""
        from .chunker import CodeChunker
        return CodeChunker().chunk_file(content, file_path, "python")


class JavaScriptASTChunker:
    """AST-aware chunker for JavaScript/TypeScript files.

    Uses regex-based extraction since Python's ast module won't parse JS.
    Extracts function and class declarations as individual chunks.
    """

    # Regex patterns for function/class detection
    FUNCTION_PATTERN = re.compile(
        r"^\s*(async\s+)?(export\s+)?(default\s+)?(function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)\s*)?=>|class\s+(\w+))",
        re.MULTILINE
    )

    def chunk_file(self, content: str, file_path: str) -> List[CodeChunk]:
        """Extract functions and classes from JavaScript/TypeScript using regex.

        Identifies function/class declarations and treats each as a semantic unit.
        """
        chunks: List[CodeChunk] = []
        lines = content.split("\n")

        # Find all function/class definitions
        for match in self.FUNCTION_PATTERN.finditer(content):
            start_pos = match.start()
            start_line = content[:start_pos].count("\n")

            # Extract function/class name
            name = match.group(5) or match.group(6) or match.group(7) or ""
            kind = "class" if match.group(7) else "function"

            # Find the matching closing brace
            end_line = self._find_block_end(lines, start_line)

            if end_line is None or (end_line - start_line) > settings.CODE_CHUNK_LINES * 2:
                continue

            code_lines = lines[start_line:end_line + 1]
            code = "\n".join(code_lines)

            chunks.append(
                CodeChunk(
                    code=code,
                    path=file_path,
                    lang="javascript",
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    name=name,
                    kind=kind,
                    docstring=None,
                )
            )

        # If no semantic chunks found, fall back to line-based
        if not chunks:
            from .chunker import CodeChunker
            return CodeChunker().chunk_file(content, file_path, "javascript")

        return chunks

    @staticmethod
    def _find_block_end(lines: List[str], start_line: int) -> Optional[int]:
        """Find the end line of a code block starting at start_line."""
        brace_count = 0
        found_opening = False

        for i in range(start_line, len(lines)):
            line = lines[i]
            for char in line:
                if char == "{":
                    brace_count += 1
                    found_opening = True
                elif char == "}":
                    brace_count -= 1
                    if found_opening and brace_count == 0:
                        return i

        return None


class ASTChunkerFactory:
    """Factory for creating language-appropriate AST chunkers."""

    _chunkers = {
        "python": PythonASTChunker,
        "javascript": JavaScriptASTChunker,
        "typescript": JavaScriptASTChunker,
    }

    @classmethod
    def get_chunker(cls, lang: str) -> Optional[object]:
        """Get AST chunker for the given language, or None if not supported."""
        chunker_class = cls._chunkers.get(lang.lower())
        if chunker_class:
            return chunker_class()
        return None
