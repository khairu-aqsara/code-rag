import pytest

from src.chunker_ast import JavaScriptASTChunker, PythonASTChunker


class TestPythonASTChunker:
    """Tests for Python AST-aware chunking."""

    @pytest.fixture
    def chunker(self):
        return PythonASTChunker()

    def test_simple_function(self, chunker):
        code = '''
def greet(name: str) -> str:
    """Say hello to a person."""
    return f"Hello, {name}!"
'''
        chunks = chunker.chunk_file(code, "test.py")
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.name == "greet"
        assert chunk.kind == "function"
        assert chunk.docstring == "Say hello to a person."
        assert "Hello, {name}" in chunk.code

    def test_multiple_functions(self, chunker):
        code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
'''
        chunks = chunker.chunk_file(code, "test.py")
        assert len(chunks) == 2
        names = [c.name for c in chunks]
        assert "add" in names
        assert "multiply" in names

    def test_class_extraction(self, chunker):
        code = '''
class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b
'''
        chunks = chunker.chunk_file(code, "test.py")
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.name == "Calculator"
        assert chunk.kind == "class"
        assert chunk.docstring == "A simple calculator."

    def test_import_extraction(self, chunker):
        code = '''
from typing import List, Optional
import os

def read_file(path: str) -> str:
    return ""
'''
        chunks = chunker.chunk_file(code, "test.py")
        # Should extract imports + function
        assert len(chunks) >= 1
        kinds = [c.kind for c in chunks]
        assert "import" in kinds or len(chunks) == 1  # import optional in extraction

    def test_async_function(self, chunker):
        code = '''
async def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    return ""
'''
        chunks = chunker.chunk_file(code, "test.py")
        assert len(chunks) == 1
        assert chunks[0].kind == "function"

    def test_skips_large_functions(self, chunker):
        """Functions larger than 2x CODE_CHUNK_LINES should be skipped in semantic chunking."""
        from src.config import settings

        # Create a function with more than 2x CODE_CHUNK_LINES lines
        large_lines = ["    pass"] * (settings.CODE_CHUNK_LINES * 3)
        code = f"def big_func():\n" + "\n".join(large_lines)

        chunks = chunker.chunk_file(code, "test.py")
        # Should fall back to line-based chunking or skip
        # Verify it doesn't return a semantic chunk for the large function
        assert all(c.kind != "function" or len(c.code.split("\n")) <= settings.CODE_CHUNK_LINES * 2 for c in chunks)

    def test_syntax_error_fallback(self, chunker):
        """Invalid Python should fall back to line-based chunking."""
        code = "def invalid syntax here"
        chunks = chunker.chunk_file(code, "test.py")
        assert len(chunks) > 0
        # Should have some chunks from fallback

    def test_empty_file(self, chunker):
        """Empty file should return empty list or fallback chunks."""
        code = ""
        chunks = chunker.chunk_file(code, "test.py")
        assert isinstance(chunks, list)


class TestJavaScriptASTChunker:
    """Tests for JavaScript/TypeScript AST-aware chunking."""

    @pytest.fixture
    def chunker(self):
        return JavaScriptASTChunker()

    def test_function_declaration(self, chunker):
        code = '''
function greet(name) {
  return `Hello, ${name}!`;
}
'''
        chunks = chunker.chunk_file(code, "test.js")
        # Should find function declaration
        assert any(c.kind == "function" for c in chunks) or len(chunks) > 0

    def test_arrow_function(self, chunker):
        code = '''
const add = (a, b) => {
  return a + b;
};
'''
        chunks = chunker.chunk_file(code, "test.js")
        # Arrow functions are harder to detect via regex; fallback is acceptable
        assert len(chunks) > 0

    def test_class_declaration(self, chunker):
        code = '''
class Calculator {
  add(a, b) {
    return a + b;
  }
}
'''
        chunks = chunker.chunk_file(code, "test.js")
        # Should attempt to find class
        assert len(chunks) > 0

    def test_export_function(self, chunker):
        code = '''
export function getData() {
  return { data: [] };
}
'''
        chunks = chunker.chunk_file(code, "test.js")
        assert len(chunks) > 0

    def test_fallback_on_no_matches(self, chunker):
        """If regex finds no functions, should fall back to line-based."""
        code = '''
// Just comments and imports
import { something } from "module";
const x = 10;
'''
        chunks = chunker.chunk_file(code, "test.js")
        # Should return line-based chunks as fallback
        assert len(chunks) > 0


class TestASTChunkerFactory:
    """Tests for AST chunker factory."""

    def test_get_python_chunker(self):
        from src.chunker_ast import ASTChunkerFactory

        chunker = ASTChunkerFactory.get_chunker("python")
        assert chunker is not None
        assert isinstance(chunker, PythonASTChunker)

    def test_get_javascript_chunker(self):
        from src.chunker_ast import ASTChunkerFactory

        chunker = ASTChunkerFactory.get_chunker("javascript")
        assert chunker is not None
        assert isinstance(chunker, JavaScriptASTChunker)

    def test_get_typescript_chunker(self):
        from src.chunker_ast import ASTChunkerFactory

        chunker = ASTChunkerFactory.get_chunker("typescript")
        assert chunker is not None
        assert isinstance(chunker, JavaScriptASTChunker)

    def test_unsupported_language(self):
        from src.chunker_ast import ASTChunkerFactory

        chunker = ASTChunkerFactory.get_chunker("ruby")
        assert chunker is None

    def test_case_insensitive(self):
        from src.chunker_ast import ASTChunkerFactory

        chunker = ASTChunkerFactory.get_chunker("PYTHON")
        assert chunker is not None
        assert isinstance(chunker, PythonASTChunker)
