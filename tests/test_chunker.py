import pytest
from gritter.models.chunk import CodeChunk
from gritter.indexing.heuristic_chunker import chunk_file_heuristic
from gritter.indexing.ast_chunker import chunk_file_ast


def test_chunk_id_is_deterministic():
    chunk = CodeChunk(
        content="def foo(): pass",
        file_path="src/foo.py",
        language="python",
        symbol_name="foo",
        symbol_type="function",
        start_line=1,
        end_line=1,
    )
    chunk2 = CodeChunk(
        content="def foo(): pass",
        file_path="src/foo.py",
        language="python",
        symbol_name="foo",
        symbol_type="function",
        start_line=1,
        end_line=1,
    )
    assert chunk.chunk_id == chunk2.chunk_id
    assert len(chunk.chunk_id) == 16


def test_embedding_text_prepends_imports():
    chunk = CodeChunk(
        content="def foo(): pass",
        file_path="src/foo.py",
        language="python",
        symbol_name="foo",
        symbol_type="function",
        start_line=1,
        end_line=1,
        imports=["import os", "from pathlib import Path"],
    )
    text = chunk.embedding_text()
    assert text.startswith("import os\nfrom pathlib import Path\n\n")
    assert "def foo(): pass" in text


def test_embedding_text_no_imports():
    chunk = CodeChunk(
        content="def foo(): pass",
        file_path="src/foo.py",
        language="python",
        symbol_name="foo",
        symbol_type="function",
        start_line=1,
        end_line=1,
    )
    assert chunk.embedding_text() == "def foo(): pass"


def test_heuristic_chunker_splits_top_level_blocks():
    content = """def foo():
    x = 1
    return x


def bar():
    y = 2
    return y
"""
    # min_tokens=1 isolates splitting behavior from the merge guardrail —
    # the blocks are short (~13 tokens each) so the default min_tokens=50 would merge them.
    chunks = chunk_file_heuristic(content, "test.go", "go", min_tokens=1)
    assert len(chunks) == 2
    assert "def foo" in chunks[0].content
    assert "def bar" in chunks[1].content


def test_heuristic_chunker_doesnt_split_internal_blank_lines():
    content = """def foo():
    x = 1

    y = 2
    return x + y
"""
    chunks = chunk_file_heuristic(content, "test.go", "go")
    # Blank line inside the function should NOT create a split
    assert len(chunks) == 1
    assert "y = 2" in chunks[0].content


def test_heuristic_chunker_merges_small_chunks():
    # Two tiny blocks that each fall under min_tokens should be merged
    content = "x = 1\n\ny = 2\n"
    chunks = chunk_file_heuristic(content, "test.go", "go", min_tokens=10)
    # Both are tiny — should be merged into one chunk
    assert len(chunks) == 1


def test_heuristic_chunker_populates_metadata():
    content = "def foo():\n    return 1\n"
    chunks = chunk_file_heuristic(content, "path/to/file.go", "go")
    assert chunks[0].file_path == "path/to/file.go"
    assert chunks[0].language == "go"
    assert chunks[0].symbol_name is None
    assert chunks[0].symbol_type == "module"
    assert chunks[0].start_line == 1



class TestPythonASTChunker:
    def test_extracts_top_level_functions(self, python_fixture_path):
        content = python_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(python_fixture_path), "python")
        function_chunks = [c for c in chunks if c.symbol_type == "function"]
        names = {c.symbol_name for c in function_chunks}
        assert "add" in names
        assert "subtract" in names

    def test_extracts_class(self, python_fixture_path):
        content = python_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(python_fixture_path), "python")
        class_or_method_chunks = [c for c in chunks if c.symbol_type in ("class", "method")]
        assert any(c.symbol_name == "Calculator" for c in class_or_method_chunks)

    def test_chunk_line_numbers_are_correct(self, python_fixture_path):
        content = python_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(python_fixture_path), "python")
        add_chunk = next(c for c in chunks if c.symbol_name == "add")
        assert add_chunk.start_line >= 1
        assert add_chunk.end_line >= add_chunk.start_line
        assert "def add" in add_chunk.content

    def test_imports_are_captured(self, python_fixture_path):
        content = python_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(python_fixture_path), "python")
        assert all(len(c.imports) > 0 for c in chunks)
        assert any("import os" in imp for chunk in chunks for imp in chunk.imports)

    def test_syntax_error_raises_value_error(self):
        with pytest.raises(ValueError, match="Parse error"):
            chunk_file_ast("def foo(", "bad.py", "python")


class TestTypeScriptASTChunker:
    def test_extracts_exported_function(self, typescript_fixture_path):
        content = typescript_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(typescript_fixture_path), "typescript")
        names = {c.symbol_name for c in chunks}
        assert "greet" in names

    def test_extracts_class(self, typescript_fixture_path):
        content = typescript_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(typescript_fixture_path), "typescript")
        assert any(c.symbol_name == "Formatter" for c in chunks)


class TestRustASTChunker:
    def test_extracts_struct(self, rust_fixture_path):
        content = rust_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(rust_fixture_path), "rust")
        names = {c.symbol_name for c in chunks}
        assert "Point" in names

    def test_extracts_standalone_function(self, rust_fixture_path):
        content = rust_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(rust_fixture_path), "rust")
        names = {c.symbol_name for c in chunks}
        assert "origin" in names

    def test_extracts_impl_block(self, rust_fixture_path):
        content = rust_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(rust_fixture_path), "rust")
        assert any("impl" in c.content for c in chunks)
