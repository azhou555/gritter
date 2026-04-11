from gritter.models.chunk import CodeChunk
from gritter.indexing.heuristic_chunker import chunk_file_heuristic


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
