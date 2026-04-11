from gritter.models.chunk import CodeChunk


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
