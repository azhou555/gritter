from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gritter.agent.tools import ToolContext, dispatch_tool, tool_read_file
from gritter.models.chunk import CodeChunk
from gritter.models.query import RetrievalResult


def _make_ctx(tmp_path: Path, retriever=None, config=None) -> ToolContext:
    from gritter.models.config import GritterConfig
    return ToolContext(
        repo_root=tmp_path,
        retriever=retriever or MagicMock(),
        config=config or GritterConfig(data_dir=tmp_path / "data"),
        index_name="test_index",
    )


def test_read_file_returns_numbered_lines(tmp_path):
    (tmp_path / "foo.py").write_text("line1\nline2\nline3\n")
    ctx = _make_ctx(tmp_path)
    result = tool_read_file(ctx, "foo.py")
    assert "1: line1" in result
    assert "2: line2" in result
    assert "3: line3" in result


def test_read_file_respects_line_range(tmp_path):
    (tmp_path / "foo.py").write_text("a\nb\nc\nd\ne\n")
    ctx = _make_ctx(tmp_path)
    result = tool_read_file(ctx, "foo.py", start_line=2, end_line=3)
    assert "2: b" in result
    assert "3: c" in result
    assert "1: a" not in result
    assert "4: d" not in result


def test_read_file_rejects_path_traversal(tmp_path):
    (tmp_path / "foo.py").write_text("secret")
    outside = tmp_path.parent / "outside.py"
    outside.write_text("nope")
    ctx = _make_ctx(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        tool_read_file(ctx, "../outside.py")


def test_read_file_rejects_missing_file(tmp_path):
    ctx = _make_ctx(tmp_path)
    with pytest.raises(ValueError, match="not a file"):
        tool_read_file(ctx, "nonexistent.py")


def test_search_code_formats_results(tmp_path):
    chunk = CodeChunk(
        content="def foo(): pass",
        file_path="foo.py",
        language="python",
        symbol_name="foo",
        symbol_type="function",
        start_line=1,
        end_line=1,
    )
    retriever = MagicMock()
    retriever.search.return_value = [RetrievalResult(chunk=chunk, score=0.9)]
    ctx = _make_ctx(tmp_path, retriever=retriever)

    result = dispatch_tool(ctx, "search_code", {"query": "foo function"})
    retriever.search.assert_called_once_with("foo function")
    assert "foo.py" in result
    assert "def foo(): pass" in result


def test_search_code_no_results(tmp_path):
    retriever = MagicMock()
    retriever.search.return_value = []
    ctx = _make_ctx(tmp_path, retriever=retriever)
    result = dispatch_tool(ctx, "search_code", {"query": "nothing"})
    assert result == "No results found."


def test_grep_python_fallback_finds_match(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)  # force python fallback
    (tmp_path / "foo.py").write_text("def validate_token(token):\n    return True\n")
    ctx = _make_ctx(tmp_path)
    result = dispatch_tool(ctx, "grep", {"pattern": "validate_token"})
    assert "foo.py:1:" in result


def test_grep_no_matches(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / "foo.py").write_text("nothing interesting\n")
    ctx = _make_ctx(tmp_path)
    result = dispatch_tool(ctx, "grep", {"pattern": "zzz_no_match"})
    assert result == "No matches found."


def test_list_symbols_returns_python_functions(tmp_path):
    (tmp_path / "foo.py").write_text("def alpha():\n    pass\n\n\ndef beta():\n    pass\n")
    ctx = _make_ctx(tmp_path)
    result = dispatch_tool(ctx, "list_symbols", {"path": "foo.py"})
    assert "alpha" in result
    assert "beta" in result


def test_list_symbols_unsupported_language(tmp_path):
    (tmp_path / "foo.txt").write_text("plain text")
    ctx = _make_ctx(tmp_path)
    result = dispatch_tool(ctx, "list_symbols", {"path": "foo.txt"})
    assert "No language support" in result


def test_dispatch_tool_unknown_name_returns_error_string(tmp_path):
    ctx = _make_ctx(tmp_path)
    result = dispatch_tool(ctx, "not_a_real_tool", {})
    assert result.startswith("Error:")


def test_dispatch_tool_catches_exceptions(tmp_path):
    ctx = _make_ctx(tmp_path)
    result = dispatch_tool(ctx, "read_file", {"path": "missing.py"})
    assert result.startswith("Error:")
