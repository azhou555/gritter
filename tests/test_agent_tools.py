from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gritter.agent.tools import (
    ToolContext,
    _grep_python,
    _grep_ripgrep,
    dispatch_tool,
    tool_read_file,
    tool_reindex,
)
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


# --- Finding 1: grep's Python fallback must not leak files outside repo_root ---


def test_grep_python_fallback_rejects_path_traversal_glob(tmp_path):
    # Set up repo_root as a subdirectory, with a sibling file outside it that
    # contains a matching secret the model should never see.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "inside.py").write_text("nothing_matching_here\n")
    outside = tmp_path / "outside_secret.py"
    outside.write_text("TOP_SECRET_TOKEN = 'leak-me'\n")

    result = _grep_python(repo_root, "TOP_SECRET_TOKEN", "../*")
    assert "leak-me" not in result
    assert "TOP_SECRET_TOKEN" not in result
    assert result == "No matches found."


def test_grep_python_fallback_still_finds_matches_inside_repo(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "inside.py").write_text("def validate_token(token):\n    return True\n")
    result = _grep_python(repo_root, "validate_token", None)
    assert "inside.py:1:" in result


# --- Finding 3: ripgrep argument injection via unescaped pattern ---


def test_grep_ripgrep_uses_double_dash_to_prevent_flag_injection(tmp_path, monkeypatch):
    captured_args = {}

    def fake_run(args, cwd, capture_output, text, timeout):
        captured_args["args"] = args

        class Result:
            returncode = 1
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("gritter.agent.tools.subprocess.run", fake_run)
    _grep_ripgrep(tmp_path, "--files", None)

    args = captured_args["args"]
    assert "--" in args
    dash_index = args.index("--")
    assert args[dash_index + 1] == "--files"
    # confirm the pattern is not present before the "--" separator (i.e. not
    # parseable as a standalone flag)
    assert "--files" not in args[:dash_index]


# --- Finding 4: reindex must refresh the live retriever, not just the on-disk index ---


def test_reindex_reassigns_ctx_retriever(tmp_path):
    old_retriever = MagicMock(name="old_retriever")
    new_retriever = MagicMock(name="new_retriever")
    ctx = _make_ctx(tmp_path, retriever=old_retriever)

    with patch("gritter.indexing.pipeline.run_indexing_pipeline", return_value={"files": 1}) as mock_run, \
         patch("gritter.retrieval.hybrid.HybridRetriever.from_config", return_value=new_retriever) as mock_from_config, \
         patch("gritter.providers.embeddings.get_embedding_provider", return_value=MagicMock()):
        result = tool_reindex(ctx)

    assert ctx.retriever is new_retriever
    assert ctx.retriever is not old_retriever
    assert "Reindexed" in result
    mock_run.assert_called_once()
    mock_from_config.assert_called_once_with(ctx.index_name, ctx.config)


# --- Finding 5: tool output must be bounded ---


def test_read_file_output_is_truncated(tmp_path):
    (tmp_path / "big.py").write_text("\n".join(f"line {i}" for i in range(5000)))
    ctx = _make_ctx(tmp_path)
    result = tool_read_file(ctx, "big.py")
    assert len(result) <= 8000 + len("\n… [truncated]") + 1
    assert result.endswith("\n… [truncated]")


def test_grep_python_output_is_truncated(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    repo_root = tmp_path
    # _grep_python caps at 50 matches, so use long lines to exceed the
    # 8000-char truncation limit within that cap.
    long_suffix = "x" * 300
    (repo_root / "big.py").write_text(
        "\n".join(f"needle_match_{i}_{long_suffix}" for i in range(2000))
    )
    ctx = _make_ctx(repo_root)
    result = dispatch_tool(ctx, "grep", {"pattern": "needle_match"})
    assert result.endswith("\n… [truncated]")
    assert len(result) <= 8000 + len("\n… [truncated]") + 1
