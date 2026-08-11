from __future__ import annotations

from gritter.generation.prompts import build_context_block
from gritter.models.chunk import CodeChunk
from gritter.models.query import RetrievalResult


def _make_chunk(
    file_path: str = "src/auth/jwt.py",
    start_line: int = 15,
    end_line: int = 45,
    symbol_name: str | None = "validate_token",
    symbol_type: str | None = "function",
    content: str = "def validate_token(token): pass",
) -> CodeChunk:
    return CodeChunk(
        content=content,
        file_path=file_path,
        language="python",
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        start_line=start_line,
        end_line=end_line,
    )


def _make_result(**kwargs) -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(**kwargs), score=0.9)


def test_build_context_block_with_symbol():
    result = _make_result(symbol_name="validate_token", symbol_type="function")
    block = build_context_block(result)
    assert "function: validate_token" in block
    assert "def validate_token(token): pass" in block
    assert "src/auth/jwt.py" in block
    assert "lines 15-45" in block


def test_build_context_block_without_symbol():
    result = _make_result(symbol_name=None, symbol_type=None)
    block = build_context_block(result)
    assert "function:" not in block
    assert "class:" not in block
    assert "src/auth/jwt.py" in block
