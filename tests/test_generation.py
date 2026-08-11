from __future__ import annotations

from collections.abc import Iterator

import pytest

from gritter.generation.generator import Generator
from gritter.generation.prompts import build_context_block, build_user_prompt
from gritter.models.chunk import CodeChunk
from gritter.models.query import RetrievalResult
from gritter.providers.llm import LLMProvider, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


class MockLLMProvider(LLMProvider):
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        yield from self._tokens


# ---------------------------------------------------------------------------
# prompts.py tests
# ---------------------------------------------------------------------------

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


def test_build_user_prompt_contains_all_chunks():
    results = [
        _make_result(file_path="src/auth/jwt.py", content="def validate_token(): pass"),
        _make_result(file_path="src/models/user.py", content="class User: pass"),
    ]
    query = "How does authentication work?"
    prompt = build_user_prompt(query, results)
    assert "src/auth/jwt.py" in prompt
    assert "src/models/user.py" in prompt
    assert "def validate_token(): pass" in prompt
    assert "class User: pass" in prompt
    assert "Question: How does authentication work?" in prompt
    assert prompt.startswith("Context:")


# ---------------------------------------------------------------------------
# generator.py tests
# ---------------------------------------------------------------------------

def test_generator_streams_tokens():
    mock_llm = MockLLMProvider(["Hello", " world"])
    gen = Generator(llm=mock_llm)
    results = [_make_result()]
    collected = list(gen.stream("What does validate_token do?", results))
    assert collected == ["Hello", " world"]
    assert len(gen.messages) == 2
    assert gen.messages[0].role == "user"
    assert gen.messages[1].role == "assistant"
    assert gen.messages[1].content == "Hello world"


def test_generator_reset_clears_history():
    mock_llm = MockLLMProvider(["Hello", " world"])
    gen = Generator(llm=mock_llm)
    results = [_make_result()]
    list(gen.stream("What does validate_token do?", results))
    assert len(gen.messages) == 2
    gen.reset()
    assert gen.messages == []
