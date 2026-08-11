# Agentic Gritter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace gritter's fixed retrieve-then-generate pipeline with an LLM-driven tool-calling loop (search_code, read_file, grep, list_symbols, reindex), and verify citations against real files instead of trusting model output blindly.

**Architecture:** A new `gritter/agent/` package holds the tool implementations (`tools.py`) and the loop orchestrator (`AgentSession` in `session.py`). The `LLMProvider` ABC is changed from a single `stream()` method to `run_tools()`, which all three providers (Claude, OpenAI, Ollama) implement against their native tool-calling APIs. `ask`/`chat` CLI commands are rewired to drive `AgentSession` instead of `HybridRetriever.search` + `Generator.stream`, which are removed.

**Tech Stack:** Python 3.11+, `anthropic` SDK (native `tool_use`), `openai` SDK (native function calling, shared by `OpenAIProvider` and `OllamaProvider`), `typer`, `rich`, existing `pytest`/`pytest-mock` test stack. Optional system `rg` (ripgrep) binary for `grep`, with a pure-Python fallback.

## Global Constraints

- Python `>=3.11` (per `pyproject.toml`) — `Path.is_relative_to` is safe to use.
- No new PyPI dependencies — ripgrep is invoked as a subprocess if present, never required.
- `ask`/`chat` are replaced outright; no dual old/fixed-pipeline mode is kept (per spec).
- Agent mode requires native tool-calling from the configured LLM provider/model — no text-based tool-call parsing fallback (per spec).
- Loop cap is 8 tool calls per turn (per spec); on cap, force a text-only final answer by omitting `tools` from the next `run_tools` call rather than raising.
- Citations are verified post-hoc against the real repo tree; unverified ones are flagged inline (`[unverified]`), never silently dropped (per spec).
- Final release version is `1.0.0`, tagged `v1.0.0` — note a stray local `v1.0.0` tag already exists pointing at an old, unrelated commit (`ca0e872`) and must be moved/replaced, not reused as-is.

---

## Task 1: Persist the indexed repo's source root in `IndexMeta`

Agent tools (`read_file`, `grep`, `list_symbols`, `reindex`) need to know where the indexed repo actually lives on disk. Today `run_indexing_pipeline` receives `root` but never persists it — `ask`/`chat` have no way to recover it. Add a `source_root` field to `IndexMeta`.

**Files:**
- Modify: `gritter/storage/index_meta.py`
- Modify: `gritter/indexing/pipeline.py`
- Test: `tests/test_incremental.py` (existing file already exercises `IndexMeta`/pipeline; add assertions there)

**Interfaces:**
- Produces: `IndexMeta.write(..., source_root: str, ...)` — now a required keyword argument. `IndexMeta.read()` returns a dict that includes `"source_root"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_incremental.py` (check existing imports at the top of the file first; reuse whatever `tmp_path`/config fixtures it already has):

```python
def test_index_meta_persists_source_root(tmp_path):
    from gritter.storage.index_meta import IndexMeta

    index_dir = tmp_path / "index"
    meta = IndexMeta(index_dir)
    meta.write(
        embedding_provider="voyage",
        embedding_model="voyage-code-3",
        embedding_dimension=1024,
        file_count=1,
        chunk_count=1,
        languages=["python"],
        source_root="/repos/example",
    )
    reloaded = IndexMeta(index_dir).read()
    assert reloaded["source_root"] == "/repos/example"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_incremental.py::test_index_meta_persists_source_root -v`
Expected: FAIL with `TypeError: write() missing 1 required keyword-only argument: 'source_root'`

- [ ] **Step 3: Update `IndexMeta.write` to accept and store `source_root`**

In `gritter/storage/index_meta.py`, change the `write` signature and body:

```python
    def write(
        self,
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        file_count: int,
        chunk_count: int,
        languages: list[str],
        source_root: str,
        indexed_commit: str | None = None,
    ) -> None:
        self._data = {
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "file_count": file_count,
            "chunk_count": chunk_count,
            "languages": sorted(languages),
            "source_root": source_root,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "indexed_commit": indexed_commit,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))
```

- [ ] **Step 4: Pass `source_root` at every `meta.write(...)` call site in `gritter/indexing/pipeline.py`**

There are three call sites: one in `_run_full`, two in `_run_incremental` (the early-return "nothing changed" branch and the main branch). All three already have `root: Path` in scope. Add `source_root=str(root)` to each:

In `_run_full`, the existing call becomes:

```python
        meta.write(
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model or _default_model(config.embedding.provider),
            embedding_dimension=embed_provider.dimension,
            file_count=len(file_pairs),
            chunk_count=len(all_chunks),
            languages=list(languages_seen),
            source_root=str(root),
            indexed_commit=current_commit,
        )
```

In `_run_incremental`'s early-return branch:

```python
        meta.write(
            embedding_provider=stored["embedding_provider"],
            embedding_model=stored["embedding_model"],
            embedding_dimension=stored["embedding_dimension"],
            file_count=stored["file_count"],
            chunk_count=stored["chunk_count"],
            languages=stored["languages"],
            source_root=stored.get("source_root", str(root)),
            indexed_commit=current_commit,
        )
```

(Using `stored.get(..., str(root))` guards against re-indexing an index built before this change, which won't have `source_root` in its stored metadata yet.)

In `_run_incremental`'s main branch, at the final `meta.write(...)`:

```python
        meta.write(
            embedding_provider=stored_meta["embedding_provider"],
            embedding_model=stored_meta["embedding_model"],
            embedding_dimension=stored_meta["embedding_dimension"],
            file_count=vector_store.count(),
            chunk_count=len(all_current_chunks),
            languages=all_languages,
            source_root=stored_meta.get("source_root", str(root)),
            indexed_commit=current_commit,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_incremental.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 6: Run the full existing suite to check nothing else broke**

Run: `pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: PASS. (Any test that calls `IndexMeta.write` directly without `source_root` will now fail — fix those call sites the same way, adding `source_root="/tmp/whatever"` or the fixture's actual root.)

- [ ] **Step 7: Commit**

```bash
git add gritter/storage/index_meta.py gritter/indexing/pipeline.py tests/test_incremental.py
git commit -m "feat: persist indexed repo's source_root in IndexMeta"
```

---

## Task 2: Replace `LLMProvider.stream` with `run_tools` across all three providers

This is the core provider-interface change. Add the shared event/message/tool-spec types, then implement `run_tools` for Claude, OpenAI, and Ollama.

**Files:**
- Modify: `gritter/providers/llm.py` (full rewrite of the file)
- Test: `tests/test_llm_provider.py` (new)

**Interfaces:**
- Produces:
  - `@dataclass ToolSpec(name: str, description: str, parameters: dict)`
  - `@dataclass ToolCall(id: str, name: str, arguments: dict)`
  - `@dataclass TextDelta(text: str)`
  - `@dataclass Done()`
  - `AgentEvent = TextDelta | ToolCall | Done`
  - `@dataclass Message(role: str, content: str = "", tool_calls: list[ToolCall] = [], tool_call_id: str | None = None)`
  - `LLMProvider.run_tools(system: str, messages: list[Message], tools: list[ToolSpec]) -> Iterator[AgentEvent]` (abstract)
  - `get_llm_provider(provider, model=None, base_url=None) -> LLMProvider` (unchanged signature)

Only the parts of `run_tools` that are provider-agnostic (message conversion, event-shape) are unit-testable without hitting a real API. The three provider classes' `run_tools` bodies are integration-tested by the existing `test_e2e.py` (already gated behind `GRITTER_RUN_E2E_TESTS=1`), updated in Task 6. Here we unit-test the pure conversion helpers.

- [ ] **Step 1: Write the failing tests for the message-conversion helpers**

Create `tests/test_llm_provider.py`:

```python
from __future__ import annotations

from gritter.providers.llm import Message, ToolCall, ToolSpec
from gritter.providers.llm import _messages_to_anthropic, _messages_to_openai


def test_messages_to_anthropic_plain_user_and_assistant():
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]
    converted = _messages_to_anthropic(messages)
    assert converted == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_messages_to_anthropic_assistant_with_tool_call():
    messages = [
        Message(
            role="assistant",
            content="checking...",
            tool_calls=[ToolCall(id="tc1", name="search_code", arguments={"query": "auth"})],
        ),
        Message(role="tool", content="found stuff", tool_call_id="tc1"),
    ]
    converted = _messages_to_anthropic(messages)
    assert converted[0]["role"] == "assistant"
    blocks = converted[0]["content"]
    assert {"type": "text", "text": "checking..."} in blocks
    assert {"type": "tool_use", "id": "tc1", "name": "search_code", "input": {"query": "auth"}} in blocks
    assert converted[1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tc1", "content": "found stuff"}],
    }


def test_messages_to_openai_assistant_with_tool_call():
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="tc1", name="search_code", arguments={"query": "auth"})],
        ),
        Message(role="tool", content="found stuff", tool_call_id="tc1"),
    ]
    converted = _messages_to_openai(messages)
    assert converted[0]["role"] == "assistant"
    assert converted[0]["content"] is None
    assert converted[0]["tool_calls"] == [
        {
            "id": "tc1",
            "type": "function",
            "function": {"name": "search_code", "arguments": '{"query": "auth"}'},
        }
    ]
    assert converted[1] == {"role": "tool", "tool_call_id": "tc1", "content": "found stuff"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_provider.py -v`
Expected: FAIL with `ImportError: cannot import name '_messages_to_anthropic'`

- [ ] **Step 3: Rewrite `gritter/providers/llm.py`**

Replace the entire file:

```python
from __future__ import annotations
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "user", "assistant", or "tool"
    content: str = ""
    tool_calls: list["ToolCall"] = field(default_factory=list)  # only on assistant messages
    tool_call_id: str | None = None  # only on role="tool" messages


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class TextDelta:
    text: str


@dataclass
class Done:
    pass


AgentEvent = TextDelta | ToolCall | Done


class LLMProvider(ABC):
    @abstractmethod
    def run_tools(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> Iterator[AgentEvent]:
        """Stream text and/or emit tool calls until the model is done.

        Emits zero or more TextDelta events, then zero or more ToolCall
        events (one per requested tool call), then exactly one Done event.
        If `tools` is empty, the model must respond with text only.
        """


# ---------------------------------------------------------------------------
# Message conversion helpers (provider-agnostic -> provider wire format)
# ---------------------------------------------------------------------------

def _messages_to_anthropic(messages: list[Message]) -> list[dict]:
    converted: list[dict] = []
    for m in messages:
        if m.role == "tool":
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }],
            })
        elif m.role == "assistant" and m.tool_calls:
            blocks: list[dict] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append({
                    "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments,
                })
            converted.append({"role": "assistant", "content": blocks})
        else:
            converted.append({"role": m.role, "content": m.content})
    return converted


def _messages_to_openai(messages: list[Message]) -> list[dict]:
    converted: list[dict] = []
    for m in messages:
        if m.role == "tool":
            converted.append({
                "role": "tool", "tool_call_id": m.tool_call_id, "content": m.content,
            })
        elif m.role == "assistant" and m.tool_calls:
            converted.append({
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ],
            })
        else:
            converted.append({"role": m.role, "content": m.content})
    return converted


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class ClaudeProvider(LLMProvider):
    """Anthropic Claude — default provider."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is required for Claude."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def run_tools(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> Iterator[AgentEvent]:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "system": system,
            "messages": _messages_to_anthropic(messages),
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield TextDelta(text)
            final = stream.get_final_message()

        for block in final.content:
            if block.type == "tool_use":
                yield ToolCall(id=block.id, name=block.name, arguments=block.input)
        yield Done()


class OpenAIProvider(LLMProvider):
    """OpenAI GPT models."""

    def __init__(self, model: str = "gpt-4o") -> None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for OpenAI."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def run_tools(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> Iterator[AgentEvent]:
        yield from _run_openai_style_tools(self._client, self._model, system, messages, tools)


class OllamaProvider(LLMProvider):
    """Local Ollama — no API key required. Requires a tool-calling-capable model
    (e.g. llama3.1+); older/small models without tool-calling support will not
    work in agent mode."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434") -> None:
        from openai import OpenAI
        self._client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        self._model = model

    def run_tools(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> Iterator[AgentEvent]:
        yield from _run_openai_style_tools(self._client, self._model, system, messages, tools)


def _run_openai_style_tools(
    client, model: str, system: str, messages: list[Message], tools: list[ToolSpec]
) -> Iterator[AgentEvent]:
    """Shared streaming + tool-call-accumulation logic for OpenAI and Ollama
    (both speak the OpenAI-compatible chat completions API)."""
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + _messages_to_openai(messages),
        "stream": True,
    }
    if tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]

    response = client.chat.completions.create(**kwargs)
    pending_calls: dict[int, dict] = {}

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            yield TextDelta(delta.content)
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                entry = pending_calls.setdefault(tc_delta.index, {"id": None, "name": "", "arguments": ""})
                if tc_delta.id:
                    entry["id"] = tc_delta.id
                if tc_delta.function and tc_delta.function.name:
                    entry["name"] += tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    entry["arguments"] += tc_delta.function.arguments

    for entry in pending_calls.values():
        yield ToolCall(
            id=entry["id"], name=entry["name"], arguments=json.loads(entry["arguments"] or "{}"),
        )
    yield Done()


def get_llm_provider(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    if provider == "claude":
        return ClaudeProvider(model or "claude-sonnet-4-6")
    elif provider == "openai":
        return OpenAIProvider(model or "gpt-4o")
    elif provider == "ollama":
        return OllamaProvider(model or "llama3.1", base_url or "http://localhost:11434")
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_provider.py -v`
Expected: PASS

- [ ] **Step 5: Fix the now-broken `MockLLMProvider` in `tests/test_generation.py`**

`MockLLMProvider` in `tests/test_generation.py` still implements the old abstract `stream()` method, which no longer exists on the ABC — it will fail to instantiate as a subclass missing `run_tools`. Leave `tests/test_generation.py` as-is for now; it gets rewritten in Task 6 when `Generator` is deleted. Confirm the breakage is understood by running:

Run: `pytest tests/test_generation.py -v`
Expected: FAIL (collection error: `TypeError: Can't instantiate abstract class MockLLMProvider`) — this is expected and will be fixed in Task 6, not this task.

- [ ] **Step 6: Commit**

```bash
git add gritter/providers/llm.py tests/test_llm_provider.py
git commit -m "feat: replace LLMProvider.stream with tool-calling run_tools"
```

---

## Task 3: Build the agent tools module

**Files:**
- Create: `gritter/agent/__init__.py`
- Create: `gritter/agent/tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: `HybridRetriever.search(query: str) -> list[RetrievalResult]` (`gritter/retrieval/hybrid.py`), `build_context_block(result: RetrievalResult) -> str` (`gritter/generation/prompts.py`), `chunk_file(content, file_path, language, min_tokens, max_tokens, overlap_tokens) -> list[CodeChunk]` (`gritter/indexing/chunker.py`), `detect_language(path: str) -> str | None` (`gritter/utils/languages.py`), `run_indexing_pipeline(root, index_name, config, embed_provider, full=False) -> dict` (`gritter/indexing/pipeline.py`), `get_embedding_provider(provider, model) -> EmbeddingProvider` (`gritter/providers/embeddings.py`), `GritterConfig` (`gritter/models/config.py`).
- Produces:
  - `@dataclass ToolContext(repo_root: Path, retriever: HybridRetriever, config: GritterConfig, index_name: str)`
  - `TOOL_SPECS: list[ToolSpec]` (5 entries: `search_code`, `read_file`, `grep`, `list_symbols`, `reindex`)
  - `dispatch_tool(ctx: ToolContext, name: str, arguments: dict) -> str` — never raises; returns an `"Error: ..."` string on failure.

- [ ] **Step 1: Create the package init**

```python
# gritter/agent/__init__.py
```

(empty file)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_agent_tools.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_agent_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gritter.agent.tools'`

- [ ] **Step 4: Implement `gritter/agent/tools.py`**

```python
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from gritter.generation.prompts import build_context_block
from gritter.models.config import GritterConfig
from gritter.providers.llm import ToolSpec
from gritter.retrieval.hybrid import HybridRetriever


@dataclass
class ToolContext:
    repo_root: Path
    retriever: HybridRetriever
    config: GritterConfig
    index_name: str


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="search_code",
        description=(
            "Hybrid dense+sparse semantic search over the indexed codebase. "
            "Returns the top matching code chunks with file paths and line ranges."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language or code search query."},
                "top_k": {"type": "integer", "description": "Max number of results to return."},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="read_file",
        description="Read a file from the indexed repository, optionally restricted to a line range.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative file path."},
                "start_line": {"type": "integer", "description": "First line to include (1-indexed)."},
                "end_line": {"type": "integer", "description": "Last line to include (1-indexed)."},
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="grep",
        description="Search file contents for a regex pattern across the repository.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for."},
                "path_glob": {"type": "string", "description": "Optional glob to restrict which files are searched."},
            },
            "required": ["pattern"],
        },
    ),
    ToolSpec(
        name="list_symbols",
        description="List top-level functions/classes/methods and their line ranges in a single file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Repo-relative file path."}},
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="reindex",
        description=(
            "Re-run incremental indexing if search results seem stale relative to file contents. "
            "Slow — only call this when you suspect the index is out of date."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
    ),
]


def dispatch_tool(ctx: ToolContext, name: str, arguments: dict) -> str:
    handlers = {
        "search_code": tool_search_code,
        "read_file": tool_read_file,
        "grep": tool_grep,
        "list_symbols": tool_list_symbols,
        "reindex": tool_reindex,
    }
    handler = handlers.get(name)
    if handler is None:
        return f"Error: unknown tool {name!r}"
    try:
        return handler(ctx, **arguments)
    except Exception as exc:
        return f"Error: {exc}"


def tool_search_code(ctx: ToolContext, query: str, top_k: int | None = None) -> str:
    results = ctx.retriever.search(query)
    if top_k is not None:
        results = results[:top_k]
    if not results:
        return "No results found."
    return "\n\n".join(build_context_block(r) for r in results)


def _resolve_path(repo_root: Path, path: str) -> Path:
    repo_root_resolved = repo_root.resolve()
    candidate = (repo_root_resolved / path).resolve()
    if not candidate.is_relative_to(repo_root_resolved):
        raise ValueError(f"Path {path!r} is outside the indexed repository.")
    if not candidate.is_file():
        raise ValueError(f"{path!r} is not a file.")
    return candidate


def tool_read_file(
    ctx: ToolContext, path: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    target = _resolve_path(ctx.repo_root, path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start_line or 1)
    end = min(len(lines), end_line or len(lines))
    numbered = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    return f"--- {path} (lines {start}-{end}) ---\n{numbered}"


def tool_grep(ctx: ToolContext, pattern: str, path_glob: str | None = None) -> str:
    if shutil.which("rg"):
        return _grep_ripgrep(ctx.repo_root, pattern, path_glob)
    return _grep_python(ctx.repo_root, pattern, path_glob)


def _grep_ripgrep(repo_root: Path, pattern: str, path_glob: str | None) -> str:
    args = ["rg", "--line-number", "--no-heading", "--max-count", "5", pattern]
    if path_glob:
        args += ["--glob", path_glob]
    result = subprocess.run(args, cwd=repo_root, capture_output=True, text=True, timeout=10)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"rg failed: {result.stderr.strip()}")
    return result.stdout.strip() or "No matches found."


def _grep_python(repo_root: Path, pattern: str, path_glob: str | None) -> str:
    regex = re.compile(pattern)
    matches: list[str] = []
    for file_path in repo_root.glob(path_glob or "**/*"):
        if len(matches) >= 50:
            break
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{file_path.relative_to(repo_root)}:{i}:{line}")
                if len(matches) >= 50:
                    break
    return "\n".join(matches) if matches else "No matches found."


def tool_list_symbols(ctx: ToolContext, path: str) -> str:
    from gritter.indexing.chunker import chunk_file
    from gritter.utils.languages import detect_language

    target = _resolve_path(ctx.repo_root, path)
    language = detect_language(path)
    if language is None:
        return f"No language support for {path!r}; cannot extract symbols."

    content = target.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_file(
        content, path, language,
        min_tokens=ctx.config.index.chunk_min_tokens,
        max_tokens=ctx.config.index.chunk_max_tokens,
        overlap_tokens=ctx.config.index.chunk_overlap_tokens,
    )
    lines = [
        f"{c.symbol_type or 'module'}: {c.symbol_name or '(anonymous)'} (lines {c.start_line}-{c.end_line})"
        for c in chunks
    ]
    return "\n".join(lines) if lines else "No symbols found."


def tool_reindex(ctx: ToolContext) -> str:
    from gritter.indexing.pipeline import run_indexing_pipeline
    from gritter.providers.embeddings import get_embedding_provider

    embed_provider = get_embedding_provider(ctx.config.embedding.provider, ctx.config.embedding.model)
    summary = run_indexing_pipeline(ctx.repo_root, ctx.index_name, ctx.config, embed_provider)
    return f"Reindexed: {json.dumps(summary)}"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_agent_tools.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gritter/agent/__init__.py gritter/agent/tools.py tests/test_agent_tools.py
git commit -m "feat: add agent tools (search_code, read_file, grep, list_symbols, reindex)"
```

---

## Task 4: Citation verification

Close the citation-trust gap: check every `path:Lstart-end` citation against the real file on disk instead of trusting it blindly. Move citation tests out of `tests/test_generation.py` into their own file since `test_generation.py` is getting gutted in Task 6.

**Files:**
- Modify: `gritter/generation/citations.py`
- Create: `tests/test_citations.py`
- Modify: `tests/test_generation.py` (remove the citation test block — it's moving)

**Interfaces:**
- Produces:
  - `extract_citations(text: str) -> list[str]` (unchanged)
  - `verify_citations(citations: list[str], repo_root: Path) -> list[tuple[str, bool]]` (new)
  - `format_sources(verified: list[tuple[str, bool]]) -> str` (**signature changed** — previously took `list[str]`; now takes the verified pairs `verify_citations` produces, and appends `" [unverified]"` to any citation whose bool is `False`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_citations.py`:

```python
from __future__ import annotations

from pathlib import Path

from gritter.generation.citations import extract_citations, format_sources, verify_citations


def test_extract_citations_basic():
    text = "See src/auth/jwt.py:L15-45 for the implementation."
    assert extract_citations(text) == ["src/auth/jwt.py:L15-45"]


def test_extract_citations_deduplication():
    text = "src/auth/jwt.py:L15-45 is mentioned and again src/auth/jwt.py:L15-45 here."
    assert extract_citations(text) == ["src/auth/jwt.py:L15-45"]


def test_extract_citations_multiple():
    text = "First see src/auth/jwt.py:L15-45, then check src/models/user.py:L10."
    assert extract_citations(text) == ["src/auth/jwt.py:L15-45", "src/models/user.py:L10"]


def test_verify_citations_real_file_in_range(tmp_path: Path):
    (tmp_path / "foo.py").write_text("\n".join(f"line{i}" for i in range(1, 21)) + "\n")
    result = verify_citations(["foo.py:L1-20"], tmp_path)
    assert result == [("foo.py:L1-20", True)]


def test_verify_citations_out_of_range(tmp_path: Path):
    (tmp_path / "foo.py").write_text("line1\nline2\n")
    result = verify_citations(["foo.py:L1-20"], tmp_path)
    assert result == [("foo.py:L1-20", False)]


def test_verify_citations_missing_file(tmp_path: Path):
    result = verify_citations(["nope.py:L1-5"], tmp_path)
    assert result == [("nope.py:L1-5", False)]


def test_verify_citations_path_traversal_rejected(tmp_path: Path):
    outside = tmp_path.parent / "secret.py"
    outside.write_text("x\n" * 10)
    result = verify_citations(["../secret.py:L1-5"], tmp_path)
    assert result == [("../secret.py:L1-5", False)]


def test_verify_citations_single_line_form(tmp_path: Path):
    (tmp_path / "foo.py").write_text("only one line\n")
    result = verify_citations(["foo.py:L1"], tmp_path)
    assert result == [("foo.py:L1", True)]


def test_format_sources_marks_unverified():
    verified = [("src/auth/jwt.py:L15-45", True), ("src/fake.py:L1-5", False)]
    output = format_sources(verified)
    assert output.startswith("Sources:")
    assert "• src/auth/jwt.py:L15-45\n" in output + "\n"
    assert "src/fake.py:L1-5 [unverified]" in output


def test_format_sources_empty():
    assert format_sources([]) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_citations.py -v`
Expected: FAIL — `verify_citations` doesn't exist yet, and `format_sources` still takes plain strings so the `[unverified]` assertions fail.

- [ ] **Step 3: Update `gritter/generation/citations.py`**

Replace the entire file:

```python
from __future__ import annotations

import re
from pathlib import Path

# Matches a citation of the form:
#   - a path containing at least one "/" (e.g. src/auth/jwt.py), OR
#   - a path ending in a known extension (.py, .ts, .tsx, .js, .rs)
# followed by :L<digits> and an optional -<digits> range.
_CITATION_RE = re.compile(
    r"(?:[\w.\-]+/[\w./\-]+|[\w./\-]+\.(?:py|ts|tsx|js|rs))"
    r":L\d+(?:-\d+)?"
)

_CITATION_PARSE_RE = re.compile(r"^(?P<path>.+):L(?P<start>\d+)(?:-(?P<end>\d+))?$")


def extract_citations(text: str) -> list[str]:
    """Return deduplicated citations from text, preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for match in _CITATION_RE.finditer(text):
        citation = match.group(0)
        if citation not in seen:
            seen.add(citation)
            result.append(citation)
    return result


def verify_citations(citations: list[str], repo_root: Path) -> list[tuple[str, bool]]:
    """Check each citation's path/line-range against the real file on disk."""
    return [(citation, _verify_one(citation, repo_root)) for citation in citations]


def _verify_one(citation: str, repo_root: Path) -> bool:
    match = _CITATION_PARSE_RE.match(citation)
    if not match:
        return False

    rel_path = match.group("path")
    start = int(match.group("start"))
    end = int(match.group("end")) if match.group("end") else start

    repo_root_resolved = repo_root.resolve()
    target = (repo_root_resolved / rel_path).resolve()
    if not target.is_relative_to(repo_root_resolved):
        return False
    if not target.is_file():
        return False

    try:
        line_count = sum(1 for _ in target.open(encoding="utf-8", errors="replace"))
    except OSError:
        return False

    return 1 <= start <= end <= line_count


def format_sources(verified: list[tuple[str, bool]]) -> str:
    """Format verified (citation, is_verified) pairs as a Sources block, or '' if empty."""
    if not verified:
        return ""
    lines = ["Sources:"]
    for citation, ok in verified:
        suffix = "" if ok else " [unverified]"
        lines.append(f"  • {citation}{suffix}")
    return "\n".join(lines)
```

- [ ] **Step 4: Remove the citation tests from `tests/test_generation.py`**

Delete the block between the `# citations.py tests` comment and the `# generator.py tests` comment (the four `test_extract_citations_*`/`test_format_sources_*` functions and the now-unused `format_sources`/`extract_citations` import) — they now live in `tests/test_citations.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_citations.py -v`
Expected: PASS

Run: `pytest tests/test_generation.py -v`
Expected: Still FAILS at collection (`MockLLMProvider` abstract-method error from Task 2, Step 5) — unrelated to this task, fixed in Task 6.

- [ ] **Step 6: Commit**

```bash
git add gritter/generation/citations.py tests/test_citations.py tests/test_generation.py
git commit -m "feat: verify citations against real files instead of trusting model output"
```

---

## Task 5: Build `AgentSession` and the agent system prompt

**Files:**
- Create: `gritter/agent/prompts.py`
- Create: `gritter/agent/session.py`
- Test: `tests/test_agent_session.py`

**Interfaces:**
- Consumes: `AgentEvent` / `TextDelta` / `ToolCall` / `Done` / `Message` / `LLMProvider` / `ToolSpec` (`gritter/providers/llm.py`, Task 2), `ToolContext` / `TOOL_SPECS` / `dispatch_tool` (`gritter/agent/tools.py`, Task 3), `extract_citations` / `verify_citations` (`gritter/generation/citations.py`, Task 4).
- Produces:
  - `AGENT_SYSTEM_PROMPT: str`
  - `@dataclass ToolStarted(name: str, arguments: dict)`
  - `@dataclass ToolFinished(name: str, result: str)`
  - `SessionEvent = TextDelta | ToolStarted | ToolFinished`
  - `class AgentSession(llm: LLMProvider, ctx: ToolContext, max_tool_calls: int = 8)` with:
    - `.messages: list[Message]`
    - `.last_citations: list[tuple[str, bool]]`
    - `.run(query: str) -> Iterator[SessionEvent]`

- [ ] **Step 1: Write `gritter/agent/prompts.py`**

```python
from __future__ import annotations

AGENT_SYSTEM_PROMPT: str = """\
You are a code assistant with tools to investigate a codebase before answering.

Available tools:
- search_code: semantic search over the indexed codebase.
- read_file: read a specific file, optionally a line range.
- grep: regex search over file contents.
- list_symbols: list functions/classes/methods in a single file.
- reindex: re-run indexing if you suspect the index is stale (slow — use sparingly).

Rules:
1. Call exactly one tool per turn. Use search_code first for open-ended questions; use \
read_file, grep, or list_symbols to verify or dig deeper into specific files once you have \
candidates.
2. Answer only using information you have actually retrieved via tools in this conversation. \
Do not use prior knowledge or assumptions about this codebase.
3. Cite every claim using the exact format: path/to/file.py:L15-45 (using the start and end \
lines you actually observed). If a claim comes from a single line, use path/to/file.py:L15.
4. If, after investigating, the answer is not present in what you found, respond with exactly: \
"I don't know from the provided context." Do not guess or hallucinate file paths or line numbers.
5. Once you have enough information, respond with your final answer as plain text instead of \
calling another tool.\
"""
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_agent_session.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gritter.agent.session import AgentSession, ToolFinished, ToolStarted
from gritter.agent.tools import ToolContext
from gritter.providers.llm import AgentEvent, Done, LLMProvider, Message, TextDelta, ToolCall, ToolSpec


class ScriptedLLMProvider(LLMProvider):
    """Replays a fixed sequence of turns, one list[AgentEvent] per call to run_tools."""

    def __init__(self, turns: list[list[AgentEvent]]) -> None:
        self._turns = list(turns)
        self.calls: list[tuple[str, list[Message], list[ToolSpec]]] = []

    def run_tools(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> Iterator[AgentEvent]:
        self.calls.append((system, list(messages), list(tools)))
        turn = self._turns.pop(0)
        yield from turn


def _make_ctx(tmp_path: Path) -> ToolContext:
    from gritter.models.config import GritterConfig
    retriever = MagicMock()
    retriever.search.return_value = []
    return ToolContext(
        repo_root=tmp_path,
        retriever=retriever,
        config=GritterConfig(data_dir=tmp_path / "data"),
        index_name="test_index",
    )


def test_session_answers_without_any_tool_call(tmp_path):
    llm = ScriptedLLMProvider([[TextDelta("Hello "), TextDelta("world"), Done()]])
    session = AgentSession(llm, _make_ctx(tmp_path))
    events = list(session.run("hi"))
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "Hello world"
    assert len(llm.calls) == 1
    assert session.messages[-1].role == "assistant"
    assert session.messages[-1].content == "Hello world"


def test_session_dispatches_one_tool_call_then_answers(tmp_path):
    (tmp_path / "foo.py").write_text("line1\n")
    llm = ScriptedLLMProvider([
        [ToolCall(id="tc1", name="read_file", arguments={"path": "foo.py"}), Done()],
        [TextDelta("It says line1."), Done()],
    ])
    session = AgentSession(llm, _make_ctx(tmp_path))
    events = list(session.run("what's in foo.py?"))

    started = [e for e in events if isinstance(e, ToolStarted)]
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert len(started) == 1 and started[0].name == "read_file"
    assert len(finished) == 1 and "line1" in finished[0].result
    assert len(llm.calls) == 2

    # tool-result message got appended to history before the second call
    tool_msgs = [m for m in session.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "tc1"


def test_session_stops_at_max_tool_calls(tmp_path):
    def always_call_tool():
        return [ToolCall(id="tcN", name="search_code", arguments={"query": "x"}), Done()]

    turns = [always_call_tool() for _ in range(2)] + [[TextDelta("final answer"), Done()]]
    llm = ScriptedLLMProvider(turns)
    session = AgentSession(llm, _make_ctx(tmp_path), max_tool_calls=2)
    events = list(session.run("query"))

    # third call must be made with an empty tool list, forcing a text-only answer
    assert llm.calls[2][2] == []
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "final answer"


def test_session_verifies_citations_after_answering(tmp_path):
    (tmp_path / "foo.py").write_text("\n".join(f"l{i}" for i in range(1, 6)) + "\n")
    llm = ScriptedLLMProvider([[TextDelta("See foo.py:L1-5 and fake.py:L1-5."), Done()]])
    session = AgentSession(llm, _make_ctx(tmp_path))
    list(session.run("q"))
    assert ("foo.py:L1-5", True) in session.last_citations
    assert ("fake.py:L1-5", False) in session.last_citations


def test_session_preserves_history_across_multiple_run_calls(tmp_path):
    llm = ScriptedLLMProvider([
        [TextDelta("first answer"), Done()],
        [TextDelta("second answer"), Done()],
    ])
    session = AgentSession(llm, _make_ctx(tmp_path))
    list(session.run("q1"))
    list(session.run("q2"))
    roles_and_content = [(m.role, m.content) for m in session.messages]
    assert roles_and_content == [
        ("user", "q1"),
        ("assistant", "first answer"),
        ("user", "q2"),
        ("assistant", "second answer"),
    ]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_agent_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gritter.agent.session'`

- [ ] **Step 4: Implement `gritter/agent/session.py`**

```python
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from gritter.agent.prompts import AGENT_SYSTEM_PROMPT
from gritter.agent.tools import TOOL_SPECS, ToolContext, dispatch_tool
from gritter.generation.citations import extract_citations, verify_citations
from gritter.providers.llm import Done, LLMProvider, Message, TextDelta, ToolCall


@dataclass
class ToolStarted:
    name: str
    arguments: dict


@dataclass
class ToolFinished:
    name: str
    result: str


SessionEvent = TextDelta | ToolStarted | ToolFinished


class AgentSession:
    def __init__(self, llm: LLMProvider, ctx: ToolContext, max_tool_calls: int = 8) -> None:
        self._llm = llm
        self._ctx = ctx
        self._max_tool_calls = max_tool_calls
        self.messages: list[Message] = []
        self.last_citations: list[tuple[str, bool]] = []

    def run(self, query: str) -> Iterator[SessionEvent]:
        self.messages.append(Message(role="user", content=query))
        calls_made = 0
        final_text = ""

        while True:
            tools = TOOL_SPECS if calls_made < self._max_tool_calls else []
            text_chunks: list[str] = []
            requested_call: ToolCall | None = None

            for event in self._llm.run_tools(AGENT_SYSTEM_PROMPT, self.messages, tools):
                if isinstance(event, TextDelta):
                    text_chunks.append(event.text)
                    yield event
                elif isinstance(event, ToolCall) and requested_call is None:
                    requested_call = event
                elif isinstance(event, Done):
                    break

            turn_text = "".join(text_chunks)

            if requested_call is None:
                self.messages.append(Message(role="assistant", content=turn_text))
                final_text = turn_text
                break

            self.messages.append(Message(
                role="assistant", content=turn_text, tool_calls=[requested_call],
            ))
            yield ToolStarted(name=requested_call.name, arguments=requested_call.arguments)
            result = dispatch_tool(self._ctx, requested_call.name, requested_call.arguments)
            yield ToolFinished(name=requested_call.name, result=result)
            self.messages.append(Message(role="tool", content=result, tool_call_id=requested_call.id))
            calls_made += 1

        citations = extract_citations(final_text)
        self.last_citations = verify_citations(citations, self._ctx.repo_root)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_agent_session.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gritter/agent/prompts.py gritter/agent/session.py tests/test_agent_session.py
git commit -m "feat: add AgentSession tool-calling loop"
```

---

## Task 6: Rewire `ask`/`chat` CLI commands onto `AgentSession`, delete `Generator`

**Files:**
- Modify: `gritter/cli/ask_cmd.py`
- Modify: `gritter/cli/chat_cmd.py`
- Delete: `gritter/generation/generator.py`
- Modify: `gritter/generation/prompts.py` (remove now-unused `SYSTEM_PROMPT` and `build_user_prompt`, keep `build_context_block`)
- Modify: `tests/test_generation.py` (remove `Generator` tests; file now only covers `build_context_block`)
- Modify: `tests/test_e2e.py` (swap `Generator` usage for `AgentSession`)

**Interfaces:**
- Consumes: `AgentSession`, `ToolStarted`, `ToolFinished` (`gritter/agent/session.py`, Task 5), `ToolContext` (`gritter/agent/tools.py`, Task 3), `TextDelta` (`gritter/providers/llm.py`, Task 2), `format_sources` (`gritter/generation/citations.py`, Task 4), `IndexMeta` (`gritter/storage/index_meta.py`, Task 1).

- [ ] **Step 1: Write the failing test for `build_context_block`-only `test_generation.py`**

Trim `tests/test_generation.py` down to just the prompts tests (remove `Generator`/`MockLLMProvider`/the generator test block and their now-dangling imports):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generation.py -v`
Expected: PASS already (this file only imports things that still exist) — this step confirms the trim didn't break anything on its own; the real failure comes from `ask_cmd`/`chat_cmd` still expecting `Generator`, checked next.

- [ ] **Step 3: Trim `gritter/generation/prompts.py`**

Remove `SYSTEM_PROMPT` and `build_user_prompt`; keep only:

```python
from __future__ import annotations

from gritter.models.query import RetrievalResult


def build_context_block(result: RetrievalResult) -> str:
    chunk = result.chunk
    symbol_part = ""
    if chunk.symbol_name is not None:
        type_prefix = f"{chunk.symbol_type}: " if chunk.symbol_type is not None else ""
        symbol_part = f", {type_prefix}{chunk.symbol_name}"
    header = (
        f"--- File: {chunk.file_path} "
        f"(lines {chunk.start_line}-{chunk.end_line}{symbol_part}) ---"
    )
    return f"{header}\n{chunk.content}"
```

- [ ] **Step 4: Delete `gritter/generation/generator.py`**

```bash
rm gritter/generation/generator.py
```

- [ ] **Step 5: Rewrite `gritter/cli/ask_cmd.py`**

```python
from __future__ import annotations

from pathlib import Path
import typer
from rich.live import Live
from rich.text import Text

from gritter.agent.session import AgentSession, ToolFinished, ToolStarted
from gritter.agent.tools import ToolContext
from gritter.generation.citations import format_sources
from gritter.models.config import GritterConfig
from gritter.providers.llm import TextDelta, get_llm_provider
from gritter.retrieval.hybrid import HybridRetriever
from gritter.storage.index_meta import IndexMeta
from gritter.utils.display import console, print_error


def ask(
    question: str = typer.Argument(..., help="Natural language question about the codebase"),
    name: str = typer.Option(None, "--name", "-n", help="Index name to query (defaults to current directory name)"),
    top_k: int = typer.Option(None, "--top-k", help="Number of results to retrieve"),
) -> None:
    """Ask a question about an indexed codebase."""
    config = GritterConfig()
    index_name = name or Path.cwd().name
    if top_k is not None:
        config.retrieval.top_k = top_k

    try:
        retriever = HybridRetriever.from_config(index_name, config)
        source_root = IndexMeta(config.index_dir(index_name)).read()["source_root"]
    except (ValueError, FileNotFoundError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    try:
        llm = get_llm_provider(config.llm.provider, config.llm.model)
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    ctx = ToolContext(
        repo_root=Path(source_root), retriever=retriever, config=config, index_name=index_name,
    )
    session = AgentSession(llm, ctx)

    response_text = Text()
    console.print()
    with Live(response_text, console=console, refresh_per_second=15):
        for event in session.run(question):
            if isinstance(event, TextDelta):
                response_text.append(event.text)
            elif isinstance(event, ToolStarted):
                console.print(f"[dim]→ {event.name}({event.arguments})[/dim]")
    console.print()

    sources = format_sources(session.last_citations)
    if sources:
        console.print(f"\n[dim]{sources}[/dim]")
```

- [ ] **Step 6: Rewrite `gritter/cli/chat_cmd.py`**

```python
from __future__ import annotations

from pathlib import Path
import typer
from rich.live import Live
from rich.text import Text

from gritter.agent.session import AgentSession, ToolFinished, ToolStarted
from gritter.agent.tools import ToolContext
from gritter.generation.citations import format_sources
from gritter.models.config import GritterConfig
from gritter.providers.llm import TextDelta, get_llm_provider
from gritter.retrieval.hybrid import HybridRetriever
from gritter.storage.index_meta import IndexMeta
from gritter.utils.display import console, print_error


def chat(
    name: str = typer.Option(None, "--name", "-n", help="Index name to query (defaults to current directory name)"),
) -> None:
    """Interactive multi-turn chat about an indexed codebase."""
    config = GritterConfig()
    index_name = name or Path.cwd().name

    try:
        retriever = HybridRetriever.from_config(index_name, config)
        source_root = IndexMeta(config.index_dir(index_name)).read()["source_root"]
    except (ValueError, FileNotFoundError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    try:
        llm = get_llm_provider(config.llm.provider, config.llm.model)
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    ctx = ToolContext(
        repo_root=Path(source_root), retriever=retriever, config=config, index_name=index_name,
    )
    session = AgentSession(llm, ctx)

    console.print(
        f"\n[bold]Gritter chat[/bold] — index: [cyan]{index_name}[/cyan]  "
        "[dim](type 'exit' or press Ctrl+C to quit)[/dim]\n"
    )

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        response_text = Text()
        console.print("\n[bold]Gritter:[/bold] ", end="")
        with Live(response_text, console=console, refresh_per_second=15):
            for event in session.run(query):
                if isinstance(event, TextDelta):
                    response_text.append(event.text)
                elif isinstance(event, ToolStarted):
                    console.print(f"[dim]→ {event.name}({event.arguments})[/dim]")
        console.print()

        sources = format_sources(session.last_citations)
        if sources:
            console.print(f"[dim]{sources}[/dim]")

        console.print()
```

- [ ] **Step 7: Update `tests/test_e2e.py`**

Replace `test_generation_returns_nonempty_response` (which imports the now-deleted `Generator`) with:

```python
def test_agent_session_returns_nonempty_response_with_citation(indexed_dir):
    """AgentSession should produce a non-empty response with at least one verified citation."""
    from gritter.agent.session import AgentSession
    from gritter.agent.tools import ToolContext
    from gritter.models.config import GritterConfig
    from gritter.providers.llm import TextDelta, get_llm_provider
    from gritter.retrieval.hybrid import HybridRetriever
    from gritter.storage.index_meta import IndexMeta

    config, index_name, root = indexed_dir
    retriever = HybridRetriever.from_config(index_name, config)
    source_root = IndexMeta(config.index_dir(index_name)).read()["source_root"]

    llm = get_llm_provider(config.llm.provider, config.llm.model)
    ctx = ToolContext(repo_root=Path(source_root), retriever=retriever, config=config, index_name=index_name)
    session = AgentSession(llm, ctx)

    events = list(session.run("What does the add function do?"))
    response = "".join(e.text for e in events if isinstance(e, TextDelta))

    assert len(response) > 0, "Expected non-empty response"
    assert len(session.messages) >= 2
    assert len(session.last_citations) >= 1, f"Expected at least one citation in: {response[:300]}"
```

Check the top of `tests/test_e2e.py` for a `from pathlib import Path` import — it's already there (used by `FIXTURE_ROOT`), so no new import is needed beyond what's shown above. Confirm the `indexed_dir` fixture returns `(config, index_name, root)` — read the existing fixture to check the exact tuple shape before finalizing this edit, and adjust unpacking if it differs.

- [ ] **Step 8: Run the full test suite (excluding e2e, which needs real API keys)**

Run: `pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: PASS across the board.

- [ ] **Step 9: Manual smoke test against gritter's own repo (requires `ANTHROPIC_API_KEY`)**

```bash
pip install -e .
gritter index . --name gritter-self
gritter ask "How does hybrid retrieval combine dense and sparse results?" --name gritter-self
```

Expected: tool-call status lines (`→ search_code(...)`) print, followed by a streamed answer citing `gritter/retrieval/hybrid.py` or `gritter/retrieval/fusion.py`, followed by a `Sources:` block with no `[unverified]` tags (since the citations should point at real files).

- [ ] **Step 10: Commit**

```bash
git add gritter/cli/ask_cmd.py gritter/cli/chat_cmd.py gritter/generation/prompts.py \
        tests/test_generation.py tests/test_e2e.py
git rm gritter/generation/generator.py
git commit -m "feat: rewire ask/chat onto AgentSession, remove fixed-pipeline Generator"
```

---

## Task 7: Version bump and release

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump the version**

In `pyproject.toml`, change:

```toml
[project]
name = "gritter"
version = "0.1.0"
```

to:

```toml
[project]
name = "gritter"
version = "1.0.0"
```

- [ ] **Step 2: Run the full test suite one more time**

Run: `pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: PASS

- [ ] **Step 3: Commit the version bump**

```bash
git add pyproject.toml
git commit -m "chore: bump version to 1.0.0"
```

- [ ] **Step 4: Handle the stray local `v1.0.0` tag, then tag and push**

A local `v1.0.0` tag already exists pointing at `ca0e872` (an old, unrelated commit) and was never pushed to `origin`. Move it to the new release commit:

```bash
git tag -d v1.0.0
git tag v1.0.0
git push origin main
git push origin v1.0.0
```

This triggers the existing `.github/workflows/publish.yml` trusted-publishing workflow, which builds and publishes to PyPI via OIDC — no token needed since the pending publisher is already registered.

- [ ] **Step 5: Verify the release**

Run: `curl -s https://pypi.org/pypi/gritter/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"`
Expected: `1.0.0` (allow a minute or two for the GitHub Actions workflow to finish and PyPI's CDN to catch up).

---

## Self-Review Notes

- **Spec coverage:** provider tool-calling requirement (Task 2), all 5 tools (Task 3), citation verification (Task 4), loop cap of 8 with forced final answer (Task 5, `test_session_stops_at_max_tool_calls`), CLI rewire replacing `ask`/`chat` outright (Task 6), eval harness left untouched (no task touches `gritter/eval/`), PyPI release as `1.0.0` (Task 7) — all covered.
- **Type consistency:** `ToolCall` (provider event + stored call in `Message.tool_calls`) is one dataclass used consistently from Task 2 through Task 6, not split into a separate `ToolCallRequest` — checked across all task code blocks.
- **New gap found and closed during planning:** the original spec didn't say how tools would find the indexed repo's root path on disk. `IndexMeta` had no such field. Task 1 adds `source_root` before any tool that needs it (Task 3) is built.
