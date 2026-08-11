# Agentic Gritter — Design Spec

**Date:** 2026-08-10
**Status:** Approved

---

## Elevator Pitch

Replace gritter's fixed retrieve-then-generate pipeline with an agentic tool-calling loop: the LLM decides when to search, read a file, grep, chase a symbol, or trigger a reindex, instead of gritter always running exactly one hybrid search per turn.

---

## Goals

- Let the model recover from a bad initial retrieval by searching again, reading a full file, or grepping for a symbol, instead of answering from a single fixed top-k.
- Close the existing citation-trust gap (`DESIGN.md`: citations are regex-extracted and never checked against real files) using the same file-access tools the agent now has.
- Ship through the existing PyPI trusted-publishing pipeline as `1.0.0`.

## Non-Goals

- Keeping the old fixed-pipeline path alongside the new one — `ask`/`chat` are replaced outright, not dual-mode.
- Reworking the eval harness (`recall@k`/`MRR`) — it measures `search_code`'s underlying retrieval quality, which is unchanged.
- Text-based tool-call parsing fallback for non-tool-calling models — agent mode requires native tool-calling support from the provider (all three current providers have it).

---

## Architecture Overview

**Current (removed):** `HybridRetriever.search(query)` → one fixed top-k → `Generator.stream(query, results)`.

**New:** `AgentSession` runs a bounded loop (max 8 tool calls per turn):

1. Send system prompt (toolbox description + grounding rules) + conversation history to the LLM via `LLMProvider.run_tools`.
2. Model emits either a tool call or final text.
3. On tool call: dispatch to the matching tool function, append the result as a tool-result message, go to 1.
4. On final text: run citation verification, print the answer, stop.
5. If the 8-call cap is hit: force a final answer from whatever context has been gathered so far (send one more turn with tool access removed).

`ask` (single-turn) and `chat` (multi-turn) both drive `AgentSession`; `chat` simply keeps the message history across turns.

---

## Provider Interface

Current `LLMProvider.stream(system, messages) -> Iterator[str]` cannot carry tool calls or tool results. It is replaced (not kept alongside) with:

```python
@dataclass
class TextDelta:
    text: str

@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict

@dataclass
class Done:
    pass

AgentEvent = TextDelta | ToolCallRequest | Done

class LLMProvider(ABC):
    @abstractmethod
    def run_tools(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> Iterator[AgentEvent]:
        """Stream text and/or emit tool call requests until the model is done."""
```

Each provider implements this against its native tool-calling API:
- `ClaudeProvider` — Anthropic `tool_use` content blocks.
- `OpenAIProvider` — OpenAI `tools` / function calling.
- `OllamaProvider` — same OpenAI-compatible surface already in use; requires a tool-calling-capable local model (documented as a requirement, not silently degraded).

The caller (`AgentSession`) is responsible for appending a `ToolResult` message and re-invoking `run_tools` after dispatching a `ToolCallRequest`.

---

## Tools

New module `gritter/agent/tools.py`. Each tool is a plain function taking parsed arguments and returning a string (or structured result rendered to string) fed back to the model.

| Tool | Backing | Notes |
|---|---|---|
| `search_code(query, top_k?)` | existing `HybridRetriever.search` | same dense+sparse+RRF+rerank path as today |
| `read_file(path, start_line?, end_line?)` | plain file read | resolved against and confined to the indexed repo root; rejects `..` traversal and absolute paths outside root |
| `grep(pattern, path_glob?)` | `rg` subprocess | falls back to a Python `re` walk if `rg` isn't on `PATH` |
| `list_symbols(path)` | existing `ast_chunker` symbol extraction, run on one file | returns name/type/line-range per symbol; heuristic-chunked languages return the module-level chunk boundaries |
| `reindex()` | existing incremental (git-diff) indexing pipeline | agent calls this only when it suspects staleness (e.g. search results reference code that doesn't match `read_file` output) |

Tool specs (name, description, JSON-schema args) live alongside the functions and are passed to `run_tools` as `ToolSpec` objects — one definition, translated to each provider's schema format at the provider boundary.

---

## Citation Verification

New `gritter/generation/citations.py` addition: `verify_citations(citations, repo_root) -> list[VerifiedCitation]`. For each extracted `path:Lstart-end`:
- File must exist under the repo root.
- `Lstart-end` must be within the file's actual line count.

Citations that fail verification are marked inline (e.g. `path:L10-20 [unverified]`) in the printed source list rather than silently dropped — the model's claim stays visible, but flagged as unchecked.

---

## CLI Surface

`gritter ask` and `gritter chat` keep their existing arguments and flags. Internally, both build one `AgentSession` (via a `from_config` classmethod mirroring `HybridRetriever.from_config`) and drive it instead of calling `HybridRetriever.search` + `Generator.stream` directly. Tool-call activity prints as a dim status line (tool name + key args) above the streamed final answer, consistent with the current Rich `Live` display pattern.

`gritter/generation/generator.py`'s `Generator` class is removed; its responsibilities (message history, streaming to `Live`) move into `AgentSession`.

---

## Error Handling

- Tool dispatch errors (bad path, grep failure, reindex failure) are caught and returned to the model as a tool-result error string, not raised — lets the model try a different approach instead of crashing the turn.
- Loop-cap exhaustion is not an error: it forces a best-effort final answer, logged at debug level.
- Provider/env errors (missing API key, no tool-calling support) still raise before the loop starts, same as current `EnvironmentError` handling in `ask_cmd.py`/`chat_cmd.py`.

---

## Testing

Follows existing style (`tests/test_generation.py`, `tests/test_retrieval.py`): mock `LLMProvider.run_tools` to emit scripted `AgentEvent` sequences.

- Loop dispatches a `ToolCallRequest` to the right tool and appends its result.
- Loop stops on `Done`.
- Loop force-answers at the 8-call cap.
- `read_file` rejects path traversal outside repo root.
- `verify_citations` flags a citation pointing at a nonexistent file / out-of-range lines, passes a real one.
- `chat` preserves message history (including prior tool results) across turns.

---

## Release

1. Implement + tests green.
2. Manual smoke test: `gritter ask` against gritter's own repo, confirm tool calls fire and citations verify.
3. Bump `pyproject.toml` version to `1.0.0`.
4. `git tag v1.0.0 && git push --tags` — triggers existing `publish.yml` trusted-publishing workflow (no new PyPI setup required; `gritter` 0.1.0 is already live). Note: a local `v1.0.0` tag already exists from a prior session and was never pushed — verify it isn't pointing at unrelated work before reusing it, or delete/replace it once this is ready to tag.
