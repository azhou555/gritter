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

def _messages_to_anthropic(messages: list[Message], strip_tool_blocks: bool = False) -> list[dict]:
    """Convert provider-agnostic messages into Anthropic wire format.

    When `strip_tool_blocks` is True, prior tool_use/tool_result content is
    rewritten as plain text blocks instead of Anthropic's structured
    tool_use/tool_result types. This is used when sending a request with no
    `tools` declared (e.g. the forced-final-answer turn after the tool-call
    cap is hit) — carrying tool_use/tool_result blocks in history alongside
    an empty `tools` list is a combination the Anthropic API may reject, so
    we defensively downgrade that history to plain text regardless of the
    exact validation rule.
    """
    converted: list[dict] = []
    for m in messages:
        if m.role == "tool":
            if strip_tool_blocks:
                converted.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"Tool result: {m.content}"}],
                })
            else:
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
                if strip_tool_blocks:
                    blocks.append({
                        "type": "text",
                        "text": f"I called `{tc.name}` with `{json.dumps(tc.arguments)}`",
                    })
                else:
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
            "messages": _messages_to_anthropic(messages, strip_tool_blocks=not tools),
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
        if not chunk.choices:
            continue
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
