from __future__ import annotations

from types import SimpleNamespace

from gritter.providers.llm import Done, Message, TextDelta, ToolCall, ToolSpec
from gritter.providers.llm import _messages_to_anthropic, _messages_to_openai, _run_openai_style_tools


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


def test_messages_to_anthropic_strips_tool_blocks_when_requested():
    """Finding 2: when a forced-final-answer turn sends no `tools`, prior
    tool_use/tool_result blocks must be downgraded to plain text so the
    Anthropic API request never carries tool_use/tool_result content
    alongside an empty `tools` list."""
    messages = [
        Message(
            role="assistant",
            content="checking...",
            tool_calls=[ToolCall(id="tc1", name="search_code", arguments={"query": "auth"})],
        ),
        Message(role="tool", content="found stuff", tool_call_id="tc1"),
    ]
    converted = _messages_to_anthropic(messages, strip_tool_blocks=True)

    def _block_types(msgs):
        types = []
        for m in msgs:
            content = m["content"]
            if isinstance(content, list):
                types.extend(b["type"] for b in content)
        return types

    types = _block_types(converted)
    assert "tool_use" not in types
    assert "tool_result" not in types
    assert types and all(t == "text" for t in types)
    # the tool call and its result are still represented, just as text
    assert any("search_code" in b["text"] for m in converted for b in m["content"])
    assert any("found stuff" in b["text"] for m in converted for b in m["content"])


def test_messages_to_anthropic_default_does_not_strip_tool_blocks():
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="tc1", name="search_code", arguments={"query": "auth"})],
        ),
        Message(role="tool", content="found stuff", tool_call_id="tc1"),
    ]
    converted = _messages_to_anthropic(messages)
    types = [b["type"] for m in converted for b in m["content"]]
    assert "tool_use" in types
    assert "tool_result" in types


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


def test_run_openai_style_tools_skips_empty_choices_chunks():
    """Minor fix: some OpenAI-compatible endpoints (incl. some Ollama
    versions) emit a trailing usage-reporting chunk with an empty `choices`
    list. Indexing into choices[0] on that chunk raises IndexError; it must
    be skipped instead."""
    text_chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="hello", tool_calls=None))]
    )
    empty_choices_chunk = SimpleNamespace(choices=[])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: iter([text_chunk, empty_choices_chunk])
            )
        )
    )

    events = list(_run_openai_style_tools(fake_client, "some-model", "system prompt", [], []))
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "hello"
    assert any(isinstance(e, Done) for e in events)
