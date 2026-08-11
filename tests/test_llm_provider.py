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
