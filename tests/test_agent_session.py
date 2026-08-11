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
