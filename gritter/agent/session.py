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
