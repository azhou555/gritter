from __future__ import annotations

from collections.abc import Iterator

from gritter.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from gritter.models.query import RetrievalResult
from gritter.providers.llm import LLMProvider, Message


class Generator:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self.messages: list[Message] = []

    def stream(self, query: str, results: list[RetrievalResult]) -> Iterator[str]:
        user_prompt = build_user_prompt(query, results)
        self.messages.append(Message(role="user", content=user_prompt))
        tokens: list[str] = []
        for token in self._llm.stream(SYSTEM_PROMPT, self.messages):
            tokens.append(token)
            yield token
        full_response = "".join(tokens)
        self.messages.append(Message(role="assistant", content=full_response))

    def reset(self) -> None:
        self.messages = []
