from __future__ import annotations
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


class LLMProvider(ABC):
    @abstractmethod
    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        """Stream response tokens for the given system prompt and message history."""


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

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        ) as stream:
            for text in stream.text_stream:
                yield text


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

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OllamaProvider(LLMProvider):
    """Local Ollama — no API key required."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        from openai import OpenAI
        self._client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        self._model = model

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


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
        return OllamaProvider(model or "llama3", base_url or "http://localhost:11434")
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
