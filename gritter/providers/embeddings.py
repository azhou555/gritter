from __future__ import annotations
import os
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class VoyageEmbedder(EmbeddingProvider):
    """voyage-code-3 — optimized for source code retrieval."""

    def __init__(self, model: str = "voyage-code-3") -> None:
        import voyageai
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "VOYAGE_API_KEY environment variable is required for Voyage embeddings."
            )
        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self._model, input_type="document")
        return result.embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedder(EmbeddingProvider):
    """text-embedding-3-small — general-purpose, widely available."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for OpenAI embeddings."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimension = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedder(EmbeddingProvider):
    """sentence-transformers — runs locally, no API key required."""

    def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model, trust_remote_code=True)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_provider(
    provider: str,
    model: str | None = None,
) -> EmbeddingProvider:
    """Factory: return the configured embedding provider."""
    if provider == "voyage":
        return VoyageEmbedder(model or "voyage-code-3")
    elif provider == "openai":
        return OpenAIEmbedder(model or "text-embedding-3-small")
    elif provider == "local":
        return LocalEmbedder(model or "nomic-ai/nomic-embed-text-v1")
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}")
