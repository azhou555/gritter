from __future__ import annotations
from abc import ABC, abstractmethod
from gritter.models.query import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Re-order results by relevance to query. Returns a reordered list."""


class NoOpReranker(Reranker):
    """Identity reranker — returns results unchanged. Placeholder for v1.1."""

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return results
