from __future__ import annotations

from abc import ABC, abstractmethod

from gritter.models.query import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Re-order results by relevance to query. Returns a reordered list."""


class NoOpReranker(Reranker):
    """Identity reranker — returns results unchanged. Used when reranking is disabled."""

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return results


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker using sentence-transformers.

    Scores each (query, chunk) pair jointly, which is more accurate than
    comparing independent embeddings but slower — suitable for reranking
    a small candidate set (top-20) down to top-k.
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2") -> None:
        import contextlib
        import io
        import logging
        import warnings
        from sentence_transformers import CrossEncoder

        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.filterwarnings("ignore", message=".*rope_parameters.*")
            self._model = CrossEncoder(model)

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return results

        pairs = [(query, r.chunk.content) for r in results]
        scores: list[float] = self._model.predict(pairs).tolist()

        reranked = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            RetrievalResult(chunk=r.chunk, score=score)
            for score, r in reranked
        ]
