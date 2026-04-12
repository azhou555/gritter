from __future__ import annotations
from gritter.models.query import RetrievalResult
from gritter.storage.bm25_store import BM25Store


def sparse_search(
    query: str,
    store: BM25Store,
    k: int = 20,
) -> list[RetrievalResult]:
    """Query BM25 index and return top-k results ordered by BM25 score."""
    results = store.query(query, k=k)
    return [RetrievalResult(chunk=chunk, score=score) for chunk, score in results]
