from __future__ import annotations
from gritter.models.query import RetrievalResult
from gritter.storage.vector_store import VectorStore


def dense_search(
    query_embedding: list[float],
    store: VectorStore,
    k: int = 20,
) -> list[RetrievalResult]:
    """Query ChromaDB and return top-k results ordered by cosine similarity."""
    results = store.query(query_embedding, k=k)
    return [RetrievalResult(chunk=chunk, score=1.0 - dist) for chunk, dist in results]
