from __future__ import annotations
from pathlib import Path

from gritter.models.config import GritterConfig
from gritter.models.query import RetrievalResult
from gritter.providers.embeddings import EmbeddingProvider, get_embedding_provider
from gritter.retrieval.dense import dense_search
from gritter.retrieval.sparse import sparse_search
from gritter.retrieval.fusion import reciprocal_rank_fusion
from gritter.retrieval.reranker import Reranker, NoOpReranker
from gritter.storage.vector_store import VectorStore
from gritter.storage.bm25_store import BM25Store
from gritter.storage.index_meta import IndexMeta


class HybridRetriever:
    """Orchestrates dense + sparse retrieval + RRF + reranking."""

    def __init__(
        self,
        index_dir: Path,
        embed_provider: EmbeddingProvider,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
        top_k: int = 5,
    ) -> None:
        self._embed_provider = embed_provider
        self._reranker = reranker or NoOpReranker()
        self._candidate_k = candidate_k
        self._top_k = top_k

        self._vector_store = VectorStore(index_dir)
        self._bm25_store = BM25Store(index_dir)
        self._bm25_store.load()

    def search(self, query: str) -> list[RetrievalResult]:
        """Return top_k results for a natural language query."""
        query_vec = self._embed_provider.embed_one(query)

        dense_results = dense_search(query_vec, self._vector_store, k=self._candidate_k)
        sparse_results = sparse_search(query, self._bm25_store, k=self._candidate_k)

        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        candidates = fused[:self._candidate_k]

        reranked = self._reranker.rerank(query, candidates)
        return reranked[:self._top_k]

    @classmethod
    def from_config(cls, index_name: str, config: GritterConfig) -> "HybridRetriever":
        """Build a HybridRetriever from GritterConfig, validating provider compatibility."""
        index_dir = config.index_dir(index_name)

        embed_provider = get_embedding_provider(
            config.embedding.provider, config.embedding.model
        )
        meta = IndexMeta(index_dir)
        meta.validate_provider(
            config.embedding.provider,
            config.embedding.model or "",
            embed_provider.dimension,
        )

        return cls(
            index_dir=index_dir,
            embed_provider=embed_provider,
            candidate_k=config.retrieval.candidate_k,
            top_k=config.retrieval.top_k,
        )
