from __future__ import annotations
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from gritter.models.chunk import CodeChunk


class BM25Store:
    """BM25 index over chunk text, persisted to disk."""

    def __init__(self, index_dir: Path) -> None:
        self._path = index_dir / "bm25.pkl"
        self._bm25: BM25Okapi | None = None
        self._chunks: list[CodeChunk] = []

    def build(self, chunks: list[CodeChunk]) -> None:
        """Build BM25 index from chunks and persist to disk."""
        self._chunks = chunks
        tokenized = [c.content.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._save()

    def load(self) -> None:
        """Load index from disk."""
        if not self._path.exists():
            raise FileNotFoundError(f"BM25 index not found at {self._path}")
        with open(self._path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]

    def query(self, query: str, k: int = 20) -> list[tuple[CodeChunk, float]]:
        """Return up to k (chunk, bm25_score) pairs."""
        if self._bm25 is None:
            raise RuntimeError("BM25 index not loaded. Call load() first.")
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)
