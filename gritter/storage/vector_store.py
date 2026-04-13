from __future__ import annotations
from pathlib import Path
import chromadb
from gritter.models.chunk import CodeChunk


class VectorStore:
    """ChromaDB-backed vector store. One collection per index."""

    def __init__(self, index_dir: Path, collection_name: str = "chunks") -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings."""
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "file_path": c.file_path,
                    "language": c.language,
                    "symbol_name": c.symbol_name or "",
                    "symbol_type": c.symbol_type or "",
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "imports": "\n".join(c.imports),
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], k: int = 20) -> list[tuple[CodeChunk, float]]:
        """Return up to k (chunk, distance) pairs ordered by cosine similarity."""
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
        )
        chunks_and_scores: list[tuple[CodeChunk, float]] = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunk = CodeChunk(
                content=doc,
                file_path=meta["file_path"],
                language=meta["language"],
                symbol_name=meta["symbol_name"] or None,
                symbol_type=meta["symbol_type"] or None,
                start_line=int(meta["start_line"]),
                end_line=int(meta["end_line"]),
                imports=meta["imports"].split("\n") if meta["imports"] else [],
            )
            chunks_and_scores.append((chunk, float(dist)))
        return chunks_and_scores

    def delete_by_file_path(self, file_path: str) -> None:
        """Delete all chunks whose file_path metadata matches."""
        self._collection.delete(where={"file_path": file_path})

    def get_all(self) -> list[CodeChunk]:
        """Return all stored chunks (no embeddings). Used for BM25 rebuild."""
        result = self._collection.get(include=["documents", "metadatas"])
        chunks: list[CodeChunk] = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            chunks.append(CodeChunk(
                content=doc,
                file_path=meta["file_path"],
                language=meta["language"],
                symbol_name=meta["symbol_name"] or None,
                symbol_type=meta["symbol_type"] or None,
                start_line=int(meta["start_line"]),
                end_line=int(meta["end_line"]),
                imports=meta["imports"].split("\n") if meta["imports"] else [],
            ))
        return chunks

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection.name)
