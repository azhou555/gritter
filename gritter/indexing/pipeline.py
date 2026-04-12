from __future__ import annotations
import logging
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TextColumn

from gritter.models.chunk import CodeChunk
from gritter.models.config import GritterConfig
from gritter.indexing.discovery import discover_files
from gritter.indexing.chunker import chunk_file
from gritter.providers.embeddings import EmbeddingProvider
from gritter.storage.vector_store import VectorStore
from gritter.storage.bm25_store import BM25Store
from gritter.storage.index_meta import IndexMeta

logger = logging.getLogger(__name__)


def run_indexing_pipeline(
    root: Path,
    index_name: str,
    config: GritterConfig,
    embed_provider: EmbeddingProvider,
) -> dict:
    """Full indexing pipeline: discover → chunk → embed → store.

    Returns a summary dict with file_count, chunk_count, languages.
    """
    index_dir = config.index_dir(index_name)
    index_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=False,
    ) as progress:
        # Stage 1: Discovery
        discover_task = progress.add_task("Discovering files...", total=None)
        file_pairs = discover_files(root, config.index.exclude_globs)
        progress.update(
            discover_task, total=1, completed=1,
            description=f"Found {len(file_pairs)} files",
        )

        # Stage 2: Chunking
        chunk_task = progress.add_task("Chunking...", total=len(file_pairs))
        all_chunks: list[CodeChunk] = []
        languages_seen: set[str] = set()

        for file_path, language in file_pairs:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                rel_path = str(file_path.relative_to(root))
                chunks = chunk_file(
                    content, rel_path, language,
                    min_tokens=config.index.chunk_min_tokens,
                    max_tokens=config.index.chunk_max_tokens,
                    overlap_tokens=config.index.chunk_overlap_tokens,
                )
                all_chunks.extend(chunks)
                languages_seen.add(language)
            except Exception as exc:
                logger.warning("Skipping %s: %s", file_path, exc)
            progress.advance(chunk_task)

        progress.update(chunk_task, description=f"Chunked → {len(all_chunks)} chunks")

        # Stage 3: Embedding (batched)
        batch_size = 64
        embed_task = progress.add_task("Embedding...", total=len(all_chunks))
        all_embeddings: list[list[float]] = []

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c.embedding_text() for c in batch]
            embeddings = embed_provider.embed(texts)
            all_embeddings.extend(embeddings)
            progress.advance(embed_task, advance=len(batch))

        # Stage 4: Storage
        store_task = progress.add_task("Storing...", total=None)

        vector_store = VectorStore(index_dir)
        vector_store.add(all_chunks, all_embeddings)

        bm25_store = BM25Store(index_dir)
        bm25_store.build(all_chunks)

        meta = IndexMeta(index_dir)
        meta.write(
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model or _default_model(config.embedding.provider),
            embedding_dimension=embed_provider.dimension,
            file_count=len(file_pairs),
            chunk_count=len(all_chunks),
            languages=list(languages_seen),
        )

        progress.update(store_task, total=1, completed=1, description="Stored ✓")

    return {
        "file_count": len(file_pairs),
        "chunk_count": len(all_chunks),
        "languages": sorted(languages_seen),
    }


def _default_model(provider: str) -> str:
    return {
        "voyage": "voyage-code-3",
        "openai": "text-embedding-3-small",
        "local": "nomic-ai/nomic-embed-text-v1",
    }.get(provider, "unknown")
