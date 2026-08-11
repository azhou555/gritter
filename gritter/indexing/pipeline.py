from __future__ import annotations

import logging
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TextColumn

from gritter.models.chunk import CodeChunk
from gritter.models.config import GritterConfig
from gritter.indexing.discovery import discover_files
from gritter.indexing.chunker import chunk_file
from gritter.indexing.incremental import get_changed_files, get_current_commit
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
    full: bool = False,
) -> dict:
    """Discover → chunk → embed → store.

    When full=False (default) and an existing index with a recorded commit SHA
    exists, only files changed since that commit are re-processed (incremental
    mode). Pass full=True to force a complete re-index regardless.

    Returns a summary dict with file_count, chunk_count, languages.
    """
    index_dir = config.index_dir(index_name)
    index_dir.mkdir(parents=True, exist_ok=True)

    current_commit = get_current_commit(root)

    # Decide whether to run incrementally
    if not full and current_commit:
        meta = IndexMeta(index_dir)
        try:
            stored = meta.read()
            prev_commit = stored.get("indexed_commit")
        except FileNotFoundError:
            prev_commit = None

        if prev_commit and prev_commit != current_commit:
            return _run_incremental(
                root, index_dir, config, embed_provider, prev_commit, current_commit
            )

    return _run_full(root, index_dir, config, embed_provider, current_commit)


# ---------------------------------------------------------------------------
# Full index
# ---------------------------------------------------------------------------

def _run_full(
    root: Path,
    index_dir: Path,
    config: GritterConfig,
    embed_provider: EmbeddingProvider,
    current_commit: str | None,
) -> dict:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=False,
    ) as progress:
        discover_task = progress.add_task("Discovering files...", total=None)
        file_pairs = discover_files(root, config.index.exclude_globs)
        progress.update(
            discover_task, total=1, completed=1,
            description=f"Found {len(file_pairs)} files",
        )

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

        batch_size = 64
        embed_task = progress.add_task("Embedding...", total=len(all_chunks))
        all_embeddings: list[list[float]] = []

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c.embedding_text() for c in batch]
            embeddings = embed_provider.embed(texts)
            all_embeddings.extend(embeddings)
            progress.advance(embed_task, advance=len(batch))

        store_task = progress.add_task("Storing...", total=None)

        vector_store = VectorStore(index_dir)
        # Clear any existing data on full re-index
        try:
            vector_store.delete_collection()
            vector_store = VectorStore(index_dir)
        except Exception:
            pass
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
            source_root=str(root),
            indexed_commit=current_commit,
        )

        progress.update(store_task, total=1, completed=1, description="Stored ✓")

    return {
        "file_count": len(file_pairs),
        "chunk_count": len(all_chunks),
        "languages": sorted(languages_seen),
    }


# ---------------------------------------------------------------------------
# Incremental index
# ---------------------------------------------------------------------------

def _run_incremental(
    root: Path,
    index_dir: Path,
    config: GritterConfig,
    embed_provider: EmbeddingProvider,
    prev_commit: str,
    current_commit: str,
) -> dict:
    modified, deleted = get_changed_files(root, prev_commit)

    # Filter to only files we actually support
    from gritter.utils.languages import detect_language
    supported_modified = [
        p for p in modified
        if p.exists() and detect_language(str(p)) is not None
    ]

    if not supported_modified and not deleted:
        # Nothing changed that we care about — update commit and return
        vector_store = VectorStore(index_dir)
        meta = IndexMeta(index_dir)
        stored = meta.read()
        meta.write(
            embedding_provider=stored["embedding_provider"],
            embedding_model=stored["embedding_model"],
            embedding_dimension=stored["embedding_dimension"],
            file_count=stored["file_count"],
            chunk_count=stored["chunk_count"],
            languages=stored["languages"],
            source_root=stored.get("source_root", str(root)),
            indexed_commit=current_commit,
        )
        return {
            "file_count": stored["file_count"],
            "chunk_count": stored["chunk_count"],
            "languages": stored["languages"],
            "incremental": True,
            "modified": 0,
            "deleted": 0,
        }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=False,
    ) as progress:
        vector_store = VectorStore(index_dir)

        # Step 1: Remove stale chunks for changed and deleted files
        remove_task = progress.add_task(
            f"Removing stale chunks ({len(supported_modified) + len(deleted)} files)...",
            total=None,
        )
        for file_path in supported_modified + deleted:
            rel_path = str(file_path.relative_to(root))
            vector_store.delete_by_file_path(rel_path)
        progress.update(remove_task, total=1, completed=1, description="Stale chunks removed ✓")

        # Step 2: Re-chunk and re-embed modified files
        new_chunks: list[CodeChunk] = []
        languages_seen: set[str] = set()

        chunk_task = progress.add_task("Chunking changes...", total=len(supported_modified))
        for file_path in supported_modified:
            language = detect_language(str(file_path))
            if language is None:
                progress.advance(chunk_task)
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                rel_path = str(file_path.relative_to(root))
                chunks = chunk_file(
                    content, rel_path, language,
                    min_tokens=config.index.chunk_min_tokens,
                    max_tokens=config.index.chunk_max_tokens,
                    overlap_tokens=config.index.chunk_overlap_tokens,
                )
                new_chunks.extend(chunks)
                languages_seen.add(language)
            except Exception as exc:
                logger.warning("Skipping %s: %s", file_path, exc)
            progress.advance(chunk_task)

        progress.update(chunk_task, description=f"Chunked → {len(new_chunks)} new chunks")

        # Step 3: Embed new chunks
        if new_chunks:
            batch_size = 64
            embed_task = progress.add_task("Embedding...", total=len(new_chunks))
            new_embeddings: list[list[float]] = []

            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks[i:i + batch_size]
                texts = [c.embedding_text() for c in batch]
                embeddings = embed_provider.embed(texts)
                new_embeddings.extend(embeddings)
                progress.advance(embed_task, advance=len(batch))

            vector_store.add(new_chunks, new_embeddings)

        # Step 4: Rebuild BM25 from all current ChromaDB chunks
        store_task = progress.add_task("Rebuilding BM25...", total=None)
        all_current_chunks = vector_store.get_all()
        bm25_store = BM25Store(index_dir)
        bm25_store.build(all_current_chunks)
        progress.update(store_task, total=1, completed=1, description="BM25 rebuilt ✓")

        # Step 5: Update metadata
        stored_meta = IndexMeta(index_dir).read()
        all_languages = sorted(
            set(stored_meta.get("languages", [])) | languages_seen
        )
        meta = IndexMeta(index_dir)
        meta.write(
            embedding_provider=stored_meta["embedding_provider"],
            embedding_model=stored_meta["embedding_model"],
            embedding_dimension=stored_meta["embedding_dimension"],
            file_count=vector_store.count(),  # approximate via chunk count
            chunk_count=len(all_current_chunks),
            languages=all_languages,
            source_root=stored_meta.get("source_root", str(root)),
            indexed_commit=current_commit,
        )

    return {
        "file_count": len(all_current_chunks),  # best approximation without full scan
        "chunk_count": len(all_current_chunks),
        "languages": all_languages,
        "incremental": True,
        "modified": len(supported_modified),
        "deleted": len(deleted),
    }


def _default_model(provider: str) -> str:
    return {
        "voyage": "voyage-code-3",
        "openai": "text-embedding-3-small",
        "local": "nomic-ai/nomic-embed-text-v1",
    }.get(provider, "unknown")
