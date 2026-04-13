from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path


class IndexMeta:
    """JSON file recording index provenance and statistics."""

    def __init__(self, index_dir: Path) -> None:
        self._path = index_dir / "meta.json"
        self._data: dict = {}

    def write(
        self,
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        file_count: int,
        chunk_count: int,
        languages: list[str],
        indexed_commit: str | None = None,
    ) -> None:
        self._data = {
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "file_count": file_count,
            "chunk_count": chunk_count,
            "languages": sorted(languages),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "indexed_commit": indexed_commit,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def read(self) -> dict:
        if not self._path.exists():
            raise FileNotFoundError(f"Index metadata not found at {self._path}")
        self._data = json.loads(self._path.read_text())
        return self._data

    def validate_provider(self, provider: str, model: str, dimension: int) -> None:
        """Raise if the configured provider/model doesn't match the stored index."""
        meta = self.read()
        if meta["embedding_provider"] != provider or meta["embedding_dimension"] != dimension:
            raise ValueError(
                f"Provider mismatch: index was built with "
                f"{meta['embedding_provider']} ({meta['embedding_dimension']}d) "
                f"but current config is {provider} ({dimension}d). "
                f"Re-run `gritter index` to rebuild the index."
            )
