from __future__ import annotations
import os
from pathlib import Path

from gritter.utils.languages import (
    detect_language,
    EXCLUDED_DIRS,
    EXCLUDED_EXTENSIONS,
    EXCLUDED_FILENAMES,
)


def discover_files(
    root: Path,
    exclude_globs: list[str] | None = None,
) -> list[tuple[Path, str]]:
    """Walk root and return (path, language) pairs for all supported files."""
    exclude_globs = exclude_globs or []
    results: list[tuple[Path, str]] = []

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)

        # Prune excluded and hidden directories in-place to avoid descending into them
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in EXCLUDED_DIRS
        ]

        for filename in filenames:
            path = current / filename

            # Skip hidden files
            if filename.startswith("."):
                continue

            # Skip excluded filenames
            if filename in EXCLUDED_FILENAMES:
                continue

            # Skip excluded extensions
            if path.suffix.lower() in EXCLUDED_EXTENSIONS:
                continue

            # Skip user-configured glob patterns
            rel = path.relative_to(root)
            if any(rel.match(g) for g in exclude_globs):
                continue

            language = detect_language(str(path))
            if language is not None:
                results.append((path, language))

    return results
