from __future__ import annotations
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

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        # Skip hidden directories and known excluded dirs
        parts = path.relative_to(root).parts
        if any(part.startswith(".") or part in EXCLUDED_DIRS for part in parts[:-1]):
            continue

        # Skip excluded filenames
        if path.name in EXCLUDED_FILENAMES:
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
