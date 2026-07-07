from __future__ import annotations

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".rs": "rust",
}

EXCLUDED_DIRS: set[str] = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    "dist", "build", "target", ".next", ".nuxt", ".cache",
    "venv", ".venv", "env", ".env", ".tox",
    "coverage", ".coverage", ".mypy_cache", ".ruff_cache",
    "tests", "test", "spec", "__tests__",
}

EXCLUDED_FILENAMES: set[str] = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Gemfile.lock",
    "composer.lock", "bun.lockb",
}

EXCLUDED_EXTENSIONS: set[str] = {
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll", ".exe",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".whl", ".egg",
    ".lock", ".map",
}


def detect_language(path: str) -> str | None:
    """Return the language name for a file path, or None if unsupported."""
    from pathlib import Path
    suffix = Path(path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(suffix)
