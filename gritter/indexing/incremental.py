from __future__ import annotations

import subprocess
from pathlib import Path


def get_current_commit(root: Path) -> str | None:
    """Return the current HEAD commit SHA, or None if root is not a git repo."""
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_changed_files(
    root: Path,
    since_commit: str,
) -> tuple[list[Path], list[Path]]:
    """Return (modified, deleted) paths changed since since_commit.

    modified: added, changed, or rename-targets — should be re-indexed
    deleted:  removed or rename-sources — should be dropped from the index

    Returns ([], []) if root is not a git repo or since_commit is unreachable.
    All paths are absolute.
    """
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-status", since_commit, "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [], []

    modified: list[Path] = []
    deleted: list[Path] = []

    for line in result.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        if status.startswith("D"):
            deleted.append(root / parts[1])
        elif status.startswith("R"):
            # Rename: old path deleted, new path added
            if len(parts) == 3:
                deleted.append(root / parts[1])
                modified.append(root / parts[2])
        else:
            # A (added), M (modified), C (copied)
            modified.append(root / parts[1])

    return modified, deleted
