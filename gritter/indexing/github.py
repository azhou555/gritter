from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Matches:
#   https://github.com/owner/repo
#   https://github.com/owner/repo.git
#   http://github.com/owner/repo
#   github.com/owner/repo
_GITHUB_RE = re.compile(
    r'^(?:https?://)?github\.com/([^/]+)/([^/\s]+?)(?:\.git)?/?$'
)


def is_github_url(s: str) -> bool:
    """Return True if s looks like a GitHub repo URL."""
    return bool(_GITHUB_RE.match(s.strip()))


def parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, repo) extracted from a GitHub URL.

    Raises ValueError if the URL doesn't match.
    """
    m = _GITHUB_RE.match(url.strip())
    if not m:
        raise ValueError(f"Not a recognised GitHub URL: {url!r}")
    return m.group(1), m.group(2)


def _run(args: list[str], cwd: Path | None = None) -> None:
    """Run a git command, raising RuntimeError on non-zero exit."""
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n"
            f"stderr: {result.stderr.strip()}"
        )


def ensure_repo(url: str, branch: str | None, cache_dir: Path) -> Path:
    """Return a local path to a cloned (and up-to-date) copy of the GitHub repo.

    On first call: blobless clone into cache_dir/<owner>__<repo>.
    On subsequent calls: fetch + reset to origin/<branch> (or origin/HEAD).

    The blobless clone (--filter=blob:none) downloads full commit history
    so git-diff works for incremental indexing, but fetches file contents
    lazily — much faster than a full clone for large repos.
    """
    owner, repo = parse_github_url(url)
    clone_path = cache_dir / f"{owner}__{repo}"

    if clone_path.exists():
        _run(["git", "-C", str(clone_path), "fetch", "origin"])
        target = f"origin/{branch}" if branch else "origin/HEAD"
        _run(["git", "-C", str(clone_path), "reset", "--hard", target])
    else:
        clone_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--filter=blob:none", url, str(clone_path)]
        if branch:
            cmd += ["--branch", branch]
        _run(cmd)

    return clone_path
