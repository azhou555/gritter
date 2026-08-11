from __future__ import annotations

import re
from pathlib import Path

# Matches a citation of the form:
#   - a path containing at least one "/" (e.g. src/auth/jwt.py), OR
#   - a path ending in a known extension (.py, .ts, .tsx, .js, .rs)
# followed by :L<digits> and an optional -<digits> range.
_CITATION_RE = re.compile(
    r"(?:[\w.\-]+/[\w./\-]+|[\w./\-]+\.(?:py|ts|tsx|js|rs))"
    r":L\d+(?:-\d+)?"
)

_CITATION_PARSE_RE = re.compile(r"^(?P<path>.+):L(?P<start>\d+)(?:-(?P<end>\d+))?$")


def extract_citations(text: str) -> list[str]:
    """Return deduplicated citations from text, preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for match in _CITATION_RE.finditer(text):
        citation = match.group(0)
        if citation not in seen:
            seen.add(citation)
            result.append(citation)
    return result


def verify_citations(citations: list[str], repo_root: Path) -> list[tuple[str, bool]]:
    """Check each citation's path/line-range against the real file on disk."""
    return [(citation, _verify_one(citation, repo_root)) for citation in citations]


def _verify_one(citation: str, repo_root: Path) -> bool:
    match = _CITATION_PARSE_RE.match(citation)
    if not match:
        return False

    rel_path = match.group("path")
    start = int(match.group("start"))
    end = int(match.group("end")) if match.group("end") else start

    repo_root_resolved = repo_root.resolve()
    target = (repo_root_resolved / rel_path).resolve()
    if not target.is_relative_to(repo_root_resolved):
        return False
    if not target.is_file():
        return False

    try:
        line_count = sum(1 for _ in target.open(encoding="utf-8", errors="replace"))
    except OSError:
        return False

    return 1 <= start <= end <= line_count


def format_sources(verified: list[tuple[str, bool]]) -> str:
    """Format verified (citation, is_verified) pairs as a Sources block, or '' if empty."""
    if not verified:
        return ""
    lines = ["Sources:"]
    for citation, ok in verified:
        suffix = "" if ok else " [unverified]"
        lines.append(f"  • {citation}{suffix}")
    return "\n".join(lines)
