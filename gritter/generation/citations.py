from __future__ import annotations

import re

# Matches a citation of the form:
#   - a path containing at least one "/" (e.g. src/auth/jwt.py), OR
#   - a path ending in a known extension (.py, .ts, .tsx, .js, .rs)
# followed by :L<digits> and an optional -<digits> range.
_CITATION_RE = re.compile(
    r"(?:[\w.\-]+/[\w./\-]+|[\w./\-]+\.(?:py|ts|tsx|js|rs))"
    r":L\d+(?:-\d+)?"
)


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


def format_sources(citations: list[str]) -> str:
    """Format a list of citation strings as a Sources block, or '' if empty."""
    if not citations:
        return ""
    lines = ["Sources:"] + [f"  \u2022 {c}" for c in citations]
    return "\n".join(lines)
