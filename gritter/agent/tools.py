from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from gritter.generation.prompts import build_context_block
from gritter.models.config import GritterConfig
from gritter.providers.llm import ToolSpec
from gritter.retrieval.hybrid import HybridRetriever


@dataclass
class ToolContext:
    repo_root: Path
    retriever: HybridRetriever
    config: GritterConfig
    index_name: str


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="search_code",
        description=(
            "Hybrid dense+sparse semantic search over the indexed codebase. "
            "Returns the top matching code chunks with file paths and line ranges."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language or code search query."},
                "top_k": {"type": "integer", "description": "Max number of results to return."},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="read_file",
        description="Read a file from the indexed repository, optionally restricted to a line range.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative file path."},
                "start_line": {"type": "integer", "description": "First line to include (1-indexed)."},
                "end_line": {"type": "integer", "description": "Last line to include (1-indexed)."},
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="grep",
        description="Search file contents for a regex pattern across the repository.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for."},
                "path_glob": {"type": "string", "description": "Optional glob to restrict which files are searched."},
            },
            "required": ["pattern"],
        },
    ),
    ToolSpec(
        name="list_symbols",
        description="List top-level functions/classes/methods and their line ranges in a single file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Repo-relative file path."}},
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="reindex",
        description=(
            "Re-run incremental indexing if search results seem stale relative to file contents. "
            "Slow — only call this when you suspect the index is out of date."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
    ),
]


def dispatch_tool(ctx: ToolContext, name: str, arguments: dict) -> str:
    handlers = {
        "search_code": tool_search_code,
        "read_file": tool_read_file,
        "grep": tool_grep,
        "list_symbols": tool_list_symbols,
        "reindex": tool_reindex,
    }
    handler = handlers.get(name)
    if handler is None:
        return f"Error: unknown tool {name!r}"
    try:
        return handler(ctx, **arguments)
    except Exception as exc:
        return f"Error: {exc}"


def tool_search_code(ctx: ToolContext, query: str, top_k: int | None = None) -> str:
    results = ctx.retriever.search(query)
    if top_k is not None:
        results = results[:top_k]
    if not results:
        return "No results found."
    return "\n\n".join(build_context_block(r) for r in results)


def _resolve_path(repo_root: Path, path: str) -> Path:
    repo_root_resolved = repo_root.resolve()
    candidate = (repo_root_resolved / path).resolve()
    if not candidate.is_relative_to(repo_root_resolved):
        raise ValueError(f"Path {path!r} is outside the indexed repository.")
    if not candidate.is_file():
        raise ValueError(f"{path!r} is not a file.")
    return candidate


def tool_read_file(
    ctx: ToolContext, path: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    target = _resolve_path(ctx.repo_root, path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start_line or 1)
    end = min(len(lines), end_line or len(lines))
    numbered = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    return f"--- {path} (lines {start}-{end}) ---\n{numbered}"


def tool_grep(ctx: ToolContext, pattern: str, path_glob: str | None = None) -> str:
    if shutil.which("rg"):
        return _grep_ripgrep(ctx.repo_root, pattern, path_glob)
    return _grep_python(ctx.repo_root, pattern, path_glob)


def _grep_ripgrep(repo_root: Path, pattern: str, path_glob: str | None) -> str:
    args = ["rg", "--line-number", "--no-heading", "--max-count", "5", pattern]
    if path_glob:
        args += ["--glob", path_glob]
    result = subprocess.run(args, cwd=repo_root, capture_output=True, text=True, timeout=10)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"rg failed: {result.stderr.strip()}")
    return result.stdout.strip() or "No matches found."


def _grep_python(repo_root: Path, pattern: str, path_glob: str | None) -> str:
    regex = re.compile(pattern)
    matches: list[str] = []
    for file_path in repo_root.glob(path_glob or "**/*"):
        if len(matches) >= 50:
            break
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{file_path.relative_to(repo_root)}:{i}:{line}")
                if len(matches) >= 50:
                    break
    return "\n".join(matches) if matches else "No matches found."


def tool_list_symbols(ctx: ToolContext, path: str) -> str:
    from gritter.indexing.chunker import chunk_file
    from gritter.utils.languages import detect_language

    target = _resolve_path(ctx.repo_root, path)
    language = detect_language(path)
    if language is None:
        return f"No language support for {path!r}; cannot extract symbols."

    content = target.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_file(
        content, path, language,
        min_tokens=ctx.config.index.chunk_min_tokens,
        max_tokens=ctx.config.index.chunk_max_tokens,
        overlap_tokens=ctx.config.index.chunk_overlap_tokens,
    )
    lines = [
        f"{c.symbol_type or 'module'}: {c.symbol_name or '(anonymous)'} (lines {c.start_line}-{c.end_line})"
        for c in chunks
    ]
    return "\n".join(lines) if lines else "No symbols found."


def tool_reindex(ctx: ToolContext) -> str:
    from gritter.indexing.pipeline import run_indexing_pipeline
    from gritter.providers.embeddings import get_embedding_provider

    embed_provider = get_embedding_provider(ctx.config.embedding.provider, ctx.config.embedding.model)
    summary = run_indexing_pipeline(ctx.repo_root, ctx.index_name, ctx.config, embed_provider)
    return f"Reindexed: {json.dumps(summary)}"
