from __future__ import annotations

from gritter.models.query import RetrievalResult

SYSTEM_PROMPT: str = """\
You are a code assistant that answers questions strictly from the provided context.

Rules:
1. Answer only using information present in the provided context blocks. Do not use prior \
knowledge or assumptions.
2. Cite every claim using the exact format: path/to/file.py:L15-45 (using the start and end \
lines from the context header). If a claim comes from a single line, use path/to/file.py:L15.
3. If the answer is not present in the provided context, respond with exactly: \
"I don't know from the provided context." Do not guess or hallucinate file paths or line numbers.\
"""


def build_context_block(result: RetrievalResult) -> str:
    chunk = result.chunk
    symbol_part = ""
    if chunk.symbol_name is not None:
        type_prefix = f"{chunk.symbol_type}: " if chunk.symbol_type is not None else ""
        symbol_part = f", {type_prefix}{chunk.symbol_name}"
    header = (
        f"--- File: {chunk.file_path} "
        f"(lines {chunk.start_line}-{chunk.end_line}{symbol_part}) ---"
    )
    return f"{header}\n{chunk.content}"


def build_user_prompt(query: str, results: list[RetrievalResult]) -> str:
    blocks = "\n\n".join(build_context_block(r) for r in results)
    return f"Context:\n{blocks}\n\nQuestion: {query}"
