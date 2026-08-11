from __future__ import annotations

from gritter.models.query import RetrievalResult


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
