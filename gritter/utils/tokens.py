from __future__ import annotations
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (GPT-4 tokenizer — good approximation for all models)."""
    return len(_enc.encode(text))


def split_at_token_boundary(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into chunks of at most max_tokens with overlap_tokens overlap."""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap_tokens
    return chunks
