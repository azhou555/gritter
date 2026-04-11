from __future__ import annotations
import logging
from pathlib import Path

from gritter.models.chunk import CodeChunk
from gritter.indexing.ast_chunker import chunk_file_ast
from gritter.indexing.heuristic_chunker import chunk_file_heuristic

logger = logging.getLogger(__name__)

_AST_LANGUAGES = {"python", "typescript", "rust"}


def chunk_file(
    content: str,
    file_path: str,
    language: str,
    min_tokens: int = 50,
    max_tokens: int = 512,
    overlap_tokens: int = 20,
) -> list[CodeChunk]:
    """Dispatch to AST or heuristic chunker based on language support.

    Falls back to heuristic if AST parsing raises (syntax errors, grammar issues).
    """
    if language in _AST_LANGUAGES:
        try:
            return chunk_file_ast(
                content, file_path, language, min_tokens, max_tokens, overlap_tokens
            )
        except Exception as exc:
            logger.warning(
                "AST chunking failed for %s (%s), falling back to heuristic: %s",
                file_path, language, exc,
            )

    return chunk_file_heuristic(
        content, file_path, language, min_tokens, max_tokens, overlap_tokens
    )
