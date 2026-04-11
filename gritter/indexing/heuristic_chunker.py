from __future__ import annotations
from gritter.models.chunk import CodeChunk
from gritter.utils.tokens import count_tokens, split_at_token_boundary


def chunk_file_heuristic(
    content: str,
    file_path: str,
    language: str,
    min_tokens: int = 50,
    max_tokens: int = 512,
    overlap_tokens: int = 20,
) -> list[CodeChunk]:
    """Split content into chunks at blank lines where the next line has zero indentation."""
    raw_blocks = _split_top_level_blocks(content)
    if not raw_blocks:
        return []

    # Track line numbers for each block
    line_cursor = 1
    block_with_lines: list[tuple[str, int, int]] = []
    for block in raw_blocks:
        block_line_count = block.count("\n") + 1
        end_line = line_cursor + block_line_count - 1
        block_with_lines.append((block, line_cursor, end_line))
        line_cursor = end_line + 2  # account for blank separator

    # Apply token guardrails
    return _apply_guardrails(block_with_lines, file_path, language, min_tokens, max_tokens, overlap_tokens)


def _split_top_level_blocks(content: str) -> list[str]:
    """Split at blank lines where the next non-blank line has zero indentation."""
    lines = content.split("\n")
    blocks: list[str] = []
    current: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            # Look ahead to find next non-blank line
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1

            if j >= len(lines):
                # Trailing blank lines — end current block
                current.append(line)
                i += 1
            elif lines[j] and lines[j][0] not in (" ", "\t"):
                # Next content is at indent level 0 — this is a block boundary
                block = "\n".join(current).rstrip()
                if block.strip():
                    blocks.append(block)
                current = []
                i = j  # skip the blank lines
            else:
                # Next content is indented — we're inside a block, stay
                current.extend(lines[i:j])
                i = j
        else:
            current.append(line)
            i += 1

    if current:
        block = "\n".join(current).rstrip()
        if block.strip():
            blocks.append(block)

    return blocks


def _apply_guardrails(
    block_with_lines: list[tuple[str, int, int]],
    file_path: str,
    language: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[CodeChunk]:
    """Merge small blocks and split large ones, then create CodeChunk objects."""
    # First pass: split blocks that exceed max_tokens
    expanded: list[tuple[str, int, int]] = []
    for content, start, end in block_with_lines:
        if count_tokens(content) > max_tokens:
            parts = split_at_token_boundary(content, max_tokens, overlap_tokens)
            for part in parts:
                expanded.append((part, start, end))  # approximate line numbers for splits
        else:
            expanded.append((content, start, end))

    # Second pass: merge consecutive small blocks
    merged: list[tuple[str, int, int]] = []
    for content, start, end in expanded:
        if merged and count_tokens(merged[-1][0]) < min_tokens:
            prev_content, prev_start, _ = merged[-1]
            merged[-1] = (prev_content + "\n\n" + content, prev_start, end)
        else:
            merged.append((content, start, end))

    return [
        CodeChunk(
            content=content,
            file_path=file_path,
            language=language,
            symbol_name=None,
            symbol_type="module",
            start_line=start,
            end_line=end,
        )
        for content, start, end in merged
        if content.strip()
    ]
