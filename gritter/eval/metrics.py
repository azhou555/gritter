from __future__ import annotations


def recall_at_k(relevant_files: list[str], retrieved_files: list[str], k: int) -> float:
    """Fraction of relevant files found in the top-k retrieved files.

    retrieved_files should be ordered by rank (best first).
    Returns 0.0 if relevant_files is empty.
    """
    if not relevant_files:
        return 0.0
    top_k = set(retrieved_files[:k])
    hits = sum(1 for f in relevant_files if f in top_k)
    return hits / len(relevant_files)


def mrr(relevant_files: list[str], retrieved_files: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant file in retrieved_files.

    Returns 0.0 if no relevant file appears in retrieved_files.
    """
    relevant_set = set(relevant_files)
    for rank, file_path in enumerate(retrieved_files, start=1):
        if file_path in relevant_set:
            return 1.0 / rank
    return 0.0
