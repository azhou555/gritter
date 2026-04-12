from __future__ import annotations
from gritter.models.query import RetrievalResult


def reciprocal_rank_fusion(
    dense: list[RetrievalResult],
    sparse: list[RetrievalResult],
    k: int = 60,
) -> list[RetrievalResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion (Cormack et al., 2009).

    RRF_score(chunk) = Σ 1 / (k + rank_in_list)
    summed across all lists the chunk appears in.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(dense):
        cid = result.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunks[cid] = result

    for rank, result in enumerate(sparse):
        cid = result.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunks[cid] = result

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        RetrievalResult(chunk=chunks[cid].chunk, score=scores[cid])
        for cid in sorted_ids
    ]
