# tests/test_retrieval.py
from unittest.mock import MagicMock, patch

import numpy as np

from gritter.models.chunk import CodeChunk
from gritter.models.query import RetrievalResult
from gritter.retrieval.fusion import reciprocal_rank_fusion
from gritter.retrieval.reranker import CrossEncoderReranker, NoOpReranker


def make_chunk(chunk_id_suffix: str, content: str = "some code", **kwargs) -> CodeChunk:
    chunk = CodeChunk(
        content=content,
        file_path="src/foo.py",
        language="python",
        symbol_name=None,
        symbol_type="function",
        start_line=1,
        end_line=5,
    )
    # Override chunk_id for predictable test IDs
    object.__setattr__(chunk, "chunk_id", f"chunk_{chunk_id_suffix}")
    return chunk


def make_result(chunk_id_suffix: str, score: float, content: str = "some code") -> RetrievalResult:
    return RetrievalResult(chunk=make_chunk(chunk_id_suffix, content=content), score=score)


class TestRRF:
    def test_chunk_in_both_lists_scores_higher(self):
        # chunk_0 is rank 0 in both lists — should have highest RRF score
        dense = [make_result(str(i), 1.0 / (i + 1)) for i in range(5)]
        sparse = [make_result(str(i), 1.0 / (i + 1)) for i in range(5)]
        fused = reciprocal_rank_fusion(dense, sparse)
        assert fused[0].chunk.chunk_id == "chunk_0"

    def test_chunk_only_in_one_list_still_included(self):
        dense = [make_result("dense_only", 1.0)]
        sparse = [make_result("sparse_only", 1.0)]
        fused = reciprocal_rank_fusion(dense, sparse)
        ids = {r.chunk.chunk_id for r in fused}
        assert "chunk_dense_only" in ids
        assert "chunk_sparse_only" in ids

    def test_rrf_score_formula(self):
        k = 60
        # chunk_1 is rank 1 in dense, rank 0 in sparse
        dense = [make_result("0", 1.0), make_result("1", 0.5)]
        sparse = [make_result("1", 1.0)]
        fused = reciprocal_rank_fusion(dense, sparse, k=k)
        scores = {r.chunk.chunk_id: r.score for r in fused}
        # chunk_1: 1/(60+1) in dense + 1/(60+0) in sparse
        expected = 1.0 / (k + 1) + 1.0 / (k + 0)
        assert abs(scores["chunk_1"] - expected) < 1e-9

    def test_results_sorted_descending(self):
        dense = [make_result(f"d{i}", 1.0) for i in range(10)]
        sparse = [make_result(f"s{i}", 1.0) for i in range(10)]
        fused = reciprocal_rank_fusion(dense, sparse)
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)


class TestNoOpReranker:
    def test_returns_results_unchanged(self):
        results = [make_result(str(i), float(i)) for i in range(5)]
        reranker = NoOpReranker()
        reranked = reranker.rerank("some query", results)
        assert reranked == results


class TestCrossEncoderReranker:
    def _make_reranker_with_mock(self, scores: list[float]) -> CrossEncoderReranker:
        """Return a CrossEncoderReranker whose underlying model is mocked."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(scores)
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = mock_model
        return reranker

    def test_reranks_by_score_descending(self):
        # Three results — mock model scores them in reverse order
        results = [make_result("a", 0.9), make_result("b", 0.8), make_result("c", 0.7)]
        reranker = self._make_reranker_with_mock([0.1, 0.5, 0.9])

        reranked = reranker.rerank("query", results)

        # Highest model score (0.9 → chunk "c") should be first
        assert reranked[0].chunk.chunk_id == "chunk_c"
        assert reranked[1].chunk.chunk_id == "chunk_b"
        assert reranked[2].chunk.chunk_id == "chunk_a"

    def test_scores_updated_to_cross_encoder_scores(self):
        results = [make_result("x", 1.0), make_result("y", 0.5)]
        reranker = self._make_reranker_with_mock([0.3, 0.8])

        reranked = reranker.rerank("query", results)

        assert abs(reranked[0].score - 0.8) < 1e-6  # chunk_y is first
        assert abs(reranked[1].score - 0.3) < 1e-6

    def test_empty_results_returned_unchanged(self):
        reranker = self._make_reranker_with_mock([])
        assert reranker.rerank("query", []) == []

    def test_predict_called_with_query_content_pairs(self):
        results = [make_result("a", 1.0, content="def foo(): pass")]
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.7])

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = mock_model

        reranker.rerank("how does foo work?", results)

        mock_model.predict.assert_called_once_with(
            [("how does foo work?", "def foo(): pass")]
        )
