from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gritter.eval.metrics import mrr, recall_at_k
from gritter.eval.runner import EvalResult, QueryResult, load_dataset, run_eval
from gritter.models.chunk import CodeChunk
from gritter.models.query import RetrievalResult


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def test_recall_at_k_perfect():
    assert recall_at_k(["a.py", "b.py"], ["a.py", "b.py", "c.py"], k=5) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a.py", "b.py"], ["a.py", "c.py", "d.py"], k=3) == 0.5


def test_recall_at_k_miss():
    assert recall_at_k(["a.py"], ["b.py", "c.py"], k=5) == 0.0


def test_recall_at_k_respects_cutoff():
    # relevant file is at position 3, but k=2
    assert recall_at_k(["a.py"], ["b.py", "c.py", "a.py"], k=2) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k([], ["a.py"], k=5) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------

def test_mrr_first_hit():
    assert mrr(["a.py"], ["a.py", "b.py"]) == 1.0


def test_mrr_second_hit():
    assert abs(mrr(["b.py"], ["a.py", "b.py"]) - 0.5) < 1e-9


def test_mrr_no_hit():
    assert mrr(["a.py"], ["b.py", "c.py"]) == 0.0


def test_mrr_uses_first_relevant():
    # b.py is at rank 2, a.py at rank 3 — MRR should be 1/2
    assert abs(mrr(["a.py", "b.py"], ["c.py", "b.py", "a.py"]) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

def test_load_dataset(tmp_path):
    data = [
        {"query": "how does foo work?", "relevant_files": ["foo.py"]},
        {"query": "what is bar?", "relevant_files": ["bar.py", "baz.py"]},
    ]
    ds_file = tmp_path / "eval.jsonl"
    ds_file.write_text("\n".join(json.dumps(d) for d in data))

    records = load_dataset(ds_file)
    assert len(records) == 2
    assert records[0]["query"] == "how does foo work?"
    assert records[1]["relevant_files"] == ["bar.py", "baz.py"]


def test_load_dataset_skips_blank_lines(tmp_path):
    ds_file = tmp_path / "eval.jsonl"
    ds_file.write_text('{"query": "q", "relevant_files": ["a.py"]}\n\n')
    records = load_dataset(ds_file)
    assert len(records) == 1


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------

def _make_retrieval_result(file_path: str) -> RetrievalResult:
    chunk = CodeChunk(
        content="some code",
        file_path=file_path,
        language="python",
        symbol_name=None,
        symbol_type=None,
        start_line=1,
        end_line=5,
    )
    return RetrievalResult(chunk=chunk, score=0.9)


def test_run_eval_computes_metrics():
    dataset = [
        {"query": "find foo", "relevant_files": ["foo.py"]},
        {"query": "find bar", "relevant_files": ["bar.py"]},
    ]
    mock_retriever = MagicMock()
    mock_retriever.search.side_effect = [
        [_make_retrieval_result("foo.py")],   # correct hit
        [_make_retrieval_result("wrong.py")], # miss
    ]

    result = run_eval(dataset, mock_retriever, k=5)

    assert len(result.query_results) == 2
    assert result.query_results[0].recall == 1.0
    assert result.query_results[1].recall == 0.0
    assert abs(result.mean_recall - 0.5) < 1e-9


def test_eval_result_mean_mrr():
    qr1 = QueryResult("q1", ["a.py"], ["a.py"], recall=1.0, mrr=1.0)
    qr2 = QueryResult("q2", ["b.py"], ["c.py", "b.py"], recall=1.0, mrr=0.5)
    result = EvalResult(query_results=[qr1, qr2])
    assert abs(result.mean_mrr - 0.75) < 1e-9


def test_eval_result_empty():
    result = EvalResult()
    assert result.mean_recall == 0.0
    assert result.mean_mrr == 0.0
