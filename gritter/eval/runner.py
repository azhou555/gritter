from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from gritter.eval.metrics import mrr, recall_at_k
from gritter.retrieval.hybrid import HybridRetriever


@dataclass
class QueryResult:
    query: str
    relevant_files: list[str]
    retrieved_files: list[str]
    recall: float
    mrr: float


@dataclass
class EvalResult:
    query_results: list[QueryResult] = field(default_factory=list)

    @property
    def mean_recall(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(r.recall for r in self.query_results) / len(self.query_results)

    @property
    def mean_mrr(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(r.mrr for r in self.query_results) / len(self.query_results)


def load_dataset(path: Path) -> list[dict]:
    """Load a JSONL eval dataset.

    Each line: {"query": "...", "relevant_files": ["path/to/file.py", ...]}
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_eval(dataset: list[dict], retriever: HybridRetriever, k: int = 5) -> EvalResult:
    """Run retrieval for each query and compute recall@k and MRR."""
    result = EvalResult()

    for record in dataset:
        query: str = record["query"]
        relevant: list[str] = record["relevant_files"]

        results = retriever.search(query)
        retrieved = [r.chunk.file_path for r in results]

        qr = QueryResult(
            query=query,
            relevant_files=relevant,
            retrieved_files=retrieved,
            recall=recall_at_k(relevant, retrieved, k),
            mrr=mrr(relevant, retrieved),
        )
        result.query_results.append(qr)

    return result
