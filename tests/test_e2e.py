"""
End-to-end smoke tests.

Gated behind the GRITTER_RUN_E2E_TESTS=1 environment variable.
These tests make real API calls and require a fully configured environment:
  - ANTHROPIC_API_KEY or OPENAI_API_KEY
  - VOYAGE_API_KEY (if using voyage embeddings)

Run with:
    GRITTER_RUN_E2E_TESTS=1 pytest tests/test_e2e.py -v
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Skip entire module unless the gate env var is set
pytestmark = pytest.mark.skipif(
    not os.environ.get("GRITTER_RUN_E2E_TESTS"),
    reason="Set GRITTER_RUN_E2E_TESTS=1 to run e2e tests",
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "python_project"


@pytest.fixture(scope="module")
def indexed_dir(tmp_path_factory):
    """Index the python fixture project into a temp data dir and return (config, index_name)."""
    from gritter.models.config import GritterConfig
    from gritter.providers.embeddings import get_embedding_provider
    from gritter.indexing.pipeline import run_indexing_pipeline

    data_dir = tmp_path_factory.mktemp("gritter_e2e")
    config = GritterConfig(data_dir=data_dir)
    index_name = "e2e_fixture"

    embed_provider = get_embedding_provider(config.embedding.provider, config.embedding.model)
    summary = run_indexing_pipeline(FIXTURE_ROOT, index_name, config, embed_provider)

    assert summary["file_count"] > 0, "No files indexed"
    assert summary["chunk_count"] > 0, "No chunks indexed"

    return config, index_name, summary


def test_index_produces_nonzero_counts(indexed_dir):
    """Indexing the fixture project should produce files and chunks."""
    _, _, summary = indexed_dir
    assert summary["file_count"] >= 1
    assert summary["chunk_count"] >= 1
    assert "python" in summary["languages"]


def test_status_reads_index_metadata(indexed_dir):
    """IndexMeta should be readable after indexing."""
    from gritter.storage.index_meta import IndexMeta

    config, index_name, summary = indexed_dir
    meta = IndexMeta(config.index_dir(index_name))
    data = meta.read()

    assert data["chunk_count"] == summary["chunk_count"]
    assert data["file_count"] == summary["file_count"]
    assert "embedding_provider" in data
    assert "embedding_dimension" in data


def test_retrieval_returns_results(indexed_dir):
    """Hybrid retrieval should return at least one result for a relevant query."""
    from gritter.retrieval.hybrid import HybridRetriever

    config, index_name, _ = indexed_dir
    retriever = HybridRetriever.from_config(index_name, config)
    results = retriever.search("How does the add function work?")

    assert len(results) > 0, "Expected at least one retrieval result"
    assert results[0].score > 0
    assert results[0].chunk.file_path != ""


def test_agent_session_returns_nonempty_response_with_citation(indexed_dir):
    """AgentSession should produce a non-empty response with at least one verified citation."""
    from gritter.agent.session import AgentSession
    from gritter.agent.tools import ToolContext
    from gritter.models.config import GritterConfig
    from gritter.providers.llm import TextDelta, get_llm_provider
    from gritter.retrieval.hybrid import HybridRetriever
    from gritter.storage.index_meta import IndexMeta

    config, index_name, summary = indexed_dir
    retriever = HybridRetriever.from_config(index_name, config)
    source_root = IndexMeta(config.index_dir(index_name)).read()["source_root"]

    llm = get_llm_provider(config.llm.provider, config.llm.model)
    ctx = ToolContext(repo_root=Path(source_root), retriever=retriever, config=config, index_name=index_name)
    session = AgentSession(llm, ctx)

    events = list(session.run("What does the add function do?"))
    response = "".join(e.text for e in events if isinstance(e, TextDelta))

    assert len(response) > 0, "Expected non-empty response"
    assert len(session.messages) >= 2
    assert len(session.last_citations) >= 1, f"Expected at least one citation in: {response[:300]}"
