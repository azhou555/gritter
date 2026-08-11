from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer

from gritter.models.config import GritterConfig
from gritter.storage.index_meta import IndexMeta


def test_ask_cmd_handles_missing_source_root_gracefully(tmp_path):
    """Verify ask_cmd exits cleanly with friendly error when source_root is missing."""
    from gritter.cli.ask_cmd import ask

    # Create an old-style index metadata (missing source_root)
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    meta_file = index_dir / "meta.json"
    old_style_meta = {
        "embedding_provider": "voyage",
        "embedding_model": "voyage-code-3",
        "embedding_dimension": 1024,
        "file_count": 1,
        "chunk_count": 1,
        "languages": ["python"],
        "indexed_at": "2025-01-01T00:00:00+00:00",
        "indexed_commit": None,
    }
    meta_file.write_text(json.dumps(old_style_meta))

    # Mock GritterConfig with all required attributes
    mock_config = Mock(spec=GritterConfig)
    mock_config.index_dir.return_value = index_dir
    mock_config.retrieval = Mock()

    # Mock HybridRetriever to not raise (we're testing the source_root check, not retriever init)
    with patch("gritter.cli.ask_cmd.GritterConfig", return_value=mock_config), \
         patch("gritter.cli.ask_cmd.HybridRetriever.from_config", return_value=Mock()), \
         patch("gritter.cli.ask_cmd.Path.cwd") as mock_cwd:

        fake_cwd = Mock()
        fake_cwd.name = "test_index"
        mock_cwd.return_value = fake_cwd

        # Attempting to call ask should exit with code 1 due to missing source_root
        with pytest.raises(typer.Exit) as exc_info:
            ask(question="test question")

        assert exc_info.value.exit_code == 1


def test_chat_cmd_handles_missing_source_root_gracefully(tmp_path):
    """Verify chat_cmd exits cleanly with friendly error when source_root is missing."""
    from gritter.cli.chat_cmd import chat

    # Create an old-style index metadata (missing source_root)
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    meta_file = index_dir / "meta.json"
    old_style_meta = {
        "embedding_provider": "voyage",
        "embedding_model": "voyage-code-3",
        "embedding_dimension": 1024,
        "file_count": 1,
        "chunk_count": 1,
        "languages": ["python"],
        "indexed_at": "2025-01-01T00:00:00+00:00",
        "indexed_commit": None,
    }
    meta_file.write_text(json.dumps(old_style_meta))

    # Mock GritterConfig with all required attributes
    mock_config = Mock(spec=GritterConfig)
    mock_config.index_dir.return_value = index_dir

    # Mock HybridRetriever to not raise (we're testing the source_root check, not retriever init)
    with patch("gritter.cli.chat_cmd.GritterConfig", return_value=mock_config), \
         patch("gritter.cli.chat_cmd.HybridRetriever.from_config", return_value=Mock()), \
         patch("gritter.cli.chat_cmd.Path.cwd") as mock_cwd:

        fake_cwd = Mock()
        fake_cwd.name = "test_index"
        mock_cwd.return_value = fake_cwd

        # Attempting to call chat should exit with code 1 due to missing source_root
        with pytest.raises(typer.Exit) as exc_info:
            chat()

        assert exc_info.value.exit_code == 1


def test_index_meta_read_old_style_missing_source_root(tmp_path):
    """Verify that reading old-style metadata without source_root returns None via .get()."""
    index_dir = tmp_path / "old_index"
    index_dir.mkdir()
    meta_file = index_dir / "meta.json"

    # Write old-style metadata (no source_root key)
    old_style = {
        "embedding_provider": "voyage",
        "embedding_model": "voyage-code-3",
        "embedding_dimension": 1024,
        "file_count": 5,
        "chunk_count": 50,
        "languages": ["python", "javascript"],
        "indexed_at": "2025-01-01T00:00:00+00:00",
        "indexed_commit": None,
    }
    meta_file.write_text(json.dumps(old_style))

    # Read the metadata and verify source_root is missing
    meta = IndexMeta(index_dir)
    data = meta.read()

    # Verify the key doesn't exist
    assert "source_root" not in data

    # Verify .get() returns None
    assert data.get("source_root") is None
