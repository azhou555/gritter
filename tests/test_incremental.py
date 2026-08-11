from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from gritter.indexing.incremental import get_changed_files, get_current_commit


# ---------------------------------------------------------------------------
# get_current_commit
# ---------------------------------------------------------------------------

def test_get_current_commit_returns_sha(tmp_path):
    with patch("gritter.indexing.incremental.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abc1234\n"
        result = get_current_commit(tmp_path)
    assert result == "abc1234"


def test_get_current_commit_returns_none_for_non_git(tmp_path):
    with patch("gritter.indexing.incremental.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 128  # git error
        result = get_current_commit(tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# get_changed_files
# ---------------------------------------------------------------------------

def _make_diff_output(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def test_get_changed_files_modified(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.py").touch()

    diff_output = _make_diff_output("M\tsrc/foo.py")
    with patch("gritter.indexing.incremental.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = diff_output
        modified, deleted = get_changed_files(tmp_path, "abc1234")

    assert tmp_path / "src" / "foo.py" in modified
    assert deleted == []


def test_get_changed_files_deleted(tmp_path):
    diff_output = _make_diff_output("D\tsrc/gone.py")
    with patch("gritter.indexing.incremental.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = diff_output
        modified, deleted = get_changed_files(tmp_path, "abc1234")

    assert modified == []
    assert tmp_path / "src" / "gone.py" in deleted


def test_get_changed_files_added(tmp_path):
    (tmp_path / "new.py").touch()

    diff_output = _make_diff_output("A\tnew.py")
    with patch("gritter.indexing.incremental.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = diff_output
        modified, deleted = get_changed_files(tmp_path, "abc1234")

    assert tmp_path / "new.py" in modified
    assert deleted == []


def test_get_changed_files_rename(tmp_path):
    (tmp_path / "new_name.py").touch()

    diff_output = _make_diff_output("R100\told_name.py\tnew_name.py")
    with patch("gritter.indexing.incremental.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = diff_output
        modified, deleted = get_changed_files(tmp_path, "abc1234")

    assert tmp_path / "new_name.py" in modified
    assert tmp_path / "old_name.py" in deleted


def test_get_changed_files_returns_empty_on_git_error(tmp_path):
    with patch("gritter.indexing.incremental.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 128
        modified, deleted = get_changed_files(tmp_path, "abc1234")

    assert modified == []
    assert deleted == []


def test_get_changed_files_multiple_statuses(tmp_path):
    (tmp_path / "modified.py").touch()
    (tmp_path / "added.rs").touch()

    diff_output = _make_diff_output(
        "M\tmodified.py",
        "D\tdeleted.ts",
        "A\tadded.rs",
    )
    with patch("gritter.indexing.incremental.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = diff_output
        modified, deleted = get_changed_files(tmp_path, "abc1234")

    assert tmp_path / "modified.py" in modified
    assert tmp_path / "added.rs" in modified
    assert tmp_path / "deleted.ts" in deleted


# ---------------------------------------------------------------------------
# IndexMeta
# ---------------------------------------------------------------------------

def test_index_meta_persists_source_root(tmp_path):
    from gritter.storage.index_meta import IndexMeta

    index_dir = tmp_path / "index"
    meta = IndexMeta(index_dir)
    meta.write(
        embedding_provider="voyage",
        embedding_model="voyage-code-3",
        embedding_dimension=1024,
        file_count=1,
        chunk_count=1,
        languages=["python"],
        source_root="/repos/example",
    )
    reloaded = IndexMeta(index_dir).read()
    assert reloaded["source_root"] == "/repos/example"
