from __future__ import annotations

from pathlib import Path
from unittest.mock import call, patch

import pytest

from gritter.indexing.github import (
    ensure_repo,
    is_github_url,
    parse_github_url,
)


# ---------------------------------------------------------------------------
# is_github_url
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo",
    "https://github.com/owner/repo.git",
    "http://github.com/owner/repo",
    "github.com/owner/repo",
    "github.com/owner/repo.git",
    "https://github.com/owner/repo/",
])
def test_is_github_url_matches(url):
    assert is_github_url(url)


@pytest.mark.parametrize("s", [
    "owner/repo",                        # ambiguous bare shorthand — not matched
    "/Users/azhou/projects/myrepo",      # local path
    ".",
    "https://gitlab.com/owner/repo",
    "https://github.com/owner",          # missing repo segment
    "",
])
def test_is_github_url_rejects(s):
    assert not is_github_url(s)


# ---------------------------------------------------------------------------
# parse_github_url
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://github.com/owner/repo", ("owner", "repo")),
    ("https://github.com/owner/repo.git", ("owner", "repo")),
    ("github.com/octocat/Hello-World", ("octocat", "Hello-World")),
    ("http://github.com/org/proj/", ("org", "proj")),
])
def test_parse_github_url(url, expected):
    assert parse_github_url(url) == expected


def test_parse_github_url_invalid():
    with pytest.raises(ValueError, match="Not a recognised GitHub URL"):
        parse_github_url("owner/repo")


# ---------------------------------------------------------------------------
# ensure_repo
# ---------------------------------------------------------------------------

def test_ensure_repo_clones_when_not_cached(tmp_path):
    cache_dir = tmp_path / "repos"
    clone_target = cache_dir / "owner__repo"

    with patch("gritter.indexing.github._run") as mock_run:
        result = ensure_repo("https://github.com/owner/repo", None, cache_dir)

    assert result == clone_target
    mock_run.assert_called_once_with(
        ["git", "clone", "--filter=blob:none",
         "https://github.com/owner/repo", str(clone_target)]
    )


def test_ensure_repo_clones_with_branch(tmp_path):
    cache_dir = tmp_path / "repos"
    clone_target = cache_dir / "owner__repo"

    with patch("gritter.indexing.github._run") as mock_run:
        ensure_repo("https://github.com/owner/repo", "dev", cache_dir)

    mock_run.assert_called_once_with(
        ["git", "clone", "--filter=blob:none",
         "https://github.com/owner/repo", str(clone_target),
         "--branch", "dev"]
    )


def test_ensure_repo_fetches_when_cached(tmp_path):
    cache_dir = tmp_path / "repos"
    clone_target = cache_dir / "owner__repo"
    clone_target.mkdir(parents=True)  # simulate existing clone

    with patch("gritter.indexing.github._run") as mock_run:
        result = ensure_repo("https://github.com/owner/repo", None, cache_dir)

    assert result == clone_target
    assert mock_run.call_count == 2
    assert mock_run.call_args_list[0] == call(
        ["git", "-C", str(clone_target), "fetch", "origin"]
    )
    assert mock_run.call_args_list[1] == call(
        ["git", "-C", str(clone_target), "reset", "--hard", "origin/HEAD"]
    )


def test_ensure_repo_fetches_with_branch_when_cached(tmp_path):
    cache_dir = tmp_path / "repos"
    clone_target = cache_dir / "owner__repo"
    clone_target.mkdir(parents=True)

    with patch("gritter.indexing.github._run") as mock_run:
        ensure_repo("https://github.com/owner/repo", "main", cache_dir)

    assert mock_run.call_args_list[1] == call(
        ["git", "-C", str(clone_target), "reset", "--hard", "origin/main"]
    )


def test_ensure_repo_raises_on_git_failure(tmp_path):
    cache_dir = tmp_path / "repos"

    with patch("gritter.indexing.github._run", side_effect=RuntimeError("clone failed")):
        with pytest.raises(RuntimeError, match="clone failed"):
            ensure_repo("https://github.com/owner/repo", None, cache_dir)
