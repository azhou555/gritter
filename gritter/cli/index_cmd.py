from __future__ import annotations

from pathlib import Path

import typer

from gritter.indexing.github import is_github_url, ensure_repo, parse_github_url
from gritter.models.config import GritterConfig
from gritter.providers.embeddings import get_embedding_provider
from gritter.indexing.pipeline import run_indexing_pipeline
from gritter.utils.display import console, print_success, print_error


def index(
    path: str = typer.Argument(..., help="Path to a local directory or GitHub repo URL"),
    name: str = typer.Option(None, "--name", "-n", help="Index name (defaults to repo/directory name)"),
    branch: str = typer.Option(None, "--branch", "-b", help="Branch to index (GitHub repos only)"),
    full: bool = typer.Option(False, "--full", help="Force full re-index, ignoring incremental state"),
) -> None:
    """Index a codebase for querying."""
    config = GritterConfig()

    if is_github_url(path):
        repos_dir = config.data_dir / "repos"
        console.print(f"\n[bold]Fetching[/bold] [cyan]{path}[/cyan]")
        try:
            root = ensure_repo(path, branch, repos_dir)
        except RuntimeError as exc:
            print_error(str(exc))
            raise typer.Exit(1)
        _, repo = parse_github_url(path)
        index_name = name or repo
    else:
        root = Path(path).resolve()
        if not root.is_dir():
            print_error(f"{path} is not a directory.")
            raise typer.Exit(1)
        if branch:
            print_error("--branch is only valid for GitHub repo URLs.")
            raise typer.Exit(1)
        index_name = name or root.name

    try:
        embed_provider = get_embedding_provider(
            config.embedding.provider, config.embedding.model
        )
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    console.print(f"\n[bold]Indexing[/bold] [cyan]{root}[/cyan] as [bold]{index_name}[/bold]\n")

    try:
        summary = run_indexing_pipeline(root, index_name, config, embed_provider, full=full)
    except Exception as exc:
        print_error(f"Indexing failed: {exc}")
        raise typer.Exit(1)

    if summary.get("incremental"):
        print_success(
            f"Index [bold]{index_name}[/bold] updated — "
            f"{summary['modified']} files re-indexed, {summary['deleted']} removed, "
            f"{summary['chunk_count']} chunks total"
        )
    else:
        print_success(
            f"Index [bold]{index_name}[/bold] ready — "
            f"{summary['file_count']} files, {summary['chunk_count']} chunks, "
            f"languages: {', '.join(summary['languages'])}"
        )
