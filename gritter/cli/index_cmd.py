from __future__ import annotations
from pathlib import Path
import typer
from gritter.models.config import GritterConfig
from gritter.providers.embeddings import get_embedding_provider
from gritter.indexing.pipeline import run_indexing_pipeline
from gritter.utils.display import console, print_success, print_error


def index(
    path: Path = typer.Argument(..., help="Path to the codebase to index"),
    name: str = typer.Option(None, "--name", "-n", help="Index name (defaults to directory name)"),
) -> None:
    """Index a codebase for querying."""
    root = path.resolve()
    if not root.is_dir():
        print_error(f"{path} is not a directory.")
        raise typer.Exit(1)

    config = GritterConfig()
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
        summary = run_indexing_pipeline(root, index_name, config, embed_provider)
    except Exception as exc:
        print_error(f"Indexing failed: {exc}")
        raise typer.Exit(1)

    print_success(
        f"Index [bold]{index_name}[/bold] ready — "
        f"{summary['file_count']} files, {summary['chunk_count']} chunks, "
        f"languages: {', '.join(summary['languages'])}"
    )
