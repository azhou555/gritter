from __future__ import annotations
import typer
from gritter.models.config import GritterConfig
from gritter.storage.index_meta import IndexMeta
from gritter.utils.display import console, make_table, print_error


def status(
    name: str = typer.Argument("default", help="Index name to inspect"),
) -> None:
    """Show statistics for a stored index."""
    config = GritterConfig()
    index_dir = config.index_dir(name)

    try:
        meta = IndexMeta(index_dir)
        data = meta.read()
    except FileNotFoundError:
        print_error(
            f"No index named [bold]{name}[/bold] found. "
            f"Run `gritter index <path>` first."
        )
        raise typer.Exit(1)

    rows = [
        ("Index name", name),
        ("Files", str(data["file_count"])),
        ("Chunks", str(data["chunk_count"])),
        ("Languages", ", ".join(data["languages"])),
        ("Embedding provider", data["embedding_provider"]),
        ("Embedding model", data["embedding_model"]),
        ("Indexed at", data["indexed_at"]),
    ]
    console.print(make_table(f"Index: {name}", rows))
