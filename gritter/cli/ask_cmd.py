from __future__ import annotations

from pathlib import Path
import typer
from rich.live import Live
from rich.text import Text

from gritter.agent.session import AgentSession, ToolFinished, ToolStarted
from gritter.agent.tools import ToolContext
from gritter.generation.citations import format_sources
from gritter.models.config import GritterConfig
from gritter.providers.llm import TextDelta, get_llm_provider
from gritter.retrieval.hybrid import HybridRetriever
from gritter.storage.index_meta import IndexMeta
from gritter.utils.display import console, print_error


def ask(
    question: str = typer.Argument(..., help="Natural language question about the codebase"),
    name: str = typer.Option(None, "--name", "-n", help="Index name to query (defaults to current directory name)"),
    top_k: int = typer.Option(None, "--top-k", help="Number of results to retrieve"),
) -> None:
    """Ask a question about an indexed codebase."""
    config = GritterConfig()
    index_name = name or Path.cwd().name
    if top_k is not None:
        config.retrieval.top_k = top_k

    try:
        retriever = HybridRetriever.from_config(index_name, config)
        meta = IndexMeta(config.index_dir(index_name)).read()
    except (ValueError, FileNotFoundError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    source_root = meta.get("source_root")
    if source_root is None:
        print_error(
            f"Index '{index_name}' was built with an older version of gritter and has no "
            "recorded source path. Re-run `gritter index` to rebuild it."
        )
        raise typer.Exit(1)

    try:
        llm = get_llm_provider(config.llm.provider, config.llm.model)
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    ctx = ToolContext(
        repo_root=Path(source_root), retriever=retriever, config=config, index_name=index_name,
    )
    session = AgentSession(llm, ctx)

    response_text = Text()
    console.print()
    with Live(response_text, console=console, refresh_per_second=15):
        for event in session.run(question):
            if isinstance(event, TextDelta):
                response_text.append(event.text)
            elif isinstance(event, ToolStarted):
                console.print(f"[dim]→ {event.name}({event.arguments})[/dim]")
    console.print()

    sources = format_sources(session.last_citations)
    if sources:
        console.print(f"\n[dim]{sources}[/dim]")
