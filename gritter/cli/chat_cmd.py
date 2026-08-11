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


def chat(
    name: str = typer.Option(None, "--name", "-n", help="Index name to query (defaults to current directory name)"),
) -> None:
    """Interactive multi-turn chat about an indexed codebase."""
    config = GritterConfig()
    index_name = name or Path.cwd().name

    try:
        retriever = HybridRetriever.from_config(index_name, config)
        source_root = IndexMeta(config.index_dir(index_name)).read()["source_root"]
    except (ValueError, FileNotFoundError) as exc:
        print_error(str(exc))
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

    console.print(
        f"\n[bold]Gritter chat[/bold] — index: [cyan]{index_name}[/cyan]  "
        "[dim](type 'exit' or press Ctrl+C to quit)[/dim]\n"
    )

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        response_text = Text()
        console.print("\n[bold]Gritter:[/bold] ", end="")
        with Live(response_text, console=console, refresh_per_second=15):
            for event in session.run(query):
                if isinstance(event, TextDelta):
                    response_text.append(event.text)
                elif isinstance(event, ToolStarted):
                    console.print(f"[dim]→ {event.name}({event.arguments})[/dim]")
        console.print()

        sources = format_sources(session.last_citations)
        if sources:
            console.print(f"[dim]{sources}[/dim]")

        console.print()
