from __future__ import annotations

from pathlib import Path
import typer
from rich.live import Live
from rich.text import Text

from gritter.generation.citations import extract_citations, format_sources
from gritter.generation.generator import Generator
from gritter.models.config import GritterConfig
from gritter.providers.embeddings import get_embedding_provider
from gritter.providers.llm import get_llm_provider
from gritter.retrieval.hybrid import HybridRetriever
from gritter.utils.display import console, print_error


def chat(
    name: str = typer.Option(None, "--name", "-n", help="Index name to query (defaults to current directory name)"),
) -> None:
    """Interactive multi-turn chat about an indexed codebase."""
    config = GritterConfig()
    index_name = name or Path.cwd().name

    try:
        retriever = HybridRetriever.from_config(index_name, config)
    except (ValueError, FileNotFoundError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    try:
        llm = get_llm_provider(config.llm.provider, config.llm.model)
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    gen = Generator(llm)

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

        results = retriever.search(query)
        if not results:
            console.print("[yellow]No relevant code found for that query.[/yellow]\n")
            continue

        response_text = Text()
        console.print("\n[bold]Gritter:[/bold] ", end="")
        with Live(response_text, console=console, refresh_per_second=15):
            for token in gen.stream(query, results):
                response_text.append(token)

        console.print()

        # Show citations after each turn
        last_assistant = next(
            (m.content for m in reversed(gen.messages) if m.role == "assistant"), ""
        )
        citations = extract_citations(last_assistant)
        sources = format_sources(citations)
        if sources:
            console.print(f"[dim]{sources}[/dim]")

        console.print()
