from __future__ import annotations

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


def ask(
    question: str = typer.Argument(..., help="Natural language question about the codebase"),
    name: str = typer.Option("default", "--name", "-n", help="Index name to query"),
    top_k: int = typer.Option(None, "--top-k", help="Number of results to retrieve"),
) -> None:
    """Ask a question about an indexed codebase."""
    config = GritterConfig()
    if top_k is not None:
        config.retrieval.top_k = top_k

    try:
        retriever = HybridRetriever.from_config(name, config)
    except (ValueError, FileNotFoundError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    try:
        llm = get_llm_provider(config.llm.provider, config.llm.model)
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    results = retriever.search(question)
    if not results:
        print_error("No relevant code found. Try re-indexing or a different query.")
        raise typer.Exit(1)

    gen = Generator(llm)
    response_text = Text()

    console.print()
    with Live(response_text, console=console, refresh_per_second=15):
        for token in gen.stream(question, results):
            response_text.append(token)

    console.print()

    # Extract and print citations
    full_response = "".join(
        m.content for m in gen.messages if m.role == "assistant"
    )
    citations = extract_citations(full_response)
    sources = format_sources(citations)
    if sources:
        console.print(f"\n[dim]{sources}[/dim]")
