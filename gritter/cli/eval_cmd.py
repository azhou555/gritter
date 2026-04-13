from __future__ import annotations

from pathlib import Path

import typer

from rich.table import Table

from gritter.eval.runner import load_dataset, run_eval
from gritter.models.config import GritterConfig
from gritter.retrieval.hybrid import HybridRetriever
from gritter.utils.display import console, print_error


def eval(
    dataset: Path = typer.Argument(..., help="Path to JSONL eval dataset"),
    name: str = typer.Option(None, "--name", "-n", help="Index name to evaluate against"),
    k: int = typer.Option(5, "--k", help="Recall@k cutoff"),
) -> None:
    """Evaluate retrieval quality against a labeled dataset."""
    if not dataset.exists():
        print_error(f"Dataset not found: {dataset}")
        raise typer.Exit(1)

    config = GritterConfig()
    index_name = name or Path.cwd().name

    try:
        retriever = HybridRetriever.from_config(index_name, config)
    except (ValueError, FileNotFoundError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    records = load_dataset(dataset)
    if not records:
        print_error("Dataset is empty.")
        raise typer.Exit(1)

    console.print(f"\n[bold]Evaluating[/bold] {len(records)} queries against index [cyan]{index_name}[/cyan] (Recall@{k}, MRR)\n")

    result = run_eval(records, retriever, k=k)

    # Per-query table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Query", style="bold", max_width=60)
    table.add_column(f"Recall@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Top retrieved")
    for qr in result.query_results:
        recall_str = f"{qr.recall:.2f}"
        mrr_str = f"{qr.mrr:.2f}"
        retrieved_str = ", ".join(qr.retrieved_files[:3])
        if len(qr.retrieved_files) > 3:
            retrieved_str += f" (+{len(qr.retrieved_files) - 3})"
        table.add_row(
            qr.query[:60] + ("…" if len(qr.query) > 60 else ""),
            recall_str,
            mrr_str,
            retrieved_str,
        )

    console.print(table)
    console.print(
        f"\n[bold]Aggregate[/bold] — "
        f"Mean Recall@{k}: [cyan]{result.mean_recall:.3f}[/cyan]  "
        f"Mean MRR: [cyan]{result.mean_mrr:.3f}[/cyan]"
        f"  ({len(result.query_results)} queries)\n"
    )
