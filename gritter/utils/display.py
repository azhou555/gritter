from __future__ import annotations
from rich.console import Console
from rich.table import Table

console = Console()


def print_error(msg: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {msg}")


def print_success(msg: str) -> None:
    console.print(f"[bold green]✓[/bold green] {msg}")


def print_warning(msg: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {msg}")


def make_table(title: str, rows: list[tuple[str, str]]) -> Table:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(key, value)
    return table
