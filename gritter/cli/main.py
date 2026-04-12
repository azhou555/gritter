from __future__ import annotations
import typer
from gritter.cli.index_cmd import index
from gritter.cli.status_cmd import status

app = typer.Typer(
    name="gritter",
    help="Query any codebase with natural language.",
    add_completion=False,
)

app.command("index")(index)
app.command("status")(status)

if __name__ == "__main__":
    app()
