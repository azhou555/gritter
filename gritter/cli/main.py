from __future__ import annotations
import typer
from gritter.cli.index_cmd import index
from gritter.cli.status_cmd import status
from gritter.cli.ask_cmd import ask
from gritter.cli.chat_cmd import chat
from gritter.cli.config_cmd import config_app

app = typer.Typer(
    name="gritter",
    help="Query any codebase with natural language.",
    add_completion=False,
)

app.command("index")(index)
app.command("status")(status)
app.command("ask")(ask)
app.command("chat")(chat)
app.add_typer(config_app, name="config")

if __name__ == "__main__":
    app()
