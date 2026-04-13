from __future__ import annotations

from pathlib import Path

import typer
import tomllib
import tomli_w

from gritter.models.config import GritterConfig
from gritter.utils.display import console, print_error, print_success

_USER_CONFIG = Path.home() / ".config" / "gritter" / "config.toml"

config_app = typer.Typer(help="Manage gritter configuration.")

# Valid dot-separated keys that can be set via CLI
_VALID_KEYS = {
    "embedding.provider",
    "embedding.model",
    "llm.provider",
    "llm.model",
    "llm.base_url",
    "index.chunk_min_tokens",
    "index.chunk_max_tokens",
    "index.chunk_overlap_tokens",
    "retrieval.top_k",
    "retrieval.candidate_k",
    "reranker.provider",
    "reranker.model",
}


def _load_toml(path: Path) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def _save_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _set_nested(d: dict, key: str, raw_value: str) -> None:
    """Set a dot-separated key in a nested dict, coercing value type."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    leaf = parts[-1]
    # Coerce to int if the key is known to be numeric
    int_keys = {
        "chunk_min_tokens", "chunk_max_tokens", "chunk_overlap_tokens",
        "top_k", "candidate_k",
    }
    if leaf in int_keys:
        try:
            d[leaf] = int(raw_value)
        except ValueError:
            raise typer.BadParameter(f"{key} requires an integer value, got {raw_value!r}")
    else:
        d[leaf] = raw_value


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g. embedding.provider)"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Write a config value to the user-level config file."""
    if key not in _VALID_KEYS:
        print_error(
            f"Unknown config key: {key!r}\n"
            f"Valid keys: {', '.join(sorted(_VALID_KEYS))}"
        )
        raise typer.Exit(1)

    data = _load_toml(_USER_CONFIG)
    try:
        _set_nested(data, key, value)
    except typer.BadParameter as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    _save_toml(_USER_CONFIG, data)
    print_success(f"Set [bold]{key}[/bold] = {value!r} in {_USER_CONFIG}")


@config_app.command("show")
def config_show() -> None:
    """Print the current effective configuration."""
    config = GritterConfig()

    table_data = {
        "embedding.provider": config.embedding.provider,
        "embedding.model": config.embedding.model or "(default)",
        "llm.provider": config.llm.provider,
        "llm.model": config.llm.model or "(default)",
        "llm.base_url": config.llm.base_url or "(default)",
        "index.chunk_min_tokens": str(config.index.chunk_min_tokens),
        "index.chunk_max_tokens": str(config.index.chunk_max_tokens),
        "index.chunk_overlap_tokens": str(config.index.chunk_overlap_tokens),
        "retrieval.top_k": str(config.retrieval.top_k),
        "retrieval.candidate_k": str(config.retrieval.candidate_k),
        "reranker.provider": config.reranker.provider,
        "reranker.model": config.reranker.model or "(default)",
        "data_dir": str(config.data_dir),
    }

    console.print("\n[bold]Effective configuration[/bold]\n")
    for k, v in table_data.items():
        console.print(f"  [cyan]{k:<30}[/cyan] {v}")
    console.print()
