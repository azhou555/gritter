from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

_USER_CONFIG = Path.home() / ".config" / "gritter" / "config.toml"
_LOCAL_CONFIG = Path(".gritterrc")


class EmbeddingConfig(BaseModel):
    provider: Literal["voyage", "openai", "local"] = "voyage"
    model: str | None = None


class LLMConfig(BaseModel):
    provider: Literal["claude", "openai", "ollama"] = "claude"
    model: str | None = None
    base_url: str | None = None  # for Ollama


class IndexConfig(BaseModel):
    chunk_min_tokens: int = 50
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 20
    exclude_globs: list[str] = Field(default_factory=list)


class RetrievalConfig(BaseModel):
    top_k: int = 5
    candidate_k: int = 20


class GritterConfig(BaseSettings):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "gritter"
    )

    model_config = SettingsConfigDict(
        env_prefix="GRITTER_",
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        for config_path in (_LOCAL_CONFIG, _USER_CONFIG):
            if config_path.exists():
                sources.append(TomlConfigSettingsSource(settings_cls, toml_file=config_path))
        return tuple(sources)

    def index_dir(self, index_name: str) -> Path:
        """Return the storage directory for a named index."""
        return self.data_dir / "indexes" / index_name
