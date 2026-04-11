from __future__ import annotations
import hashlib
from dataclasses import dataclass, field


@dataclass
class CodeChunk:
    content: str
    file_path: str
    language: str
    symbol_name: str | None
    symbol_type: str | None  # "function", "class", "method", "module"
    start_line: int
    end_line: int
    imports: list[str] = field(default_factory=list)
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        raw = f"{self.file_path}:{self.start_line}:{self.content}"
        self.chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def embedding_text(self) -> str:
        """Text sent to the embedding model: imports header + content."""
        if self.imports:
            return "\n".join(self.imports) + "\n\n" + self.content
        return self.content
