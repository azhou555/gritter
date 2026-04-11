from __future__ import annotations
from dataclasses import dataclass
from gritter.models.chunk import CodeChunk


@dataclass
class RetrievalResult:
    chunk: CodeChunk
    score: float
