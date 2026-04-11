# Gritter RAG CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `gritter`, a CLI tool that indexes any codebase with AST-aware chunking and hybrid RAG retrieval, answering natural language questions with file-and-line citations.

**Architecture:** Three independent pipelines (indexing, retrieval, generation) share `CodeChunk` and `RetrievalResult` data models. CLI is a thin orchestrator. Embeddings and LLMs sit behind provider ABCs (Voyage/OpenAI/local and Claude/OpenAI/Ollama respectively). Hybrid retrieval combines ChromaDB dense search with BM25 sparse search via Reciprocal Rank Fusion. Re-ranking interface is stubbed as a no-op for v1.

**Tech Stack:** Python 3.11+, Typer, Rich, tree-sitter 0.23+, tree-sitter-python/typescript/rust, ChromaDB, rank-bm25, tiktoken, pydantic-settings, anthropic SDK, openai SDK, voyageai SDK, sentence-transformers, hatchling

---

## File Map

```
gritter/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── main.py                 # Typer app, registers all commands
│   ├── index_cmd.py            # gritter index <path>
│   ├── ask_cmd.py              # gritter ask "<question>"
│   ├── chat_cmd.py             # gritter chat (REPL)
│   ├── status_cmd.py           # gritter status
│   └── config_cmd.py           # gritter config set/show
├── models/
│   ├── __init__.py
│   ├── chunk.py                # CodeChunk dataclass
│   ├── query.py                # RetrievalResult dataclass
│   └── config.py               # GritterConfig (pydantic-settings)
├── utils/
│   ├── __init__.py
│   ├── languages.py            # EXTENSION_TO_LANGUAGE, EXCLUDED_DIRS, detect_language()
│   ├── tokens.py               # count_tokens(), split_at_token_boundary()
│   └── display.py              # Rich console, panel/table helpers
├── indexing/
│   ├── __init__.py
│   ├── discovery.py            # discover_files(root, exclude_globs) → [(Path, lang)]
│   ├── heuristic_chunker.py    # chunk_file_heuristic(content, path, lang, ...) → [CodeChunk]
│   ├── ast_chunker.py          # chunk_file_ast(content, path, lang, ...) → [CodeChunk]
│   ├── chunker.py              # chunk_file dispatcher (public interface)
│   └── pipeline.py             # run_indexing_pipeline(root, config, embed_provider)
├── providers/
│   ├── __init__.py
│   ├── embeddings.py           # EmbeddingProvider ABC + Voyage/OpenAI/Local
│   └── llm.py                  # LLMProvider ABC + Claude/OpenAI/Ollama
├── storage/
│   ├── __init__.py
│   ├── vector_store.py         # VectorStore (ChromaDB wrapper)
│   ├── bm25_store.py           # BM25Store (rank_bm25 + pickle persistence)
│   └── index_meta.py           # IndexMeta (JSON metadata read/write/validate)
├── retrieval/
│   ├── __init__.py
│   ├── dense.py                # dense_search(query_vec, store, k) → [RetrievalResult]
│   ├── sparse.py               # sparse_search(query, bm25_store, chunks, k) → [RetrievalResult]
│   ├── fusion.py               # reciprocal_rank_fusion(dense, sparse) → [RetrievalResult]
│   ├── reranker.py             # Reranker ABC + NoOpReranker
│   └── hybrid.py               # HybridRetriever.search(query, top_k) → [RetrievalResult]
└── generation/
    ├── __init__.py
    ├── prompts.py              # SYSTEM_PROMPT, build_context_block(results)
    ├── citations.py            # extract_citations(text), format_sources(citations)
    └── generator.py            # Generator.stream(query, results) → Iterator[str]
tests/
├── conftest.py
├── fixtures/
│   ├── python_project/sample.py
│   ├── typescript_project/sample.ts
│   └── rust_project/sample.rs
├── test_discovery.py
├── test_chunker.py
├── test_retrieval.py
└── test_generation.py
pyproject.toml
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `gritter/__init__.py` (and all sub-package `__init__.py` files)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "gritter"
version = "0.1.0"
description = "Query any codebase with natural language using RAG"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "typer[all]>=0.12",
    "rich>=13",
    "chromadb>=0.5",
    "rank-bm25>=0.2",
    "tree-sitter>=0.23",
    "tree-sitter-python>=0.23",
    "tree-sitter-typescript>=0.23",
    "tree-sitter-rust>=0.23",
    "pydantic-settings>=2.3",
    "anthropic>=0.30",
    "openai>=1.0",
    "voyageai>=0.3",
    "sentence-transformers>=3",
    "tiktoken>=0.7",
    "tomli-w>=1",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-mock>=3"]

[project.scripts]
gritter = "gritter.cli.main:app"

[tool.hatch.build.targets.wheel]
packages = ["gritter"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package `__init__.py` files**

```bash
mkdir -p gritter/{cli,models,utils,indexing,providers,storage,retrieval,generation}
mkdir -p tests/fixtures/{python_project,typescript_project,rust_project}
touch gritter/__init__.py
touch gritter/cli/__init__.py
touch gritter/models/__init__.py
touch gritter/utils/__init__.py
touch gritter/indexing/__init__.py
touch gritter/providers/__init__.py
touch gritter/storage/__init__.py
touch gritter/retrieval/__init__.py
touch gritter/generation/__init__.py
touch tests/__init__.py
touch tests/fixtures/__init__.py
```

- [ ] **Step 3: Install in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: no errors, `gritter` command available (will fail until `main.py` exists).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml gritter/ tests/
git commit -m "chore: scaffold project structure"
```

---

## Task 2: Core Data Models

**Files:**
- Create: `gritter/models/chunk.py`
- Create: `gritter/models/query.py`
- Create: `gritter/utils/tokens.py`
- Test: `tests/test_chunker.py` (partial — token counting test only at this stage)

- [ ] **Step 1: Write failing test for CodeChunk**

```python
# tests/test_chunker.py
from gritter.models.chunk import CodeChunk


def test_chunk_id_is_deterministic():
    chunk = CodeChunk(
        content="def foo(): pass",
        file_path="src/foo.py",
        language="python",
        symbol_name="foo",
        symbol_type="function",
        start_line=1,
        end_line=1,
    )
    chunk2 = CodeChunk(
        content="def foo(): pass",
        file_path="src/foo.py",
        language="python",
        symbol_name="foo",
        symbol_type="function",
        start_line=1,
        end_line=1,
    )
    assert chunk.chunk_id == chunk2.chunk_id
    assert len(chunk.chunk_id) == 16


def test_embedding_text_prepends_imports():
    chunk = CodeChunk(
        content="def foo(): pass",
        file_path="src/foo.py",
        language="python",
        symbol_name="foo",
        symbol_type="function",
        start_line=1,
        end_line=1,
        imports=["import os", "from pathlib import Path"],
    )
    text = chunk.embedding_text()
    assert text.startswith("import os\nfrom pathlib import Path\n\n")
    assert "def foo(): pass" in text


def test_embedding_text_no_imports():
    chunk = CodeChunk(
        content="def foo(): pass",
        file_path="src/foo.py",
        language="python",
        symbol_name="foo",
        symbol_type="function",
        start_line=1,
        end_line=1,
    )
    assert chunk.embedding_text() == "def foo(): pass"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_chunker.py::test_chunk_id_is_deterministic -v
```

Expected: `ModuleNotFoundError: No module named 'gritter.models.chunk'`

- [ ] **Step 3: Implement `gritter/models/chunk.py`**

```python
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
        raw = f"{self.file_path}:{self.content}"
        self.chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def embedding_text(self) -> str:
        """Text sent to the embedding model: imports header + content."""
        if self.imports:
            return "\n".join(self.imports) + "\n\n" + self.content
        return self.content
```

- [ ] **Step 4: Implement `gritter/models/query.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from gritter.models.chunk import CodeChunk


@dataclass
class RetrievalResult:
    chunk: CodeChunk
    score: float
```

- [ ] **Step 5: Implement `gritter/utils/tokens.py`**

```python
from __future__ import annotations
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (GPT-4 tokenizer — good approximation for all models)."""
    return len(_enc.encode(text))


def split_at_token_boundary(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into chunks of at most max_tokens with overlap_tokens overlap."""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap_tokens
    return chunks
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v -k "chunk_id or embedding_text"
```

Expected: 3 PASSED

- [ ] **Step 7: Commit**

```bash
git add gritter/models/ gritter/utils/tokens.py tests/test_chunker.py
git commit -m "feat: add CodeChunk, RetrievalResult, and token counting utility"
```

---

## Task 3: Language Detection and File Discovery

**Files:**
- Create: `gritter/utils/languages.py`
- Create: `gritter/indexing/discovery.py`
- Create: `tests/test_discovery.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_discovery.py
from pathlib import Path
import pytest
from gritter.utils.languages import detect_language
from gritter.indexing.discovery import discover_files


def test_detect_python():
    assert detect_language("src/foo.py") == "python"


def test_detect_typescript():
    assert detect_language("src/foo.ts") == "typescript"
    assert detect_language("src/foo.tsx") == "typescript"


def test_detect_rust():
    assert detect_language("src/foo.rs") == "rust"


def test_detect_unknown_returns_none():
    assert detect_language("src/foo.go") is None
    assert detect_language("src/foo.rb") is None


def test_discover_files_returns_supported_files(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "lib.ts").write_text("export const x = 1;")
    (tmp_path / "notes.txt").write_text("some notes")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    results = discover_files(tmp_path)
    paths = {p.name for p, _ in results}
    assert "main.py" in paths
    assert "lib.ts" in paths
    assert "notes.txt" not in paths
    assert "image.png" not in paths


def test_discover_files_skips_excluded_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lodash.ts").write_text("export default {}")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.ts").write_text("export const app = 1;")
    results = discover_files(tmp_path)
    paths = {p.name for p, _ in results}
    assert "lodash.ts" not in paths
    assert "app.ts" in paths


def test_discover_files_respects_exclude_globs(tmp_path):
    (tmp_path / "main.py").write_text("x = 1")
    (tmp_path / "generated.py").write_text("x = 2")
    results = discover_files(tmp_path, exclude_globs=["*generated*"])
    paths = {p.name for p, _ in results}
    assert "main.py" in paths
    assert "generated.py" not in paths


def test_discover_files_skips_lockfiles(tmp_path):
    (tmp_path / "app.ts").write_text("export {}")
    (tmp_path / "package-lock.json").write_text("{}")
    results = discover_files(tmp_path)
    paths = {p.name for p, _ in results}
    assert "app.ts" in paths
    assert "package-lock.json" not in paths
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_discovery.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `gritter/utils/languages.py`**

```python
from __future__ import annotations

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".rs": "rust",
}

EXCLUDED_DIRS: set[str] = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    "dist", "build", "target", ".next", ".nuxt", ".cache",
    "venv", ".venv", "env", ".env", ".tox",
    "coverage", ".coverage", ".mypy_cache", ".ruff_cache",
}

EXCLUDED_FILENAMES: set[str] = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Gemfile.lock",
    "composer.lock", "bun.lockb",
}

EXCLUDED_EXTENSIONS: set[str] = {
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll", ".exe",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".whl", ".egg",
    ".lock", ".min.js", ".map",
}


def detect_language(path: str) -> str | None:
    """Return the language name for a file path, or None if unsupported."""
    from pathlib import Path
    suffix = Path(path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(suffix)
```

- [ ] **Step 4: Implement `gritter/indexing/discovery.py`**

```python
from __future__ import annotations
from pathlib import Path

from gritter.utils.languages import (
    detect_language,
    EXCLUDED_DIRS,
    EXCLUDED_EXTENSIONS,
    EXCLUDED_FILENAMES,
)


def discover_files(
    root: Path,
    exclude_globs: list[str] | None = None,
) -> list[tuple[Path, str]]:
    """Walk root and return (path, language) pairs for all supported files."""
    exclude_globs = exclude_globs or []
    results: list[tuple[Path, str]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        # Skip hidden directories and known excluded dirs
        parts = path.relative_to(root).parts
        if any(part.startswith(".") or part in EXCLUDED_DIRS for part in parts[:-1]):
            continue

        # Skip excluded filenames
        if path.name in EXCLUDED_FILENAMES:
            continue

        # Skip excluded extensions
        if path.suffix.lower() in EXCLUDED_EXTENSIONS:
            continue

        # Skip user-configured glob patterns
        rel = path.relative_to(root)
        if any(rel.match(g) for g in exclude_globs):
            continue

        language = detect_language(str(path))
        if language is not None:
            results.append((path, language))

    return results
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_discovery.py -v
```

Expected: 7 PASSED

- [ ] **Step 6: Commit**

```bash
git add gritter/utils/languages.py gritter/indexing/discovery.py tests/test_discovery.py
git commit -m "feat: add language detection and file discovery"
```

---

## Task 4: Test Fixture Files

**Files:**
- Create: `tests/fixtures/python_project/sample.py`
- Create: `tests/fixtures/typescript_project/sample.ts`
- Create: `tests/fixtures/rust_project/sample.rs`
- Create: `tests/conftest.py`

These fixture files have a known, fixed structure that tests assert against.

- [ ] **Step 1: Create `tests/fixtures/python_project/sample.py`**

```python
# tests/fixtures/python_project/sample.py
import os
from pathlib import Path


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


class Calculator:
    """A simple calculator."""

    def __init__(self, initial: int = 0) -> None:
        self.value = initial

    def multiply(self, factor: int) -> int:
        """Multiply current value by factor."""
        self.value *= factor
        return self.value

    def reset(self) -> None:
        """Reset to zero."""
        self.value = 0
```

- [ ] **Step 2: Create `tests/fixtures/typescript_project/sample.ts`**

```typescript
// tests/fixtures/typescript_project/sample.ts
import { readFileSync } from "fs";
import path from "path";

export function greet(name: string): string {
    return `Hello, ${name}!`;
}

export class Formatter {
    private prefix: string;

    constructor(prefix: string) {
        this.prefix = prefix;
    }

    format(value: string): string {
        return `${this.prefix}: ${value}`;
    }
}
```

- [ ] **Step 3: Create `tests/fixtures/rust_project/sample.rs`**

```rust
// tests/fixtures/rust_project/sample.rs
use std::fmt;

pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    pub fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

pub fn origin() -> Point {
    Point::new(0.0, 0.0)
}
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
# tests/conftest.py
from pathlib import Path
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def python_fixture_path() -> Path:
    return FIXTURES_DIR / "python_project" / "sample.py"


@pytest.fixture
def typescript_fixture_path() -> Path:
    return FIXTURES_DIR / "typescript_project" / "sample.ts"


@pytest.fixture
def rust_fixture_path() -> Path:
    return FIXTURES_DIR / "rust_project" / "sample.rs"
```

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/ tests/conftest.py
git commit -m "test: add fixture files for Python, TypeScript, and Rust"
```

---

## Task 5: Heuristic Chunker

**Files:**
- Create: `gritter/indexing/heuristic_chunker.py`
- Modify: `tests/test_chunker.py`

- [ ] **Step 1: Add heuristic chunker tests**

Add to `tests/test_chunker.py`:

```python
from gritter.indexing.heuristic_chunker import chunk_file_heuristic


def test_heuristic_chunker_splits_top_level_blocks():
    content = """def foo():
    x = 1
    return x


def bar():
    y = 2
    return y
"""
    chunks = chunk_file_heuristic(content, "test.go", "go")
    assert len(chunks) == 2
    assert "def foo" in chunks[0].content
    assert "def bar" in chunks[1].content


def test_heuristic_chunker_doesnt_split_internal_blank_lines():
    content = """def foo():
    x = 1

    y = 2
    return x + y
"""
    chunks = chunk_file_heuristic(content, "test.go", "go")
    # Blank line inside the function should NOT create a split
    assert len(chunks) == 1
    assert "y = 2" in chunks[0].content


def test_heuristic_chunker_merges_small_chunks():
    # Two tiny blocks that each fall under min_tokens should be merged
    content = "x = 1\n\ny = 2\n"
    chunks = chunk_file_heuristic(content, "test.go", "go", min_tokens=10)
    # Both are tiny — should be merged into one chunk
    assert len(chunks) == 1


def test_heuristic_chunker_populates_metadata():
    content = "def foo():\n    return 1\n"
    chunks = chunk_file_heuristic(content, "path/to/file.go", "go")
    assert chunks[0].file_path == "path/to/file.go"
    assert chunks[0].language == "go"
    assert chunks[0].symbol_name is None
    assert chunks[0].symbol_type == "module"
    assert chunks[0].start_line == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunker.py -v -k "heuristic"
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `gritter/indexing/heuristic_chunker.py`**

```python
from __future__ import annotations
from gritter.models.chunk import CodeChunk
from gritter.utils.tokens import count_tokens, split_at_token_boundary


def chunk_file_heuristic(
    content: str,
    file_path: str,
    language: str,
    min_tokens: int = 50,
    max_tokens: int = 512,
    overlap_tokens: int = 20,
) -> list[CodeChunk]:
    """Split content into chunks at blank lines where the next line has zero indentation."""
    raw_blocks = _split_top_level_blocks(content)
    if not raw_blocks:
        return []

    # Track line numbers for each block
    line_cursor = 1
    block_with_lines: list[tuple[str, int, int]] = []
    for block in raw_blocks:
        block_line_count = block.count("\n") + 1
        end_line = line_cursor + block_line_count - 1
        block_with_lines.append((block, line_cursor, end_line))
        line_cursor = end_line + 2  # account for blank separator

    # Apply token guardrails
    return _apply_guardrails(block_with_lines, file_path, language, min_tokens, max_tokens, overlap_tokens)


def _split_top_level_blocks(content: str) -> list[str]:
    """Split at blank lines where the next non-blank line has zero indentation."""
    lines = content.split("\n")
    blocks: list[str] = []
    current: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            # Look ahead to find next non-blank line
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1

            if j >= len(lines):
                # Trailing blank lines — end current block
                current.append(line)
                i += 1
            elif lines[j] and lines[j][0] not in (" ", "\t"):
                # Next content is at indent level 0 — this is a block boundary
                block = "\n".join(current).rstrip()
                if block.strip():
                    blocks.append(block)
                current = []
                i = j  # skip the blank lines
            else:
                # Next content is indented — we're inside a block, stay
                current.extend(lines[i:j])
                i = j
        else:
            current.append(line)
            i += 1

    if current:
        block = "\n".join(current).rstrip()
        if block.strip():
            blocks.append(block)

    return blocks


def _apply_guardrails(
    block_with_lines: list[tuple[str, int, int]],
    file_path: str,
    language: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[CodeChunk]:
    """Merge small blocks and split large ones, then create CodeChunk objects."""
    # First pass: split blocks that exceed max_tokens
    expanded: list[tuple[str, int, int]] = []
    for content, start, end in block_with_lines:
        if count_tokens(content) > max_tokens:
            parts = split_at_token_boundary(content, max_tokens, overlap_tokens)
            for part in parts:
                expanded.append((part, start, end))  # approximate line numbers for splits
        else:
            expanded.append((content, start, end))

    # Second pass: merge consecutive small blocks
    merged: list[tuple[str, int, int]] = []
    for content, start, end in expanded:
        if merged and count_tokens(merged[-1][0]) < min_tokens:
            prev_content, prev_start, _ = merged[-1]
            merged[-1] = (prev_content + "\n\n" + content, prev_start, end)
        else:
            merged.append((content, start, end))

    return [
        CodeChunk(
            content=content,
            file_path=file_path,
            language=language,
            symbol_name=None,
            symbol_type="module",
            start_line=start,
            end_line=end,
        )
        for content, start, end in merged
        if content.strip()
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v -k "heuristic"
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add gritter/indexing/heuristic_chunker.py tests/test_chunker.py
git commit -m "feat: add heuristic chunker with blank-line splitting and token guardrails"
```

---

## Task 6: AST Chunker

**Files:**
- Create: `gritter/indexing/ast_chunker.py`
- Modify: `tests/test_chunker.py`

- [ ] **Step 1: Add AST chunker tests**

Add to `tests/test_chunker.py`:

```python
from gritter.indexing.ast_chunker import chunk_file_ast


class TestPythonASTChunker:
    def test_extracts_top_level_functions(self, python_fixture_path):
        content = python_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(python_fixture_path), "python")
        function_chunks = [c for c in chunks if c.symbol_type == "function"]
        names = {c.symbol_name for c in function_chunks}
        assert "add" in names
        assert "subtract" in names

    def test_extracts_class(self, python_fixture_path):
        content = python_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(python_fixture_path), "python")
        class_chunks = [c for c in chunks if c.symbol_type in ("class", "method")]
        assert any(c.symbol_name == "Calculator" for c in class_chunks)

    def test_chunk_line_numbers_are_correct(self, python_fixture_path):
        content = python_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(python_fixture_path), "python")
        add_chunk = next(c for c in chunks if c.symbol_name == "add")
        assert add_chunk.start_line >= 1
        assert add_chunk.end_line >= add_chunk.start_line
        assert "def add" in add_chunk.content

    def test_imports_are_captured(self, python_fixture_path):
        content = python_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(python_fixture_path), "python")
        assert all(len(c.imports) > 0 for c in chunks)
        assert any("import os" in imp for chunk in chunks for imp in chunk.imports)

    def test_syntax_error_raises_value_error(self):
        with pytest.raises(ValueError, match="Parse error"):
            chunk_file_ast("def foo(", "bad.py", "python")


class TestTypeScriptASTChunker:
    def test_extracts_exported_function(self, typescript_fixture_path):
        content = typescript_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(typescript_fixture_path), "typescript")
        names = {c.symbol_name for c in chunks}
        assert "greet" in names

    def test_extracts_class(self, typescript_fixture_path):
        content = typescript_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(typescript_fixture_path), "typescript")
        assert any(c.symbol_name == "Formatter" for c in chunks)


class TestRustASTChunker:
    def test_extracts_struct(self, rust_fixture_path):
        content = rust_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(rust_fixture_path), "rust")
        names = {c.symbol_name for c in chunks}
        assert "Point" in names

    def test_extracts_standalone_function(self, rust_fixture_path):
        content = rust_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(rust_fixture_path), "rust")
        names = {c.symbol_name for c in chunks}
        assert "origin" in names

    def test_extracts_impl_block(self, rust_fixture_path):
        content = rust_fixture_path.read_text()
        chunks = chunk_file_ast(content, str(rust_fixture_path), "rust")
        # Either the impl block itself or its methods should appear
        assert any("impl" in c.content for c in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunker.py -v -k "AST"
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `gritter/indexing/ast_chunker.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts
import tree_sitter_rust as tsrust

from gritter.models.chunk import CodeChunk
from gritter.utils.tokens import count_tokens, split_at_token_boundary

# --- Language setup ---

_PARSERS: dict[str, Parser] = {
    "python": Parser(Language(tspython.language())),
    "typescript": Parser(Language(tsts.language_typescript())),
    "rust": Parser(Language(tsrust.language())),
}

# Node types at top level that we extract as chunks
_TOP_LEVEL_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "typescript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
    },
    "rust": {
        "function_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
        "macro_rules",
        "type_item",
        "const_item",
    },
}

# Node types representing import/use statements
_IMPORT_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "typescript": {"import_statement"},
    "rust": {"use_declaration"},
}

# Node types for methods inside a class/impl body
_METHOD_TYPES: dict[str, set[str]] = {
    "python": {"function_definition"},
    "typescript": {"method_definition"},
    "rust": {"function_item"},
}

# Body container node types for class/impl
_BODY_TYPES: dict[str, set[str]] = {
    "python": {"block"},
    "typescript": {"class_body"},
    "rust": {"declaration_list"},
}


# --- Public interface ---

def chunk_file_ast(
    content: str,
    file_path: str,
    language: str,
    min_tokens: int = 50,
    max_tokens: int = 512,
    overlap_tokens: int = 20,
) -> list[CodeChunk]:
    """Parse content with tree-sitter and extract top-level definitions as chunks.

    Raises ValueError if the file has syntax errors.
    """
    source = content.encode("utf-8")
    parser = _PARSERS[language]
    tree = parser.parse(source)

    if tree.root_node.has_error:
        raise ValueError(f"Parse error in {file_path}")

    imports = _extract_imports(tree.root_node, source, language)
    top_level_types = _TOP_LEVEL_TYPES[language]

    raw_chunks: list[CodeChunk] = []
    for node in tree.root_node.children:
        if node.type not in top_level_types:
            continue

        node_text = source[node.start_byte:node.end_byte].decode("utf-8")
        symbol_name, symbol_type = _get_symbol(node, source, language)
        tokens = count_tokens(node_text)

        if tokens > max_tokens:
            sub = _split_large_node(
                node, source, file_path, language, imports,
                min_tokens, max_tokens, overlap_tokens,
            )
            raw_chunks.extend(sub)
        else:
            raw_chunks.append(CodeChunk(
                content=node_text,
                file_path=file_path,
                language=language,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                imports=imports,
            ))

    return _merge_small_chunks(raw_chunks, min_tokens)


# --- Symbol extraction ---

def _get_symbol(node: Node, source: bytes, language: str) -> tuple[str | None, str | None]:
    if language == "python":
        return _get_python_symbol(node, source)
    elif language == "typescript":
        return _get_typescript_symbol(node, source)
    elif language == "rust":
        return _get_rust_symbol(node, source)
    return (None, "module")


def _get_python_symbol(node: Node, source: bytes) -> tuple[str | None, str | None]:
    if node.type == "function_definition":
        name = node.child_by_field_name("name")
        return (_node_text(name, source), "function")
    elif node.type == "class_definition":
        name = node.child_by_field_name("name")
        return (_node_text(name, source), "class")
    elif node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _get_python_symbol(child, source)
    return (None, "module")


def _get_typescript_symbol(node: Node, source: bytes) -> tuple[str | None, str | None]:
    if node.type in ("function_declaration", "generator_function_declaration"):
        name = node.child_by_field_name("name")
        return (_node_text(name, source), "function")
    elif node.type == "class_declaration":
        name = node.child_by_field_name("name")
        return (_node_text(name, source), "class")
    elif node.type == "export_statement":
        for child in node.children:
            if child.type in ("function_declaration", "class_declaration",
                               "lexical_declaration", "generator_function_declaration"):
                return _get_typescript_symbol(child, source)
    elif node.type == "lexical_declaration":
        for child in node.children:
            if child.type == "variable_declarator":
                name = child.child_by_field_name("name")
                return (_node_text(name, source), "function")
    return (None, "module")


def _get_rust_symbol(node: Node, source: bytes) -> tuple[str | None, str | None]:
    type_map = {
        "function_item": "function",
        "struct_item": "class",
        "enum_item": "class",
        "trait_item": "class",
        "impl_item": "class",
        "macro_rules": "function",
        "type_item": "module",
        "const_item": "module",
    }
    symbol_type = type_map.get(node.type, "module")
    name = node.child_by_field_name("name")
    return (_node_text(name, source), symbol_type)


# --- Import extraction ---

def _extract_imports(root: Node, source: bytes, language: str) -> list[str]:
    import_types = _IMPORT_TYPES[language]
    return [
        source[child.start_byte:child.end_byte].decode("utf-8")
        for child in root.children
        if child.type in import_types
    ]


# --- Large node splitting ---

def _split_large_node(
    node: Node,
    source: bytes,
    file_path: str,
    language: str,
    imports: list[str],
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[CodeChunk]:
    """Split a large node by extracting methods/functions from its body."""
    method_types = _METHOD_TYPES.get(language, set())
    body_types = _BODY_TYPES.get(language, set())
    parent_name, _ = _get_symbol(node, source, language)

    methods: list[Node] = []
    for child in node.children:
        if child.type in body_types:
            for grandchild in child.children:
                if grandchild.type in method_types:
                    methods.append(grandchild)

    if not methods:
        # Cannot split further — fall back to token-boundary splitting
        node_text = source[node.start_byte:node.end_byte].decode("utf-8")
        parts = split_at_token_boundary(node_text, max_tokens, overlap_tokens)
        return [
            CodeChunk(
                content=part,
                file_path=file_path,
                language=language,
                symbol_name=parent_name,
                symbol_type="module",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                imports=imports,
            )
            for part in parts
        ]

    chunks: list[CodeChunk] = []
    for method in methods:
        method_text = source[method.start_byte:method.end_byte].decode("utf-8")
        method_name, _ = _get_symbol(method, source, language)
        qualified_name = (
            f"{parent_name}.{method_name}"
            if parent_name and method_name
            else method_name or parent_name
        )

        tokens = count_tokens(method_text)
        if tokens < min_tokens and chunks:
            prev = chunks[-1]
            chunks[-1] = CodeChunk(
                content=prev.content + "\n\n" + method_text,
                file_path=prev.file_path,
                language=prev.language,
                symbol_name=prev.symbol_name,
                symbol_type=prev.symbol_type,
                start_line=prev.start_line,
                end_line=method.end_point[0] + 1,
                imports=imports,
            )
        else:
            chunks.append(CodeChunk(
                content=method_text,
                file_path=file_path,
                language=language,
                symbol_name=qualified_name,
                symbol_type="method",
                start_line=method.start_point[0] + 1,
                end_line=method.end_point[0] + 1,
                imports=imports,
            ))

    return chunks


# --- Helpers ---

def _node_text(node: Node | None, source: bytes) -> str | None:
    if node is None:
        return None
    return source[node.start_byte:node.end_byte].decode("utf-8")


def _merge_small_chunks(chunks: list[CodeChunk], min_tokens: int) -> list[CodeChunk]:
    """Merge consecutive chunks that are below the minimum token threshold."""
    if not chunks:
        return []
    merged: list[CodeChunk] = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        if count_tokens(prev.content) < min_tokens:
            merged[-1] = CodeChunk(
                content=prev.content + "\n\n" + chunk.content,
                file_path=prev.file_path,
                language=prev.language,
                symbol_name=prev.symbol_name,
                symbol_type=prev.symbol_type,
                start_line=prev.start_line,
                end_line=chunk.end_line,
                imports=prev.imports,
            )
        else:
            merged.append(chunk)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v -k "AST"
```

Expected: all AST tests PASSED

- [ ] **Step 5: Commit**

```bash
git add gritter/indexing/ast_chunker.py tests/test_chunker.py
git commit -m "feat: add AST chunker for Python, TypeScript, and Rust via tree-sitter"
```

---

## Task 7: Chunker Dispatcher

**Files:**
- Create: `gritter/indexing/chunker.py`
- Modify: `tests/test_chunker.py`

- [ ] **Step 1: Add dispatcher tests**

Add to `tests/test_chunker.py`:

```python
from gritter.indexing.chunker import chunk_file


def test_dispatcher_routes_python_to_ast(python_fixture_path):
    content = python_fixture_path.read_text()
    chunks = chunk_file(content, str(python_fixture_path), "python")
    assert any(c.symbol_type == "function" for c in chunks)


def test_dispatcher_routes_unknown_to_heuristic():
    content = "func main() {\n    fmt.Println(\"hello\")\n}\n\nfunc add(a, b int) int {\n    return a + b\n}\n"
    chunks = chunk_file(content, "main.go", "go")
    assert len(chunks) >= 1
    assert all(c.symbol_type == "module" for c in chunks)


def test_dispatcher_falls_back_to_heuristic_on_parse_error():
    # Python file with a syntax error — should fall back gracefully
    bad_content = "def foo(:\n    pass\n"
    chunks = chunk_file(bad_content, "broken.py", "python")
    # Should not raise, should return at least one chunk
    assert len(chunks) >= 1
    assert all(c.symbol_type == "module" for c in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunker.py -v -k "dispatcher"
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `gritter/indexing/chunker.py`**

```python
from __future__ import annotations
import logging
from pathlib import Path

from gritter.models.chunk import CodeChunk
from gritter.indexing.ast_chunker import chunk_file_ast
from gritter.indexing.heuristic_chunker import chunk_file_heuristic
from gritter.utils.languages import EXTENSION_TO_LANGUAGE

logger = logging.getLogger(__name__)

_AST_LANGUAGES = set(EXTENSION_TO_LANGUAGE.values())  # python, typescript, rust


def chunk_file(
    content: str,
    file_path: str,
    language: str,
    min_tokens: int = 50,
    max_tokens: int = 512,
    overlap_tokens: int = 20,
) -> list[CodeChunk]:
    """Dispatch to AST or heuristic chunker based on language support.

    Falls back to heuristic if AST parsing raises (syntax errors, grammar issues).
    """
    if language in _AST_LANGUAGES:
        try:
            return chunk_file_ast(
                content, file_path, language, min_tokens, max_tokens, overlap_tokens
            )
        except Exception as exc:
            logger.warning(
                "AST chunking failed for %s (%s), falling back to heuristic: %s",
                file_path, language, exc,
            )

    return chunk_file_heuristic(
        content, file_path, language, min_tokens, max_tokens, overlap_tokens
    )
```

- [ ] **Step 4: Run all chunker tests**

```bash
pytest tests/test_chunker.py -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add gritter/indexing/chunker.py tests/test_chunker.py
git commit -m "feat: add chunker dispatcher with AST fallback to heuristic"
```

---

## Task 8: Configuration Model

**Files:**
- Create: `gritter/models/config.py`

No tests needed here — pydantic-settings validates at construction time; the defaults are trivially correct.

- [ ] **Step 1: Implement `gritter/models/config.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Literal, Any
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
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
python -c "from gritter.models.config import GritterConfig; c = GritterConfig(); print(c.retrieval.top_k)"
```

Expected: `5`

- [ ] **Step 3: Commit**

```bash
git add gritter/models/config.py
git commit -m "feat: add GritterConfig pydantic-settings model"
```

---

## Task 9: Embedding Providers

**Files:**
- Create: `gritter/providers/embeddings.py`

- [ ] **Step 1: Implement `gritter/providers/embeddings.py`**

```python
from __future__ import annotations
import os
import time
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class VoyageEmbedder(EmbeddingProvider):
    """voyage-code-3 — optimized for source code retrieval."""

    def __init__(self, model: str = "voyage-code-3") -> None:
        import voyageai
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "VOYAGE_API_KEY environment variable is required for Voyage embeddings."
            )
        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self._model, input_type="document")
        return result.embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedder(EmbeddingProvider):
    """text-embedding-3-small — general-purpose, widely available."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for OpenAI embeddings."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimension = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedder(EmbeddingProvider):
    """sentence-transformers — runs locally, no API key required."""

    def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model, trust_remote_code=True)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_provider(
    provider: str,
    model: str | None = None,
) -> EmbeddingProvider:
    """Factory: return the configured embedding provider."""
    if provider == "voyage":
        return VoyageEmbedder(model or "voyage-code-3")
    elif provider == "openai":
        return OpenAIEmbedder(model or "text-embedding-3-small")
    elif provider == "local":
        return LocalEmbedder(model or "nomic-ai/nomic-embed-text-v1")
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}")
```

- [ ] **Step 2: Commit**

```bash
git add gritter/providers/embeddings.py
git commit -m "feat: add EmbeddingProvider ABC with Voyage, OpenAI, and local implementations"
```

---

## Task 10: Storage Layer

**Files:**
- Create: `gritter/storage/vector_store.py`
- Create: `gritter/storage/bm25_store.py`
- Create: `gritter/storage/index_meta.py`

- [ ] **Step 1: Implement `gritter/storage/vector_store.py`**

```python
from __future__ import annotations
from pathlib import Path
import chromadb
from gritter.models.chunk import CodeChunk


class VectorStore:
    """ChromaDB-backed vector store. One collection per index."""

    def __init__(self, index_dir: Path, collection_name: str = "chunks") -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings."""
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "file_path": c.file_path,
                    "language": c.language,
                    "symbol_name": c.symbol_name or "",
                    "symbol_type": c.symbol_type or "",
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "imports": "\n".join(c.imports),
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], k: int = 20) -> list[tuple[CodeChunk, float]]:
        """Return up to k (chunk, distance) pairs ordered by cosine similarity."""
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
        )
        chunks_and_scores: list[tuple[CodeChunk, float]] = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunk = CodeChunk(
                content=doc,
                file_path=meta["file_path"],
                language=meta["language"],
                symbol_name=meta["symbol_name"] or None,
                symbol_type=meta["symbol_type"] or None,
                start_line=int(meta["start_line"]),
                end_line=int(meta["end_line"]),
                imports=meta["imports"].split("\n") if meta["imports"] else [],
            )
            chunks_and_scores.append((chunk, float(dist)))
        return chunks_and_scores

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection.name)
```

- [ ] **Step 2: Implement `gritter/storage/bm25_store.py`**

```python
from __future__ import annotations
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from gritter.models.chunk import CodeChunk


class BM25Store:
    """BM25 index over chunk text, persisted to disk."""

    def __init__(self, index_dir: Path) -> None:
        self._path = index_dir / "bm25.pkl"
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunks: list[CodeChunk] = []

    def build(self, chunks: list[CodeChunk]) -> None:
        """Build BM25 index from chunks."""
        self._chunks = chunks
        self._chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [c.content.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._save()

    def load(self) -> None:
        """Load index from disk."""
        if not self._path.exists():
            raise FileNotFoundError(f"BM25 index not found at {self._path}")
        with open(self._path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]
        self._chunks = data["chunks"]

    def query(self, query: str, k: int = 20) -> list[tuple[CodeChunk, float]]:
        """Return up to k (chunk, bm25_score) pairs."""
        if self._bm25 is None:
            raise RuntimeError("BM25 index not loaded. Call load() first.")
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(
                {"bm25": self._bm25, "chunk_ids": self._chunk_ids, "chunks": self._chunks},
                f,
            )
```

- [ ] **Step 3: Implement `gritter/storage/index_meta.py`**

```python
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path


class IndexMeta:
    """JSON file recording index provenance and statistics."""

    def __init__(self, index_dir: Path) -> None:
        self._path = index_dir / "meta.json"
        self._data: dict = {}

    def write(
        self,
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        file_count: int,
        chunk_count: int,
        languages: list[str],
    ) -> None:
        self._data = {
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "file_count": file_count,
            "chunk_count": chunk_count,
            "languages": sorted(languages),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def read(self) -> dict:
        if not self._path.exists():
            raise FileNotFoundError(f"Index metadata not found at {self._path}")
        self._data = json.loads(self._path.read_text())
        return self._data

    def validate_provider(self, provider: str, model: str, dimension: int) -> None:
        """Raise if the configured provider/model doesn't match the stored index."""
        meta = self.read()
        if meta["embedding_provider"] != provider or meta["embedding_dimension"] != dimension:
            raise ValueError(
                f"Provider mismatch: index was built with "
                f"{meta['embedding_provider']} ({meta['embedding_dimension']}d) "
                f"but current config is {provider} ({dimension}d). "
                f"Re-run `gritter index` to rebuild the index."
            )
```

- [ ] **Step 4: Commit**

```bash
git add gritter/storage/
git commit -m "feat: add VectorStore (ChromaDB), BM25Store, and IndexMeta storage layer"
```

---

## Task 11: Indexing Pipeline

**Files:**
- Create: `gritter/indexing/pipeline.py`

- [ ] **Step 1: Implement `gritter/indexing/pipeline.py`**

```python
from __future__ import annotations
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TextColumn

from gritter.models.chunk import CodeChunk
from gritter.models.config import GritterConfig
from gritter.indexing.discovery import discover_files
from gritter.indexing.chunker import chunk_file
from gritter.providers.embeddings import EmbeddingProvider
from gritter.storage.vector_store import VectorStore
from gritter.storage.bm25_store import BM25Store
from gritter.storage.index_meta import IndexMeta


def run_indexing_pipeline(
    root: Path,
    index_name: str,
    config: GritterConfig,
    embed_provider: EmbeddingProvider,
) -> dict:
    """Full indexing pipeline: discover → chunk → embed → store.

    Returns a summary dict with file_count, chunk_count, languages.
    """
    index_dir = config.index_dir(index_name)
    index_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=False,
    ) as progress:
        # Stage 1: Discovery
        discover_task = progress.add_task("Discovering files...", total=None)
        file_pairs = discover_files(root, config.index.exclude_globs)
        progress.update(discover_task, total=1, completed=1,
                        description=f"Found {len(file_pairs)} files")

        # Stage 2: Chunking
        chunk_task = progress.add_task("Chunking...", total=len(file_pairs))
        all_chunks: list[CodeChunk] = []
        languages_seen: set[str] = set()

        for file_path, language in file_pairs:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                rel_path = str(file_path.relative_to(root))
                chunks = chunk_file(
                    content, rel_path, language,
                    min_tokens=config.index.chunk_min_tokens,
                    max_tokens=config.index.chunk_max_tokens,
                    overlap_tokens=config.index.chunk_overlap_tokens,
                )
                all_chunks.extend(chunks)
                languages_seen.add(language)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Skipping %s: %s", file_path, exc)
            progress.advance(chunk_task)

        progress.update(chunk_task, description=f"Chunked → {len(all_chunks)} chunks")

        # Stage 3: Embedding (batched)
        batch_size = 64
        embed_task = progress.add_task("Embedding...", total=len(all_chunks))
        all_embeddings: list[list[float]] = []

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c.embedding_text() for c in batch]
            embeddings = embed_provider.embed(texts)
            all_embeddings.extend(embeddings)
            progress.advance(embed_task, advance=len(batch))

        # Stage 4: Storage
        store_task = progress.add_task("Storing...", total=None)

        vector_store = VectorStore(index_dir)
        vector_store.add(all_chunks, all_embeddings)

        bm25_store = BM25Store(index_dir)
        bm25_store.build(all_chunks)

        meta = IndexMeta(index_dir)
        meta.write(
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model or _default_model(config.embedding.provider),
            embedding_dimension=embed_provider.dimension,
            file_count=len(file_pairs),
            chunk_count=len(all_chunks),
            languages=list(languages_seen),
        )

        progress.update(store_task, total=1, completed=1, description="Stored ✓")

    return {
        "file_count": len(file_pairs),
        "chunk_count": len(all_chunks),
        "languages": sorted(languages_seen),
    }


def _default_model(provider: str) -> str:
    return {
        "voyage": "voyage-code-3",
        "openai": "text-embedding-3-small",
        "local": "nomic-ai/nomic-embed-text-v1",
    }.get(provider, "unknown")
```

- [ ] **Step 2: Commit**

```bash
git add gritter/indexing/pipeline.py
git commit -m "feat: add indexing pipeline (discover → chunk → embed → store)"
```

---

## Task 12: `gritter index` and `gritter status` CLI

**Files:**
- Create: `gritter/cli/main.py`
- Create: `gritter/cli/index_cmd.py`
- Create: `gritter/cli/status_cmd.py`
- Create: `gritter/utils/display.py`

- [ ] **Step 1: Implement `gritter/utils/display.py`**

```python
from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

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
```

- [ ] **Step 2: Implement `gritter/cli/index_cmd.py`**

```python
from __future__ import annotations
from pathlib import Path
import typer
from gritter.models.config import GritterConfig
from gritter.providers.embeddings import get_embedding_provider
from gritter.indexing.pipeline import run_indexing_pipeline
from gritter.utils.display import console, print_success, print_error

app = typer.Typer()


def index(
    path: Path = typer.Argument(..., help="Path to the codebase to index"),
    name: str = typer.Option(None, "--name", "-n", help="Index name (defaults to directory name)"),
) -> None:
    """Index a codebase for querying."""
    root = path.resolve()
    if not root.is_dir():
        print_error(f"{path} is not a directory.")
        raise typer.Exit(1)

    config = GritterConfig()
    index_name = name or root.name

    try:
        embed_provider = get_embedding_provider(
            config.embedding.provider, config.embedding.model
        )
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    console.print(f"\n[bold]Indexing[/bold] [cyan]{root}[/cyan] as [bold]{index_name}[/bold]\n")

    try:
        summary = run_indexing_pipeline(root, index_name, config, embed_provider)
    except Exception as exc:
        print_error(f"Indexing failed: {exc}")
        raise typer.Exit(1)

    print_success(
        f"Index [bold]{index_name}[/bold] ready — "
        f"{summary['file_count']} files, {summary['chunk_count']} chunks, "
        f"languages: {', '.join(summary['languages'])}"
    )
```

- [ ] **Step 3: Implement `gritter/cli/status_cmd.py`**

```python
from __future__ import annotations
import typer
from gritter.models.config import GritterConfig
from gritter.storage.index_meta import IndexMeta
from gritter.utils.display import console, make_table, print_error


def status(
    name: str = typer.Argument("default", help="Index name to inspect"),
) -> None:
    """Show statistics for a stored index."""
    config = GritterConfig()
    index_dir = config.index_dir(name)

    try:
        meta = IndexMeta(index_dir)
        data = meta.read()
    except FileNotFoundError:
        print_error(
            f"No index named [bold]{name}[/bold] found. "
            f"Run `gritter index <path>` first."
        )
        raise typer.Exit(1)

    rows = [
        ("Index name", name),
        ("Files", str(data["file_count"])),
        ("Chunks", str(data["chunk_count"])),
        ("Languages", ", ".join(data["languages"])),
        ("Embedding provider", data["embedding_provider"]),
        ("Embedding model", data["embedding_model"]),
        ("Indexed at", data["indexed_at"]),
    ]
    console.print(make_table(f"Index: {name}", rows))
```

- [ ] **Step 4: Implement `gritter/cli/main.py`**

```python
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
```

- [ ] **Step 5: Smoke test the CLI**

```bash
gritter --help
```

Expected: help text showing `index` and `status` commands.

- [ ] **Step 6: Commit**

```bash
git add gritter/cli/ gritter/utils/display.py
git commit -m "feat: add gritter index and gritter status CLI commands"
```

---

## Task 13: Retrieval Pipeline

**Files:**
- Create: `gritter/retrieval/dense.py`
- Create: `gritter/retrieval/sparse.py`
- Create: `gritter/retrieval/fusion.py`
- Create: `gritter/retrieval/reranker.py`
- Create: `gritter/retrieval/hybrid.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write retrieval tests**

```python
# tests/test_retrieval.py
import random
from pathlib import Path

import pytest

from gritter.models.chunk import CodeChunk
from gritter.models.query import RetrievalResult
from gritter.retrieval.fusion import reciprocal_rank_fusion
from gritter.retrieval.reranker import NoOpReranker


def make_chunk(chunk_id: str, content: str = "some code") -> CodeChunk:
    c = CodeChunk(
        content=content,
        file_path="src/foo.py",
        language="python",
        symbol_name=None,
        symbol_type="function",
        start_line=1,
        end_line=5,
    )
    object.__setattr__(c, "chunk_id", chunk_id)
    return c


def make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(chunk=make_chunk(chunk_id), score=score)


class TestRRF:
    def test_chunk_in_both_lists_scores_higher(self):
        dense = [make_result(f"chunk_{i}", 1.0 / (i + 1)) for i in range(5)]
        sparse = [make_result(f"chunk_{i}", 1.0 / (i + 1)) for i in range(5)]
        # chunk_0 is rank 0 in both — should have highest RRF score
        fused = reciprocal_rank_fusion(dense, sparse)
        assert fused[0].chunk.chunk_id == "chunk_0"

    def test_chunk_only_in_one_list_still_included(self):
        dense = [make_result("exclusive_dense", 1.0)]
        sparse = [make_result("exclusive_sparse", 1.0)]
        fused = reciprocal_rank_fusion(dense, sparse)
        ids = {r.chunk.chunk_id for r in fused}
        assert "exclusive_dense" in ids
        assert "exclusive_sparse" in ids

    def test_rrf_score_is_sum_of_reciprocals(self):
        k = 60
        # chunk_0 is rank 0 in dense, not in sparse
        dense = [make_result("chunk_0", 1.0), make_result("chunk_1", 0.5)]
        sparse = [make_result("chunk_1", 1.0)]
        fused = reciprocal_rank_fusion(dense, sparse, k=k)
        scores = {r.chunk.chunk_id: r.score for r in fused}
        # chunk_1: rank 1 in dense + rank 0 in sparse
        expected_chunk1 = 1 / (k + 1) + 1 / (k + 0)
        assert abs(scores["chunk_1"] - expected_chunk1) < 1e-9

    def test_results_sorted_descending(self):
        dense = [make_result(f"d{i}", 1.0) for i in range(10)]
        sparse = [make_result(f"s{i}", 1.0) for i in range(10)]
        fused = reciprocal_rank_fusion(dense, sparse)
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)


class TestNoOpReranker:
    def test_returns_results_unchanged(self):
        results = [make_result(f"chunk_{i}", float(i)) for i in range(5)]
        reranker = NoOpReranker()
        reranked = reranker.rerank("some query", results)
        assert reranked == results
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retrieval.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `gritter/retrieval/fusion.py`**

```python
from __future__ import annotations
from gritter.models.query import RetrievalResult


def reciprocal_rank_fusion(
    dense: list[RetrievalResult],
    sparse: list[RetrievalResult],
    k: int = 60,
) -> list[RetrievalResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion (Cormack et al., 2009).

    RRF_score(chunk) = Σ 1 / (k + rank_in_list)
    summed across all lists the chunk appears in.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(dense):
        cid = result.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunks[cid] = result

    for rank, result in enumerate(sparse):
        cid = result.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunks[cid] = result

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        RetrievalResult(chunk=chunks[cid].chunk, score=scores[cid])
        for cid in sorted_ids
    ]
```

- [ ] **Step 4: Implement `gritter/retrieval/reranker.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from gritter.models.query import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Re-order results by relevance to query. Returns a reordered list."""


class NoOpReranker(Reranker):
    """Identity reranker — returns results unchanged. Placeholder for v1.1."""

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return results
```

- [ ] **Step 5: Implement `gritter/retrieval/dense.py`**

```python
from __future__ import annotations
from gritter.models.query import RetrievalResult
from gritter.storage.vector_store import VectorStore


def dense_search(
    query_embedding: list[float],
    store: VectorStore,
    k: int = 20,
) -> list[RetrievalResult]:
    """Query ChromaDB and return top-k results ordered by cosine similarity."""
    results = store.query(query_embedding, k=k)
    return [RetrievalResult(chunk=chunk, score=1.0 - dist) for chunk, dist in results]
```

- [ ] **Step 6: Implement `gritter/retrieval/sparse.py`**

```python
from __future__ import annotations
from gritter.models.query import RetrievalResult
from gritter.storage.bm25_store import BM25Store


def sparse_search(
    query: str,
    store: BM25Store,
    k: int = 20,
) -> list[RetrievalResult]:
    """Query BM25 index and return top-k results ordered by BM25 score."""
    results = store.query(query, k=k)
    return [RetrievalResult(chunk=chunk, score=score) for chunk, score in results]
```

- [ ] **Step 7: Implement `gritter/retrieval/hybrid.py`**

```python
from __future__ import annotations
from pathlib import Path

from gritter.models.config import GritterConfig
from gritter.models.query import RetrievalResult
from gritter.providers.embeddings import EmbeddingProvider, get_embedding_provider
from gritter.retrieval.dense import dense_search
from gritter.retrieval.sparse import sparse_search
from gritter.retrieval.fusion import reciprocal_rank_fusion
from gritter.retrieval.reranker import Reranker, NoOpReranker
from gritter.storage.vector_store import VectorStore
from gritter.storage.bm25_store import BM25Store
from gritter.storage.index_meta import IndexMeta


class HybridRetriever:
    """Orchestrates dense + sparse retrieval + RRF + reranking."""

    def __init__(
        self,
        index_dir: Path,
        embed_provider: EmbeddingProvider,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
        top_k: int = 5,
    ) -> None:
        self._embed_provider = embed_provider
        self._reranker = reranker or NoOpReranker()
        self._candidate_k = candidate_k
        self._top_k = top_k

        self._vector_store = VectorStore(index_dir)
        self._bm25_store = BM25Store(index_dir)
        self._bm25_store.load()

    def search(self, query: str) -> list[RetrievalResult]:
        """Return top_k results for a natural language query."""
        query_vec = self._embed_provider.embed_one(query)

        dense_results = dense_search(query_vec, self._vector_store, k=self._candidate_k)
        sparse_results = sparse_search(query, self._bm25_store, k=self._candidate_k)

        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        candidates = fused[:self._candidate_k]

        reranked = self._reranker.rerank(query, candidates)
        return reranked[:self._top_k]

    @classmethod
    def from_config(cls, index_name: str, config: GritterConfig) -> "HybridRetriever":
        """Build a HybridRetriever from GritterConfig, validating provider compatibility."""
        index_dir = config.index_dir(index_name)

        meta = IndexMeta(index_dir)
        embed_provider = get_embedding_provider(
            config.embedding.provider, config.embedding.model
        )
        meta.validate_provider(
            config.embedding.provider,
            config.embedding.model or "",
            embed_provider.dimension,
        )

        return cls(
            index_dir=index_dir,
            embed_provider=embed_provider,
            candidate_k=config.retrieval.candidate_k,
            top_k=config.retrieval.top_k,
        )
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
pytest tests/test_retrieval.py -v
```

Expected: all PASSED

- [ ] **Step 9: Commit**

```bash
git add gritter/retrieval/ tests/test_retrieval.py
git commit -m "feat: add hybrid retrieval (dense + BM25 + RRF + reranker stub)"
```

---

## Task 14: LLM Providers

**Files:**
- Create: `gritter/providers/llm.py`

- [ ] **Step 1: Implement `gritter/providers/llm.py`**

```python
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


class LLMProvider(ABC):
    @abstractmethod
    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        """Stream response tokens for the given system prompt and message history."""


class ClaudeProvider(LLMProvider):
    """Anthropic Claude — default provider."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is required for Claude."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        ) as stream:
            for text in stream.text_stream:
                yield text


class OpenAIProvider(LLMProvider):
    """OpenAI GPT models."""

    def __init__(self, model: str = "gpt-4o") -> None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for OpenAI."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OllamaProvider(LLMProvider):
    """Local Ollama — no API key required."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        from openai import OpenAI
        self._client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        self._model = model

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def get_llm_provider(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    if provider == "claude":
        return ClaudeProvider(model or "claude-sonnet-4-6")
    elif provider == "openai":
        return OpenAIProvider(model or "gpt-4o")
    elif provider == "ollama":
        return OllamaProvider(model or "llama3", base_url or "http://localhost:11434")
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
```

- [ ] **Step 2: Commit**

```bash
git add gritter/providers/llm.py
git commit -m "feat: add LLMProvider ABC with Claude, OpenAI, and Ollama implementations"
```

---

## Task 15: Generation Pipeline

**Files:**
- Create: `gritter/generation/prompts.py`
- Create: `gritter/generation/citations.py`
- Create: `gritter/generation/generator.py`
- Create: `tests/test_generation.py`

- [ ] **Step 1: Write generation tests**

```python
# tests/test_generation.py
from gritter.generation.citations import extract_citations, format_sources
from gritter.generation.prompts import build_context_block
from gritter.models.chunk import CodeChunk
from gritter.models.query import RetrievalResult


def make_result(path: str, start: int, end: int, content: str) -> RetrievalResult:
    chunk = CodeChunk(
        content=content,
        file_path=path,
        language="python",
        symbol_name="foo",
        symbol_type="function",
        start_line=start,
        end_line=end,
    )
    return RetrievalResult(chunk=chunk, score=0.9)


class TestCitations:
    def test_extracts_file_line_references(self):
        text = "The auth logic is in `src/auth/jwt.py:L15-45` and `src/middleware.py:L1-10`."
        citations = extract_citations(text)
        assert "src/auth/jwt.py:L15-45" in citations
        assert "src/middleware.py:L1-10" in citations

    def test_deduplicates_citations(self):
        text = "See `src/foo.py:L1-10` and also `src/foo.py:L1-10`."
        citations = extract_citations(text)
        assert citations.count("src/foo.py:L1-10") == 1

    def test_format_sources_produces_bullet_list(self):
        citations = ["src/auth.py:L1-10", "src/utils.py:L5-20"]
        formatted = format_sources(citations)
        assert "src/auth.py:L1-10" in formatted
        assert "src/utils.py:L5-20" in formatted


class TestContextBlock:
    def test_block_includes_file_path_and_lines(self):
        result = make_result("src/foo.py", 10, 20, "def foo(): pass")
        block = build_context_block([result])
        assert "src/foo.py" in block
        assert "lines 10-20" in block
        assert "def foo(): pass" in block

    def test_block_includes_symbol_name(self):
        result = make_result("src/foo.py", 1, 5, "def foo(): pass")
        block = build_context_block([result])
        assert "foo" in block
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_generation.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `gritter/generation/prompts.py`**

```python
from __future__ import annotations
from gritter.models.query import RetrievalResult

SYSTEM_PROMPT = """\
You are a code assistant answering questions about a specific codebase.

Rules:
1. Answer ONLY from the provided code context below.
2. For every factual claim, cite the source file and lines as `path/to/file.py:L10-25`.
3. If the context does not contain enough information to answer, say exactly:
   "I don't know from the provided context."
4. Do not invent file paths, function names, or behavior not shown in the context.
5. Format code references inline using backticks.
"""


def build_context_block(results: list[RetrievalResult]) -> str:
    """Build the context string to inject into the user prompt."""
    sections = []
    for result in results:
        chunk = result.chunk
        header_parts = [f"File: {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line}"]
        if chunk.symbol_name:
            header_parts.append(f", {chunk.symbol_type}: {chunk.symbol_name}")
        header_parts.append(")")
        header = "".join(header_parts)
        sections.append(f"--- {header} ---\n{chunk.content}")
    return "\n\n".join(sections)


def build_user_prompt(query: str, results: list[RetrievalResult]) -> str:
    context = build_context_block(results)
    return f"Context:\n\n{context}\n\nQuestion: {query}"
```

- [ ] **Step 4: Implement `gritter/generation/citations.py`**

```python
from __future__ import annotations
import re

# Matches patterns like `src/foo.py:L10-25` or `src/foo.py:L10`
_CITATION_RE = re.compile(r"`([^`]+\.(?:py|ts|tsx|rs|js|jsx|go|java|cpp|c|h):L\d+(?:-\d+)?)`")


def extract_citations(text: str) -> list[str]:
    """Extract unique file:line citations from generated text."""
    seen: dict[str, None] = {}
    for match in _CITATION_RE.finditer(text):
        ref = match.group(1)
        if ref not in seen:
            seen[ref] = None
    return list(seen)


def format_sources(citations: list[str]) -> str:
    """Format a list of citations as a bulleted Sources block."""
    if not citations:
        return ""
    lines = ["", "Sources:"] + [f"  • {c}" for c in citations]
    return "\n".join(lines)
```

- [ ] **Step 5: Implement `gritter/generation/generator.py`**

```python
from __future__ import annotations
from collections.abc import Iterator

from gritter.generation.citations import extract_citations, format_sources
from gritter.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from gritter.models.query import RetrievalResult
from gritter.providers.llm import LLMProvider, Message


class Generator:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider
        self._history: list[Message] = []

    def stream(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> Iterator[str]:
        """Stream the LLM response token by token, then yield citations block."""
        user_prompt = build_user_prompt(query, results)
        self._history.append(Message(role="user", content=user_prompt))

        full_response = ""
        for token in self._llm.stream(SYSTEM_PROMPT, self._history):
            full_response += token
            yield token

        self._history.append(Message(role="assistant", content=full_response))

        citations = extract_citations(full_response)
        if citations:
            yield format_sources(citations)

    def reset(self) -> None:
        """Clear conversation history (for starting a new chat session)."""
        self._history = []
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_generation.py -v
```

Expected: all PASSED

- [ ] **Step 7: Commit**

```bash
git add gritter/generation/ tests/test_generation.py
git commit -m "feat: add generation pipeline (prompts, citations, streaming generator)"
```

---

## Task 16: `gritter ask` and `gritter chat` CLI

**Files:**
- Create: `gritter/cli/ask_cmd.py`
- Create: `gritter/cli/chat_cmd.py`
- Modify: `gritter/cli/main.py`

- [ ] **Step 1: Implement `gritter/cli/ask_cmd.py`**

```python
from __future__ import annotations
import typer
from rich.live import Live
from rich.text import Text
from gritter.models.config import GritterConfig
from gritter.providers.llm import get_llm_provider
from gritter.retrieval.hybrid import HybridRetriever
from gritter.generation.generator import Generator
from gritter.utils.display import console, print_error


def ask(
    question: str = typer.Argument(..., help="Natural language question about the codebase"),
    name: str = typer.Option("default", "--name", "-n", help="Index name to query"),
) -> None:
    """Ask a one-off question about an indexed codebase."""
    config = GritterConfig()

    try:
        retriever = HybridRetriever.from_config(name, config)
    except FileNotFoundError:
        print_error(f"No index named [bold]{name}[/bold]. Run `gritter index <path>` first.")
        raise typer.Exit(1)
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(1)
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    try:
        llm = get_llm_provider(config.llm.provider, config.llm.model, config.llm.base_url)
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    results = retriever.search(question)
    if not results:
        print_error("No relevant code found in the index.")
        raise typer.Exit(1)

    generator = Generator(llm)
    console.print()

    accumulated = ""
    with Live(Text(accumulated), console=console, refresh_per_second=20) as live:
        for token in generator.stream(question, results):
            accumulated += token
            live.update(Text(accumulated))

    console.print()
```

- [ ] **Step 2: Implement `gritter/cli/chat_cmd.py`**

```python
from __future__ import annotations
import typer
from rich.live import Live
from rich.text import Text
from gritter.models.config import GritterConfig
from gritter.providers.llm import get_llm_provider
from gritter.retrieval.hybrid import HybridRetriever
from gritter.generation.generator import Generator
from gritter.utils.display import console, print_error


def chat(
    name: str = typer.Option("default", "--name", "-n", help="Index name to query"),
) -> None:
    """Start an interactive chat session with an indexed codebase."""
    config = GritterConfig()

    try:
        retriever = HybridRetriever.from_config(name, config)
    except FileNotFoundError:
        print_error(f"No index named [bold]{name}[/bold]. Run `gritter index <path>` first.")
        raise typer.Exit(1)
    except (ValueError, EnvironmentError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    try:
        llm = get_llm_provider(config.llm.provider, config.llm.model, config.llm.base_url)
    except EnvironmentError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    generator = Generator(llm)
    console.print(f"\n[bold cyan]gritter chat[/bold cyan] — index: [bold]{name}[/bold]")
    console.print("Type your question and press Enter. Type [bold]exit[/bold] or Ctrl-C to quit.\n")

    while True:
        try:
            query = console.input("[bold green]>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        results = retriever.search(query)
        if not results:
            console.print("[dim]No relevant code found.[/dim]\n")
            continue

        accumulated = ""
        with Live(Text(accumulated), console=console, refresh_per_second=20) as live:
            for token in generator.stream(query, results):
                accumulated += token
                live.update(Text(accumulated))

        console.print()
```

- [ ] **Step 3: Update `gritter/cli/main.py`**

```python
from __future__ import annotations
import typer
from gritter.cli.index_cmd import index
from gritter.cli.status_cmd import status
from gritter.cli.ask_cmd import ask
from gritter.cli.chat_cmd import chat

app = typer.Typer(
    name="gritter",
    help="Query any codebase with natural language.",
    add_completion=False,
)

app.command("index")(index)
app.command("status")(status)
app.command("ask")(ask)
app.command("chat")(chat)

if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Smoke test**

```bash
gritter --help
```

Expected: all four commands listed (`index`, `status`, `ask`, `chat`).

- [ ] **Step 5: Commit**

```bash
git add gritter/cli/ask_cmd.py gritter/cli/chat_cmd.py gritter/cli/main.py
git commit -m "feat: add gritter ask and gritter chat CLI commands"
```

---

## Task 17: Config CLI Commands

**Files:**
- Create: `gritter/cli/config_cmd.py`
- Modify: `gritter/cli/main.py`

- [ ] **Step 1: Implement `gritter/cli/config_cmd.py`**

```python
from __future__ import annotations
from pathlib import Path
import typer
import tomli_w
from gritter.models.config import GritterConfig
from gritter.utils.display import console, print_success, print_error

app = typer.Typer(help="Manage gritter configuration.")

_USER_CONFIG = Path.home() / ".config" / "gritter" / "config.toml"


@app.command("show")
def config_show() -> None:
    """Print the current effective configuration."""
    config = GritterConfig()
    console.print_json(config.model_dump_json(indent=2))


@app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dot-separated config key, e.g. embedding.provider"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Write a config value to the user-level config file (~/.config/gritter/config.toml)."""
    _USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config file if present
    existing: dict = {}
    if _USER_CONFIG.exists():
        import tomllib
        existing = tomllib.loads(_USER_CONFIG.read_text())

    # Navigate/create nested dict from dot-separated key
    parts = key.split(".")
    node = existing
    for part in parts[:-1]:
        node = node.setdefault(part, {})

    # Coerce value type: int if numeric, bool if true/false, else string
    coerced: int | bool | str
    if value.isdigit():
        coerced = int(value)
    elif value.lower() in ("true", "false"):
        coerced = value.lower() == "true"
    else:
        coerced = value

    node[parts[-1]] = coerced
    _USER_CONFIG.write_bytes(tomli_w.dumps(existing).encode())
    print_success(f"Set [bold]{key}[/bold] = [cyan]{value}[/cyan] in {_USER_CONFIG}")
```

- [ ] **Step 2: Update `gritter/cli/main.py`**

```python
from __future__ import annotations
import typer
from gritter.cli.index_cmd import index
from gritter.cli.status_cmd import status
from gritter.cli.ask_cmd import ask
from gritter.cli.chat_cmd import chat
from gritter.cli import config_cmd

app = typer.Typer(
    name="gritter",
    help="Query any codebase with natural language.",
    add_completion=False,
)

app.command("index")(index)
app.command("status")(status)
app.command("ask")(ask)
app.command("chat")(chat)
app.add_typer(config_cmd.app, name="config")

if __name__ == "__main__":
    app()
```

- [ ] **Step 3: Smoke test config commands**

```bash
gritter config show
gritter config set retrieval.top_k 8
gritter config show
```

Expected: `top_k` changes to 8 in output.

- [ ] **Step 4: Commit**

```bash
git add gritter/cli/config_cmd.py gritter/cli/main.py
git commit -m "feat: add gritter config set/show commands"
```

---

## Task 18: End-to-End Smoke Test

**Files:**
- Create: `tests/test_e2e.py`

This test is gated behind `GRITTER_RUN_E2E_TESTS=1` and requires real API keys.

- [ ] **Step 1: Create `tests/test_e2e.py`**

```python
# tests/test_e2e.py
"""End-to-end tests. Gated behind GRITTER_RUN_E2E_TESTS=1.

Requires:
  - VOYAGE_API_KEY or OPENAI_API_KEY in environment
  - ANTHROPIC_API_KEY in environment
"""
import os
import pytest
from pathlib import Path
from gritter.models.config import GritterConfig, EmbeddingConfig, LLMConfig
from gritter.providers.embeddings import get_embedding_provider
from gritter.providers.llm import get_llm_provider
from gritter.indexing.pipeline import run_indexing_pipeline
from gritter.retrieval.hybrid import HybridRetriever
from gritter.generation.generator import Generator
from gritter.storage.index_meta import IndexMeta

FIXTURES_DIR = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not os.environ.get("GRITTER_RUN_E2E_TESTS"),
    reason="Set GRITTER_RUN_E2E_TESTS=1 to run end-to-end tests",
)


@pytest.fixture
def e2e_config(tmp_path):
    """Config that stores the index in a temp directory."""
    config = GritterConfig(data_dir=tmp_path)
    config.embedding.provider = os.environ.get("GRITTER_TEST_EMBED_PROVIDER", "openai")
    config.llm.provider = os.environ.get("GRITTER_TEST_LLM_PROVIDER", "claude")
    return config


def test_index_and_query_python_fixture(e2e_config):
    root = FIXTURES_DIR / "python_project"
    embed_provider = get_embedding_provider(e2e_config.embedding.provider)
    summary = run_indexing_pipeline(root, "test_python", e2e_config, embed_provider)

    assert summary["file_count"] >= 1
    assert summary["chunk_count"] >= 1
    assert "python" in summary["languages"]

    meta = IndexMeta(e2e_config.index_dir("test_python"))
    data = meta.read()
    assert data["chunk_count"] == summary["chunk_count"]

    retriever = HybridRetriever.from_config("test_python", e2e_config)
    results = retriever.search("how does add work?")
    assert len(results) >= 1
    assert any("add" in r.chunk.content for r in results)


def test_generator_returns_citation(e2e_config):
    root = FIXTURES_DIR / "python_project"
    embed_provider = get_embedding_provider(e2e_config.embedding.provider)
    run_indexing_pipeline(root, "test_gen", e2e_config, embed_provider)

    retriever = HybridRetriever.from_config("test_gen", e2e_config)
    results = retriever.search("what does subtract do?")

    llm = get_llm_provider(e2e_config.llm.provider)
    generator = Generator(llm)

    full_response = "".join(generator.stream("what does subtract do?", results))
    assert len(full_response) > 20
    # Response should reference some file
    assert ".py" in full_response
```

- [ ] **Step 2: Verify the test is skipped without the env var**

```bash
pytest tests/test_e2e.py -v
```

Expected: 2 SKIPPED

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: add gated end-to-end smoke tests"
```

---

## Task 19: PyPI Packaging Prep

**Files:**
- Create: `README.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v --ignore=tests/test_e2e.py
```

Expected: all non-E2E tests PASSED

- [ ] **Step 2: Create a minimal `README.md`**

```markdown
# gritter

Query any codebase with natural language using RAG.

## Installation

```bash
pip install gritter
```

## Quick Start

```bash
# Index a codebase
gritter index /path/to/your/repo

# Ask a question
gritter ask "how does authentication work?"

# Interactive chat
gritter chat
```

## Configuration

Set your API keys:

```bash
export VOYAGE_API_KEY=...      # or OPENAI_API_KEY for embeddings
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY for generation
```

Configure defaults:

```bash
gritter config set embedding.provider openai
gritter config set llm.provider claude
gritter config show
```

## Supported Languages (AST chunking)

- Python
- TypeScript / JavaScript
- Rust

All other languages use heuristic (blank-line) chunking.

## Provider Support

| Component  | Providers |
|---|---|
| Embeddings | Voyage (`voyage-code-3`), OpenAI (`text-embedding-3-small`), Local (sentence-transformers) |
| Generation | Claude (default), OpenAI, Ollama |
```

- [ ] **Step 3: Add classifiers and metadata to `pyproject.toml`**

Add to the `[project]` section:

```toml
license = {text = "MIT"}
keywords = ["rag", "llm", "code", "search", "cli"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries",
]
```

- [ ] **Step 4: Build and verify the package**

```bash
pip install build
python -m build
ls dist/
```

Expected: a `.whl` and `.tar.gz` file in `dist/`.

- [ ] **Step 5: Verify the installed CLI works**

```bash
pip install dist/gritter-0.1.0-*.whl --force-reinstall
gritter --help
```

Expected: help text with all commands listed.

- [ ] **Step 6: Final commit**

```bash
git add README.md pyproject.toml dist/
git commit -m "chore: add README, package metadata, and built distribution"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] AST chunking for Python, TypeScript, Rust — Tasks 6-7
- [x] Heuristic fallback for other languages — Task 5
- [x] Multi-provider embeddings (Voyage, OpenAI, local) — Task 9
- [x] ChromaDB vector store — Task 10
- [x] BM25 index — Task 10
- [x] Index metadata with provider validation — Task 10
- [x] Hybrid retrieval: dense + BM25 + RRF — Task 13
- [x] Reranker ABC + NoOpReranker stub — Task 13
- [x] Multi-provider LLMs (Claude, OpenAI, Ollama) — Task 14
- [x] Prompt construction with citations — Task 15
- [x] Streaming generation — Task 15
- [x] `gritter index` — Task 12
- [x] `gritter status` — Task 12
- [x] `gritter ask` — Task 16
- [x] `gritter chat` — Task 16
- [x] `gritter config set/show` — Task 17
- [x] Configuration via TOML file + env vars — Task 8
- [x] Gated E2E tests — Task 18
- [x] PyPI packaging — Task 19
- [x] Token guardrails (min 50, max 512, 20-token overlap) — Tasks 5, 6
- [x] Syntax error fallback to heuristic — Task 7 (`chunk_file` dispatcher)
- [x] `imports` field prepended to embedding text — Task 2 (`embedding_text()`)

**Type consistency check:**
- `chunk_file()` in `chunker.py` → called in `pipeline.py` with same signature ✓
- `CodeChunk.chunk_id` used as ChromaDB `id` and BM25 positional key ✓
- `RetrievalResult` flows from retrieval → generator unchanged ✓
- `EmbeddingProvider.embed_one()` used in `hybrid.py` ✓
- `LLMProvider.stream()` used in `generator.py` ✓
- `HybridRetriever.from_config()` called in `ask_cmd.py` and `chat_cmd.py` ✓

**No placeholders found.**
