# Gritter RAG CLI — Design Spec

**Date:** 2026-04-10
**Status:** Approved

---

## Elevator Pitch

A command-line tool that ingests any codebase, builds a searchable index using code-aware chunking and embeddings, and lets you ask natural language questions about the code — answered with cited file and line number references.

---

## Goals

- Demonstrate production-grade RAG engineering: AST-based chunking, hybrid retrieval, reciprocal rank fusion, multi-provider embeddings and LLMs
- Function as a genuinely useful daily tool for querying unfamiliar codebases
- Publishable to PyPI (`pip install gritter`)

## Non-Goals (v1)

- Cross-encoder re-ranking (interface stubbed, implementation deferred to v1.1)
- Evaluation harness (`gritter eval`)
- Incremental indexing via git diff
- Multi-repo support
- Web UI

---

## Language Support

AST-based chunking (tree-sitter) for:
- **Python** — `function_definition`, `class_definition` (split into methods if >512 tokens), module-level code between definitions
- **TypeScript** — `function_declaration`, arrow functions assigned to variables, `class_declaration` (split into methods if large); JS shares the same grammar
- **Rust** — `fn`, `struct`, `enum`, `trait`, `impl` (split into methods if large); `macro_rules!` treated as atomic chunks

Heuristic fallback chunker for all other languages: blank-line-separated blocks respecting indentation depth, with character-limit splitting as a last resort.

---

## Architecture Overview

Three independent pipelines sharing a common data model:

**Indexing pipeline** (batch, run once): file discovery → language detection → chunking → embedding → storage

**Retrieval pipeline** (per query): embed query → dense search → BM25 search → RRF fusion → reranker (no-op in v1) → top-k results

**Generation pipeline** (per query): retrieval results → prompt construction → LLM stream → citation rendering

The CLI layer is thin: each command is a lightweight orchestrator wiring these pipelines together. No business logic in the CLI layer.

All pipelines communicate through `CodeChunk` and `RetrievalResult` types defined in `models/`. Configuration flows in via a Pydantic settings object loaded once at startup.

---

## Data Model

```python
@dataclass
class CodeChunk:
    content: str           # source code content
    file_path: str         # relative path from repo root
    language: str          # detected language
    symbol_name: str | None    # function/class name if applicable
    symbol_type: str | None    # "function", "class", "method", "module"
    start_line: int
    end_line: int
    imports: list[str]     # top-of-file imports prepended for embedding context
    chunk_id: str          # deterministic SHA256(path + content) for deduplication

@dataclass
class RetrievalResult:
    chunk: CodeChunk
    score: float           # RRF score
```

---

## Indexing Pipeline

### Stage 1: Discovery

Walk the directory tree and produce a filtered list of `(path, language)` pairs.

- Language detection: extension-based lookup (`.py` → Python, `.ts`/`.tsx` → TypeScript, `.rs` → Rust)
- Excluded automatically: binaries, lockfiles (`package-lock.json`, `Cargo.lock`, `poetry.lock`), build artifacts (`dist/`, `target/`, `__pycache__/`), hidden directories
- User-configurable glob exclusions via config

Discovery is fast and reads no file content.

### Stage 2: Chunking

A `Chunker` dispatcher routes each file to the correct implementation:

**`ASTChunker`** (Python, TypeScript, Rust):
- Loads tree-sitter grammar for the language
- Walks syntax tree, extracts top-level definitions as chunks
- Applies chunk size guardrails: min 50 tokens (merge with adjacent chunk from same file), max 512 tokens (split with 20-token overlap)
- On tree-sitter parse error: logs warning, falls back to `HeuristicChunker` for that file, continues processing

**`HeuristicChunker`** (all other languages):
- Splits on blank-line-separated blocks
- Respects indentation depth to avoid splitting mid-block
- Falls back to character-limit splitting if a block still exceeds the token max

`imports` field: top-of-file import statements from the same file are prepended to every chunk from that file. This gives the embedding model context about what the chunk depends on.

### Stage 3: Embedding

```python
class EmbeddingProvider(ABC):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...
```

Implementations:
- `VoyageEmbedder` — `voyage-code-3`, recommended for code retrieval quality
- `OpenAIEmbedder` — `text-embedding-3-small`, general-purpose fallback
- `LocalEmbedder` — sentence-transformers, no API key required

Chunks are batched (default batch size: 64). Batching, retries with exponential backoff, and rate limit handling live inside each provider implementation.

### Stage 4: Storage

Two stores written in parallel:

**ChromaDB** — receives chunk content, embedding vector, and metadata (path, language, symbol name, lines). Collection name derived from index name so multiple repos can coexist.

**BM25 index** (`rank_bm25`) — receives tokenized chunk content. Persisted to disk as a pickle alongside a positional map so chunk IDs can be recovered from BM25 rank positions.

**Index metadata file** (JSON) — records: provider + model used, embedding dimension, number of files/chunks, languages seen, timestamp, config snapshot. Read by `gritter status`. Validates provider compatibility on subsequent queries: mismatched provider or dimension raises a clear error with an explicit re-index instruction.

Progress displayed via Rich progress bar (chunks processed / total with ETA).

---

## Retrieval Pipeline

### Step 1: Query Embedding

Query string embedded using the same `EmbeddingProvider` recorded in index metadata. Provider/dimension mismatch raises an error before any search.

### Step 2: Dual Retrieval

Both searches run independently against the same chunk corpus:

- **Dense search** — ChromaDB `collection.query(query_embeddings=[...], n_results=20)`. Results ordered by cosine similarity.
- **Sparse search** — BM25 index loaded from disk, queried with the raw query string. Results ordered by BM25 score.

Top-20 candidates from each.

### Step 3: Reciprocal Rank Fusion

```
RRF_score(chunk) = 1/(60 + rank_in_dense) + 1/(60 + rank_in_sparse)
```

- Chunks appearing in both lists receive a double contribution
- Chunks only in one list still score — not discarded
- k=60 is the standard RRF constant (Cormack et al., 2009)
- Sort descending, take top-10

### Step 4: Reranker Interface

```python
class Reranker(ABC):
    def rerank(self, query: str, chunks: list[CodeChunk]) -> list[CodeChunk]: ...

class NoOpReranker(Reranker):
    def rerank(self, query, chunks): return chunks
```

`NoOpReranker` is the default in v1. A `CrossEncoderReranker` implementation drops in here with zero changes to surrounding code.

### Step 5: Return

Top-5 `RetrievalResult` objects returned to generation pipeline (`top_k` configurable). RRF score available for display in debug mode.

---

## Generation Pipeline

### Prompt Construction

Context block format for each chunk:

```
--- File: src/auth/jwt.py (lines 15-45, function: validate_token) ---
[chunk content]
```

System prompt instructs the model to:
1. Answer only from provided context
2. Cite every claim as `path/to/file.py:L15-45`
3. Say "I don't know from the provided context" when the answer isn't present (prevents hallucinated file paths)

### LLM Provider Interface

```python
class LLMProvider(ABC):
    def stream(self, system: str, messages: list[Message]) -> Iterator[str]: ...
```

Implementations:
- `ClaudeProvider` — Anthropic SDK, default
- `OpenAIProvider` — OpenAI SDK
- `OllamaProvider` — local inference, no API key required

Tokens streamed directly to terminal via Rich `Live` display as they arrive.

### Citation Rendering

After stream completes, file path references (`src/auth/jwt.py:L15-45`) are parsed from the response and printed as a deduplicated "Sources" block. Gives the user a clean list of files separate from the prose.

### Chat Mode

`gritter chat` maintains a `messages` list in memory across turns. Each turn re-runs retrieval against the new query — conversation history is passed to the LLM but does not influence retrieval. Multi-turn query rewriting is deferred to v2.

---

## Configuration

**Load priority:** CLI flags → environment variables → `.gritterrc` (TOML, project root or `~/.config/gritter/config.toml`) → defaults

**API keys:** read from environment variables only — never written to config files.

**`gritter config set <key> <value>`** writes to user-level config file.

Key config fields:
- `embedding.provider` — `voyage` | `openai` | `local`
- `embedding.model` — model name override
- `llm.provider` — `claude` | `openai` | `ollama`
- `llm.model` — model name override
- `index.chunk_min_tokens` — default 50
- `index.chunk_max_tokens` — default 512
- `index.chunk_overlap_tokens` — default 20
- `index.exclude_globs` — list of glob patterns
- `retrieval.top_k` — default 5
- `retrieval.candidate_k` — default 20 (fed into RRF)

---

## CLI Commands

| Command | Description |
|---|---|
| `gritter index <path>` | Index a local directory. Runs full indexing pipeline with Rich progress bar. |
| `gritter ask "<question>"` | Single query. Streams answer, prints Sources block. |
| `gritter chat` | Interactive REPL for multi-turn queries. |
| `gritter status` | Reads index metadata. Prints table: files, chunks, languages, provider, last indexed. |
| `gritter config set <key> <value>` | Write a config value to user-level config file. |
| `gritter config show` | Print current effective configuration. |

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Missing API key | Clear message naming the expected env var |
| Provider/dimension mismatch on existing index | Error with explicit re-index instruction |
| File with syntax errors | Warning logged, file falls back to `HeuristicChunker`, indexing continues |
| LLM context window exceeded | Trim lowest-scoring chunks until prompt fits, warn user |
| ChromaDB/BM25 store missing or corrupt | Error with instruction to re-run `gritter index` |

---

## Project Structure

```
gritter/
├── cli/
│   ├── main.py             # Typer app, top-level commands
│   ├── index_cmd.py
│   ├── ask_cmd.py
│   ├── chat_cmd.py
│   ├── status_cmd.py
│   └── config_cmd.py
├── indexing/
│   ├── pipeline.py         # Orchestrates: discover → filter → chunk → embed → store
│   ├── discovery.py        # Walk tree, filter files, detect languages
│   ├── chunker.py          # Dispatcher: routes to AST or heuristic chunker
│   ├── ast_chunker.py      # Tree-sitter chunking for Python, TypeScript, Rust
│   └── heuristic_chunker.py
├── retrieval/
│   ├── hybrid.py           # Orchestrates dense + sparse + RRF + reranker
│   ├── dense.py            # ChromaDB query wrapper
│   ├── sparse.py           # BM25 index query wrapper
│   ├── fusion.py           # Reciprocal Rank Fusion
│   └── reranker.py         # Reranker ABC + NoOpReranker
├── generation/
│   ├── generator.py        # Prompt construction, LLM streaming
│   ├── prompts.py          # System and user prompt templates
│   └── citations.py        # Parse and format source citations
├── storage/
│   ├── vector_store.py     # ChromaDB wrapper
│   ├── bm25_store.py       # BM25 index persistence
│   └── index_meta.py       # Index metadata (stats, provider, timestamps)
├── providers/
│   ├── embeddings.py       # EmbeddingProvider ABC + Voyage, OpenAI, Local impls
│   └── llm.py              # LLMProvider ABC + Claude, OpenAI, Ollama impls
├── models/
│   ├── chunk.py            # CodeChunk dataclass
│   ├── query.py            # RetrievalResult dataclass
│   └── config.py           # Pydantic settings model
├── utils/
│   ├── languages.py        # Language detection, extension mapping
│   └── display.py          # Rich-based terminal formatting helpers
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── fixtures/
│       ├── python_project/
│       ├── typescript_project/
│       └── rust_project/
└── pyproject.toml
```

---

## Testing Strategy

**Unit tests (offline, no API calls):**
- Each language gets a fixture file with known structure
- Assert: correct chunk count, correct `symbol_name`/`symbol_type`, correct line boundaries, token guardrails applied, syntax error triggers fallback without crash

**Integration tests (mocked embeddings):**
- Pre-indexed fixture corpus with random vectors (tests retrieval machinery, not semantic quality)
- Assert: RRF produces different ordering than dense-only or sparse-only; `NoOpReranker` returns results unchanged

**End-to-end tests (real API, gated):**
- Gated behind `GRITTER_RUN_E2E_TESTS=1`
- Index a fixture project, run a query, assert non-empty response with at least one valid citation
- `gritter status` reflects correct counts post-index

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| CLI framework | Typer |
| AST parsing | tree-sitter (`py-tree-sitter`) |
| Vector store | ChromaDB |
| BM25 | rank_bm25 |
| Terminal UI | Rich |
| Config | pydantic-settings |
| Testing | pytest |
| Packaging | PyPI via hatchling |

---

## Development Milestones

**Sprint 1:** Chunking engine — tree-sitter setup, AST chunker for Python/TypeScript/Rust, heuristic fallback, file discovery, unit tests

**Sprint 2:** Indexing + storage — embedding providers, ChromaDB + BM25 storage, `gritter index` and `gritter status` commands, test on a real repo

**Sprint 3:** Retrieval + generation — dense/sparse retrieval, RRF fusion, reranker stub, prompt construction, LLM streaming, `gritter ask` and `gritter chat`

**Sprint 4:** Polish — configuration system, Rich terminal output, README with demo, basic end-to-end tests, PyPI packaging
