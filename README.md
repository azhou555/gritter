# gritter

Query any codebase with natural language. Gritter indexes your code using AST-aware chunking and hybrid retrieval (dense + BM25 with reciprocal rank fusion), then answers questions with cited file and line number references.

```
$ gritter index ./my-project
Indexing /home/user/my-project as my-project
✓ Index my-project ready — 142 files, 1,847 chunks, languages: python, typescript, rust

$ gritter ask "How does authentication work?"
Token validation happens in src/auth/jwt.py:L15-45 via the validate_token function,
which decodes the JWT and checks expiry using the SECRET_KEY from environment config
(src/config.py:L8).

Sources:
  • src/auth/jwt.py:L15-45
  • src/config.py:L8
```

---

## Installation

```bash
pip install gritter
```

### API keys

Gritter reads API keys from environment variables — never written to config files.

| Provider | Variable |
|---|---|
| Anthropic (default LLM) | `ANTHROPIC_API_KEY` |
| Voyage (default embeddings) | `VOYAGE_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |

---

## Quick start

```bash
# Index a codebase
gritter index /path/to/repo

# Ask a single question
gritter ask "Where is the database connection configured?"

# Interactive multi-turn chat
gritter chat

# Check index status
gritter status

# Use a different embedding provider (no API key needed)
GRITTER_EMBEDDING__PROVIDER=local gritter index /path/to/repo
```

---

## Commands

| Command | Description |
|---|---|
| `gritter index <path>` | Index a local directory |
| `gritter ask "<question>"` | Single query — streams answer, prints Sources |
| `gritter chat` | Interactive REPL for multi-turn queries |
| `gritter status [name]` | Show index stats (files, chunks, languages, provider) |
| `gritter config set <key> <value>` | Write a config value to `~/.config/gritter/config.toml` |
| `gritter config show` | Print current effective configuration |

---

## Configuration

Config is loaded in priority order: CLI flags → environment variables → `.gritterrc` (project root) → `~/.config/gritter/config.toml` → defaults.

```bash
# Switch to OpenAI embeddings
gritter config set embedding.provider openai

# Switch to local embeddings (no API key)
gritter config set embedding.provider local

# Use Ollama for LLM (no API key)
gritter config set llm.provider ollama
gritter config set llm.model llama3.1

# Adjust retrieval
gritter config set retrieval.top_k 10
```

The default Ollama model is `llama3.1` (not `llama3`) — agent mode requires
a tool-calling-capable model, which `llama3` does not support.

Environment variable override format: `GRITTER_<SECTION>__<KEY>` (double underscore for nesting).

```bash
GRITTER_LLM__PROVIDER=openai gritter ask "..."
```

### Key config fields

| Key | Default | Description |
|---|---|---|
| `embedding.provider` | `voyage` | `voyage` \| `openai` \| `local` |
| `embedding.model` | (provider default) | Model name override |
| `llm.provider` | `claude` | `claude` \| `openai` \| `ollama` |
| `llm.model` | (provider default) | Model name override |
| `index.chunk_min_tokens` | `50` | Merge chunks smaller than this |
| `index.chunk_max_tokens` | `512` | Split chunks larger than this |
| `retrieval.top_k` | `5` | Number of results returned |
| `retrieval.candidate_k` | `20` | Candidates fed into RRF from each retriever |

---

## Language support

AST-based chunking (tree-sitter) for **Python**, **TypeScript/JavaScript**, and **Rust** — extracts functions, classes, and methods as individual chunks with correct line boundaries.

All other languages use a heuristic chunker: blank-line-separated blocks respecting indentation depth, with token-limit splitting as a fallback.

---

## Architecture

Three independent pipelines sharing a common data model:

- **Indexing**: file discovery → language detection → AST/heuristic chunking → embedding → ChromaDB + BM25 storage
- **Retrieval**: embed query → dense search (ChromaDB cosine) + sparse search (BM25) → reciprocal rank fusion → top-k
- **Generation (agentic)**: the LLM drives an agentic tool-calling loop rather
  than following a fixed retrieve-then-generate pipeline. It decides when to
  call `search_code`, `read_file`, `grep`, `list_symbols`, or `reindex` via
  native tool-calling, looping up to 8 tool calls per turn before being
  forced to produce a text-only answer. Citations in the final answer are
  then verified against the real files/line-ranges on disk — not just
  parsed and trusted — with unverified citations flagged `[unverified]`.

The CLI layer is a thin orchestrator — no business logic.

---

## Multiple indexes

Multiple repos can coexist. Use `--name` to distinguish them:

```bash
gritter index ~/work/backend --name backend
gritter index ~/work/frontend --name frontend

gritter ask "How are API routes defined?" --name backend
gritter chat --name frontend
```

---

## Development

```bash
git clone https://github.com/yourname/gritter
cd gritter
pip install -e ".[dev]"
pytest
```

End-to-end tests (requires API keys):
```bash
GRITTER_RUN_E2E_TESTS=1 pytest tests/test_e2e.py -v
```

## Download statistics

The **PyPI download report** GitHub Actions workflow generates daily download totals and downloadable history. See [download tracking](docs/analytics/README.md) for reports, local usage, retention, and release-level analysis. These statistics measure downloads rather than unique installations.
