# Gritter: Design Notes

Gritter is a CLI that indexes a codebase (local or GitHub) with AST-aware
chunking, retrieves relevant code with hybrid dense+sparse search, and
answers natural-language questions with file:line citations. This document
records the decisions behind the current implementation, the tradeoffs
knowingly accepted, and where the design is (and isn't) novel relative to
typical RAG tooling.

## Chunking

- **AST-aware for Python, TypeScript, Rust** via tree-sitter
  (`indexing/ast_chunker.py`); everything else falls through to a
  language-agnostic **heuristic chunker** that splits on blank-line +
  zero-indent boundaries (`indexing/heuristic_chunker.py`).
- Any tree-sitter parse failure falls back to the heuristic chunker rather
  than failing the index (`indexing/chunker.py`) — availability over
  completeness.
- Named symbols (functions/classes/impls) are never merged across
  boundaries, even if small; only anonymous chunks get merged to hit the
  token floor. This trades chunk-size uniformity for semantic boundaries
  that make citations meaningful ("here's the function" beats "here's an
  arbitrary 200-token window").
- **Tradeoff accepted:** heuristic-chunked (non-AST) languages produce only
  `symbol_type="module"` chunks — no symbol name, no method-level splitting.
  Query answers for unsupported languages will cite coarser spans.

## Retrieval

- Dense (ChromaDB, cosine) and sparse (BM25Okapi) run independently at
  `candidate_k`, then combine via **Reciprocal Rank Fusion** (`k=60`,
  Cormack et al. 2009) rather than score blending. RRF is rank-based, so it
  sidesteps normalizing cosine similarity against BM25 scores — a common
  source of bugs in naive hybrid search.
- Cross-encoder reranking (`sentence-transformers`,
  `cross-encoder/ms-marco-MiniLM-L-12-v2`) is implemented but **off by
  default** (`reranker.provider = "none"` → `NoOpReranker`). Rationale: it's
  the one component with a hard `torch` dependency and added latency; users
  who want the recall/precision bump can opt in without everyone paying the
  install/latency cost.
- BM25 is treated as a *derived* index — ChromaDB holds the canonical chunk
  records, and BM25 rebuilds from `VectorStore.get_all()`. Simpler than
  keeping two sources of truth in sync, at the cost of a full BM25 rebuild
  (whole pickle rewrite) on every reindex; there is no incremental BM25
  update.

## Providers

- `EmbeddingProvider` and `LLMProvider` are minimal ABCs with string-keyed
  factory functions (Voyage/OpenAI/local embeddings; Claude/OpenAI/Ollama
  generation) — not a plugin registry. Adding a provider means adding a
  branch, not registering a class. Deliberately not over-abstracted for
  three options.
- Ollama reuses the OpenAI SDK pointed at a local `/v1` endpoint with a
  dummy API key instead of a dedicated client — avoids a fourth SDK
  dependency for a protocol-compatible target.
- Ollama's default model changed from `llama3` to `llama3.1` in this
  branch: agent mode requires a tool-calling-capable model, and `llama3`
  doesn't support tool calls.
- Each provider imports its SDK lazily in `__init__`, so installing Gritter
  doesn't require every embedding/LLM SDK to be present — only the one
  actually configured.

## Incremental & GitHub indexing

- Incremental indexing is **git-diff-based, not content-hash-based**:
  `git diff --name-status <indexed_commit> HEAD`, with renames handled as
  delete-old/add-new. This means it only works for git-tracked trees with a
  previously recorded commit (`IndexMeta.indexed_commit`); a non-git
  directory always does a full reindex. Chosen over content hashing because
  git already tracks this for free and handles renames/moves correctly.
- GitHub repos are cloned with `--filter=blob:none` (blobless clone) —
  keeps full commit history (needed for the diff above) while deferring
  blob downloads, and repeat indexing runs are `fetch` + `reset --hard`
  against the cached clone rather than a fresh clone each time.

## Generation & citations

- The system prompt hard-constrains the model to answer only from supplied
  context, cite as `path:Lstart-end`, and emit a fixed refusal string when
  the answer isn't in context. This prompt-level guardrail is backed by the
  post-hoc verification described below, rather than being trusted alone.
- Citation extraction (`generation/citations.py`) is still a regex pull from
  the model's own output text, but it is now cross-checked post-hoc:
  `verify_citations` resolves each cited `path:Lstart-end` against the real
  file on disk (confined to `repo_root`) and confirms the line range is
  in-bounds. Citations that don't verify aren't dropped — they're passed
  through with an inline `[unverified]` marker (see `format_sources`) so
  the caller can see the model cited something but the tool couldn't
  confirm it against real files.

## Eval harness

- Metrics are file-level: `recall@k` (fraction of relevant files present in
  top-k retrieved files) and `MRR` (reciprocal rank of first relevant
  file), not chunk-level or graded relevance. Dataset is JSONL
  (`query` + `relevant_files`). No nDCG, no precision — deliberately the
  smallest harness that answers "did retrieval find the right file,"
  matching the file+line citation framing of the product. Coarser than a
  full RAG eval suite, but sufficient to catch retrieval regressions across
  chunking/fusion/reranking changes.

## Config

- Layered via `pydantic_settings.BaseSettings`: init args → env vars
  (`GRITTER_*`) → project `.gritterrc` → user
  `~/.config/gritter/config.toml`. Standard layering, not novel, but
  complete enough that per-repo overrides (e.g. exclude globs, chunk size)
  don't require env vars or code changes.
- `IndexMeta` records the embedding provider/model/dimension used to build
  an index and `validate_provider` rejects querying with a mismatched
  provider — cheap guard against silent dimension-mismatch failures in
  ChromaDB.

## What's novel here vs. a typical RAG-over-code tool

- Symbol-boundary-preserving AST chunking with per-language symbol
  extraction (not just "split by function keyword regex").
- RRF-based hybrid fusion is standard practice, but avoiding score
  normalization bugs by using it correctly is worth calling out — a lot of
  "hybrid search" implementations get this wrong.
- Git-diff-driven incremental indexing that also handles the GitHub clone
  case with blobless clones, rather than the more common
  hash-every-file-on-every-run approach.
- Ollama-via-OpenAI-SDK reuse is a small but genuinely useful trick to avoid
  a needless dependency.

## What's *not* novel / acknowledged as standard

- The overall pipeline shape (chunk → embed → hybrid retrieve → rerank →
  generate with citations) is the standard modern code-RAG architecture,
  not a new idea.
- Config layering, provider factory functions, and the eval harness are all
  conventional implementations of well-known patterns.
