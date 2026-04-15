# Maestro RAG Example

A minimal, local Retrieval-Augmented-Generation pipeline built entirely on Maestro's capabilities — nothing on top, no framework.

## Architecture

```
                        ┌────────────────────────────┐
                        │          Maestro           │
                        │  :8080 /v1/extract         │
                        │        /v1/segment         │
                        │        /v1/embeddings      │
                        │        /v1/chat/completions│
                        └──┬────────┬────────────┬───┘
                           │        │            │
                   gRPC ───┘        └─── gRPC    └── HTTP ──▶ Ollama (host)
                   │                     │                   qwen3
                   ▼                     ▼                   nomic-embed-text
           maestro-extractor    maestro-segmenter
           (PDF/DOCX/images)    (Markdown-aware
                                 chunking)

                        ┌────────────────────────────┐
                        │    ChromaDB :8000          │◀── rag.py
                        │    vector store            │
                        └────────────────────────────┘
```

Every pipeline stage is one Maestro capability:

| Stage         | Capability  | Backend                       |
|---------------|-------------|-------------------------------|
| read file     | `Extractor` | `maestro-extractor` (gRPC)    |
| split text    | `Segmenter` | `maestro-segmenter` (gRPC)    |
| embed chunks  | `Embedder`  | Ollama `nomic-embed-text`     |
| store vectors | `VectorStore` | ChromaDB                    |
| answer        | `Completer` | Ollama `qwen3`                |

See [../../docs/CAPABILITIES.md](../../docs/CAPABILITIES.md) for the full protocol reference.

## Prerequisites

- Docker + Docker Compose
- [Ollama](https://ollama.com) running on the host
- [uv](https://docs.astral.sh/uv/) (Python 3.12 is pinned via `.python-version`)

Pull the two models once:

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5
```

## Run

```bash
# 1. Start Maestro + extractor + segmenter + ChromaDB
docker compose up -d

# 2. Index the bundled sample document (uv auto-installs deps on first run)
uv run rag ingest sample/sample.pdf

# 3. Ask a question
uv run rag ask "which sensors does each Aurora node carry?"
uv run rag ask "where are the deployment sites and how many nodes per range?"
uv run rag ask "what hardware limitation does Aurora v4 address?"
```

`uv run` reads `pyproject.toml`, creates `.venv/` on first invocation, and executes the `rag` script entry point.

### The sample document

`sample/sample.pdf` is a short fictional tech brief about "Project Aurora", a distributed alpine weather sensor network. It has seven sections (overview, hardware, deployment sites, data pipeline, known limitations, team & timeline, contact) with concrete facts — sensor types, battery capacity, site coordinates, failure modes — that make retrieval quality easy to eyeball.

To regenerate it:

```bash
uv run --script sample/make_sample.py
```

The generator script uses PEP 723 inline metadata, so uv handles its `reportlab` dependency without touching this project's `pyproject.toml`.

## What the CLI does

`rag.py` is ~100 lines and hits only documented endpoints.

**`ingest <file>`**

1. `POST /v1/extract` — upload the file, get plain text (Markdown) back.
2. `POST /v1/segment` — chunk the text.
3. `POST /v1/embeddings` — embed all chunks in one call.
4. `chroma.upsert(ids, embeddings, documents, metadatas)`.

**`ask <question>`**

1. `POST /v1/embeddings` — embed the question.
2. `chroma.query(n_results=k)` — top-k chunks.
3. `POST /v1/chat/completions` — system prompt enforces "answer from context only, cite `[file#chunk]`".

## Customizing

All model and index choices live at the top of `rag.py`:

```python
MAESTRO      = "http://localhost:8080/v1"
EMBED_MODEL  = "nomic-embed-text"
CHAT_MODEL   = "qwen3"
COLLECTION   = "docs"
```

Want to use a cloud model for answers? Add an OpenAI / OpenRouter / Anthropic block to the `providers:` list in `compose.yaml` and change `CHAT_MODEL`.

Want better retrieval? Add a `reranker` to Maestro and slot a second step between Chroma and the chat call — the script leaves room for it.

## Ports

| Service   | Port  | Purpose                        |
|-----------|-------|--------------------------------|
| Maestro   | 8080  | OpenAI-compatible API          |
| ChromaDB  | 8000  | Vector store HTTP API          |
| Extractor | —     | gRPC, internal to compose net  |
| Segmenter | —     | gRPC, internal to compose net  |

## Troubleshooting

- **`connection refused` to Ollama** — Ollama must listen on all interfaces for `host.docker.internal` to reach it. On macOS that's the default; on Linux set `OLLAMA_HOST=0.0.0.0` before starting `ollama serve`.
- **`extractor not found`** — The first run pulls the satellite images; wait until `docker compose ps` shows both `extractor` and `segmenter` as healthy.
- **Empty answers** — Check that `ingest` reported a non-zero chunk count. A scanned PDF without OCR will extract to an empty string.
