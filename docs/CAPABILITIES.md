# Capabilities

Maestro is built around a small set of **capability protocols** (Python `typing.Protocol`) defined in [`src/maestro/core/protocols.py`](../src/maestro/core/protocols.py). Every provider implements one or more of these protocols. Capabilities are the atomic building blocks you chain into pipelines.

Two families:

- **Core AI** — single model calls (LLM, embedding, TTS, STT, image, rerank)
- **Pipeline** — document-oriented operations typically backed by satellite services (`maestro-extractor`, `maestro-segmenter`, …)

All methods are `async`. Streaming generators are used where it matters (`Completer`). File I/O uses the shared `FileData` type (bytes + mime-type + optional filename).

---

## Core AI Capabilities

### Completer
Generate chat/text. The workhorse of every LLM backend.

- **Input**: `list[Message]` (role + content, possibly multimodal) + `CompleteOptions` (model, temperature, tools, …)
- **Output**: `AsyncIterator[Completion]` — streaming chunks (text deltas, tool calls, usage, finish reason)
- **Typical providers**: `openai`, `anthropic`, `google`, `ollama`, `openrouter`
- **Chains with**: anything upstream that produces text or structured messages; anything downstream that consumes text (`Synthesizer`, `Translator`, `Summarizer`, another `Completer` for multi-step reasoning)

### Embedder
Turn text into a fixed-length vector for semantic similarity and retrieval.

- **Input**: `list[str]` + `EmbedOptions` (model)
- **Output**: `Embedding` (list of float vectors, one per input)
- **Typical providers**: `openai` (`text-embedding-3-*`), `ollama` (`nomic-embed-text`), `google`
- **Chains with**: `Segmenter` → `Embedder` → `VectorStore` (indexing); user query → `Embedder` → `VectorStore.search` (retrieval)

### Reranker
Given a query and a list of candidate documents, return them reordered by relevance. Typically a cross-encoder that attends to query and document jointly, so it is more accurate than vector cosine similarity but slower — used as a second stage over a larger candidate set.

- **Input**: `query: str`, `documents: list[str]`
- **Output**: `list[RankedDocument]` (document + score + original index)
- **Chains with**: `VectorStore.search` → `Reranker` → `Completer` (classic RAG with reranking)

### Renderer
Generate an image from a text prompt.

- **Input**: `prompt: str`
- **Output**: `FileData` (PNG/JPEG bytes + mime)
- **Typical models**: `dall-e-3`, `flux`, `stable-diffusion-*`
- **Chains with**: `Completer` (prompt refinement) → `Renderer`; or `Transcriber` → `Completer` → `Renderer` for "draw what I said".

### Synthesizer
Text-to-speech.

- **Input**: `text: str`
- **Output**: `FileData` (audio bytes — MP3/WAV)
- **Typical models**: `tts-1`, ElevenLabs, Google TTS
- **Chains with**: `Completer` → `Synthesizer` (voice assistant output); `Translator` → `Synthesizer` (dubbing)

### Transcriber
Speech-to-text.

- **Input**: `FileData` (audio bytes)
- **Output**: `str` (transcript)
- **Typical models**: `whisper-*`, `gpt-4o-transcribe`, local `whisper.cpp`
- **Chains with**: `Transcriber` → `Completer` (voice chat); `Transcriber` → `Segmenter` → `Embedder` → `VectorStore` (index call recordings for search)

---

## Pipeline Capabilities

These are the dedicated building blocks for document and knowledge workflows. They are defined as satellite interfaces so external projects (gRPC services, SaaS APIs) can drop in without touching the core server.

### Extractor
Pull plain text out of a binary document.

- **Input**: `FileData` (PDF, DOCX, PPTX, HTML, image, …)
- **Output**: `str` (typically Markdown-ish plain text, layout-aware for good extractors)
- **Typical backends**: `tika`, `azure` (Document Intelligence), `mistral` (OCR), `docling`, `unstructured`, `kreuzberg`, `grpc` (maestro-extractor)
- **Chains with**: start of any document pipeline — `Extractor` → `Segmenter` → `Embedder` or `Extractor` → `Summarizer`

### Segmenter
Split long text into chunks that fit a model's context window and preserve semantic boundaries (headings, code blocks, sentences).

- **Input**: `text: str`
- **Output**: `list[Segment]` (text + offsets + optional metadata like heading path)
- **Typical backends**: built-in Markdown/code-aware splitter, `jina` segment API, `unstructured`, `kreuzberg`, tree-sitter for source code
- **Chains with**: `Extractor` → `Segmenter` → `Embedder`; `Transcriber` → `Segmenter` → `Summarizer` (per-chunk summaries, then merge)

### Searcher
High-level document search. An opinionated capability that hides the underlying stack (often `Embedder` + `VectorStore` + `Reranker`, or a full external search engine like Elastic / Typesense).

- **Input**: `query: str`, `limit: int`
- **Output**: `list[SearchResult]` (document + score + metadata)
- **Chains with**: `Searcher` → `Completer` (RAG with pluggable backend); multiple `Searcher`s fused for hybrid search

### Summarizer
Reduce a long text to a shorter one while preserving key information.

- **Input**: `text: str`
- **Output**: `str`
- **Chains with**: `Extractor` → `Summarizer` (executive summary of a PDF); `Transcriber` → `Segmenter` → per-chunk `Summarizer` → merge (meeting minutes); recursive `Summarizer` for very long inputs (summary of summaries)

### Translator
Translate text to a target language.

- **Input**: `text: str`, `target: str` (e.g. `"de"`, `"ja"`)
- **Output**: `str`
- **Chains with**: `Extractor` → `Segmenter` → `Translator` → merge (large-document translation); `Translator` → `Synthesizer` (voice dubbing); `Translator` at query time (cross-lingual RAG)

---

## Storage: VectorStore

Not a "capability" in the AI sense but the storage counterpart that most pipelines need. Defined next to the capability protocols for symmetry.

- **Operations**: `upsert(ids, embeddings, documents, metadatas)`, `search(embedding, limit, filters)`, `delete(ids)`, `count()`
- **Providers**: `chroma` (embedded or remote), plus anything you register under the same protocol
- **Chains with**: write path: `Embedder` → `VectorStore.upsert`; read path: `Embedder` → `VectorStore.search` → `Reranker` → `Completer`

---

## Canonical Pipelines

Concrete compositions that cover most real-world uses. Each arrow is one capability call.

### 1. Plain chat
```
User messages ─▶ Completer ─▶ stream
```

### 2. Voice assistant (end-to-end)
```
Mic → Transcriber → Completer → Synthesizer → Speaker
```

### 3. Image generation with prompt polish
```
Idea → Completer (prompt rewrite) → Renderer → image
```

### 4. RAG indexing (write side)
```
File → Extractor → Segmenter → Embedder → VectorStore.upsert
```

### 5. RAG query with reranking (read side)
```
Query ─▶ Embedder ─▶ VectorStore.search ─▶ Reranker ─▶ Completer ─▶ answer + citations
                     └─────────── Searcher can collapse these 3 ──┘
```

### 6. Hybrid search
```
                      ┌─ Searcher (keyword / BM25) ─┐
Query ─▶ dispatch ─▶ ┤                              ├─▶ fuse scores ─▶ Reranker ─▶ Completer
                      └─ Searcher (vector)  ────────┘
```

### 7. Document summarization (map-reduce)
```
PDF → Extractor → Segmenter ─┬─▶ Summarizer (per chunk) ─┐
                             ├─▶ Summarizer              ├─▶ merge ─▶ Summarizer ─▶ final summary
                             └─▶ Summarizer              ┘
```

### 8. Meeting minutes from audio
```
Audio → Transcriber → Segmenter ─┬─▶ Summarizer → minutes
                                 └─▶ Embedder → VectorStore  (searchable archive)
```

### 9. Large-document translation
```
PDF → Extractor → Segmenter → Translator (per chunk) → merge → target text
                                                   └─▶ optional Synthesizer → dubbed audio
```

### 10. Cross-lingual knowledge base
```
Ingest: File → Extractor → Translator (→ lingua franca) → Segmenter → Embedder → VectorStore
Query:  User question (any language) → Translator → Searcher → Completer → Translator (→ user language)
```

### 11. Agent with tools
```
User → Completer ⇄ tool calls (Searcher, Extractor, external APIs) → Completer → answer
```
Implemented in [`src/maestro/chains/agent.py`](../src/maestro/chains/agent.py): a loop that feeds tool results back into a `Completer` until it produces a final answer.

---

## Design notes

- **Composability over inheritance** — every capability is a single-method `Protocol`. Composition is just calling one after another; no framework-level pipeline DSL is required.
- **One interface, many backends** — a `Summarizer` might be an LLM behind a `Completer`, a dedicated summarization API, or a local model. The calling code does not care.
- **Middleware is transparent** — rate limiting and tracing wrap every provider instance via `wrap()` in [`src/maestro/middleware/wrappers.py`](../src/maestro/middleware/wrappers.py), so chains get observability and throttling for free.
- **Streaming where it matters** — only `Completer` streams by default; pipeline stages operate on complete values because chunk-level streaming rarely makes sense when the next stage needs the full text.
