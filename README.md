# Maestro

Intelligent LLM orchestration platform. A single OpenAI-compatible API that routes requests to any LLM provider — OpenAI, Anthropic, Google, Ollama, OpenRouter, and more.

```
Any OpenAI-compatible client
        │
        ▼
   ┌─────────┐      ┌──────────────────────────────┐
   │ Maestro  │─────▶│  OpenAI  │ Anthropic │ Google │
   │  :8080   │      │  Ollama  │ OpenRouter│  gRPC  │
   └─────────┘      └──────────────────────────────┘
```

Connect **any** client that speaks the OpenAI API — [LibreChat](https://librechat.ai), [OpenCode](https://opencode.ai), [Continue](https://continue.dev), curl — and Maestro handles the rest.

## Features

- **Multi-provider routing** — Configure multiple LLM providers in one YAML file. Maestro exposes them all through a single `/v1/` endpoint.
- **OpenAI-compatible API** — Chat Completions (`/v1/chat/completions`) and Responses API (`/v1/responses`) with full streaming support.
- **Provider error handling** — Structured error propagation with semantic error codes (`rate_limit_exceeded`, `authentication_error`, etc.) and mid-stream error events.
- **Proactive rate limiting** — Reads rate-limit headers from upstream providers and throttles before hitting 429s.
- **Auth middleware** — Optional Bearer token authentication, configurable via YAML.
- **Document extraction** — Extract text from PDFs, DOCX, images via multiple backends (Tika, Azure, Mistral OCR, and more).
- **Text segmentation** — Markdown-aware and code-aware chunking for RAG pipelines, with optional tree-sitter support.
- **Web scraping** — Built-in HTML-to-text scraper that strips navigation, footers, and non-content elements.
- **Docker-ready** — Multi-arch Docker image, Compose files for full-stack setups.

## Quick Start

### Option A: Docker (recommended)

```bash
docker compose up
```

Maestro starts on `http://localhost:8080`. Configure providers in the inline `compose.yaml` config.

### Option B: Local Development

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Install dependencies
uv pip install -e ".[all,dev]"

# Configure your providers
cp config.yaml config.local.yaml
# Edit config.local.yaml with your API keys

# Start the server
uv run python -m maestro --config config.local.yaml
```

### Option C: Taskfile

If you have [Task](https://taskfile.dev) installed:

```bash
task install    # Install dependencies
task server     # Start server (reads .env for API keys)
task test       # Run test suite
task lint       # Run linter
task fmt        # Auto-format code
```

## Configuration

Maestro is configured via a single YAML file. Environment variables are supported with `${VAR}` syntax.

### Providers

```yaml
address: ":8080"

providers:
  # OpenAI
  - type: openai
    token: ${OPENAI_API_KEY}
    models:
      gpt-4o:
      gpt-4o-mini:
      text-embedding-3-small:    # auto-detected as embedder

  # Anthropic
  - type: anthropic
    token: ${ANTHROPIC_API_KEY}
    models:
      claude-sonnet-4-5-20251022:

  # Google Gemini
  - type: google
    token: ${GOOGLE_API_KEY}
    models:
      gemini-2.5-pro:

  # Ollama (local)
  - type: ollama
    models:
      llama3:
      qwen3:

  # OpenRouter (200+ models via one API key)
  - type: openrouter
    token: ${OPENROUTER_API_KEY}
    models:
      deepseek-r1:
        id: deepseek/deepseek-r1
      llama-4-maverick:
        id: meta-llama/llama-4-maverick

  # Any OpenAI-compatible endpoint
  - type: openai
    url: https://your-custom-endpoint.com/v1
    token: ${CUSTOM_API_KEY}
    models:
      custom-model:
        id: actual-model-id
```

### Model Capabilities

Maestro auto-detects what each model can do based on its name. You can override this with the `type` field:

| Capability    | Auto-detected patterns              | Description            |
|---------------|--------------------------------------|------------------------|
| `completer`   | gpt, claude, gemini, llama, qwen ... | Chat completions       |
| `embedder`    | embed, embedding                     | Text embeddings        |
| `renderer`    | dall-e, flux, stable-diffusion       | Image generation       |
| `synthesizer` | tts                                  | Text-to-speech         |
| `transcriber` | whisper, stt                         | Speech-to-text         |
| `reranker`    | rerank                               | Document reranking     |

### Authentication

```yaml
auth:
  tokens:
    - "your-secret-token"
    - "another-token"
```

When tokens are configured, all requests must include `Authorization: Bearer <token>`. If no tokens are listed, the API is open.

### Document Extraction

```yaml
extractors:
  text:
    type: text                           # Local text reader (no service needed)

  tika:
    type: tika
    url: http://localhost:9998

  azure:
    type: azure
    url: https://your-instance.cognitiveservices.azure.com
    token: ${AZURE_DOC_KEY}
    model: prebuilt-layout

  grpc:
    type: grpc
    url: grpc://extractor:50051          # maestro-extractor service
```

Supported types: `text`, `tika`, `kreuzberg`, `mistral`, `unstructured`, `docling`, `azure`, `grpc`

### Text Segmentation

```yaml
segmenters:
  default:
    type: text                           # Built-in markdown/code-aware splitter
    segment_length: 1000
    segment_overlap: 100
```

Supported types: `text`, `jina`, `kreuzberg`, `unstructured`, `grpc`

### Rate Limiting

Rate limits can be set at the provider level or per model:

```yaml
providers:
  - type: openai
    token: ${OPENAI_API_KEY}
    limit: 10                            # 10 req/s for all models in this block
    models:
      gpt-4o:
        limit: 5                         # Override: 5 req/s for gpt-4o only
      gpt-4o-mini:                       # Inherits provider-level 10 req/s
```

Additionally, the OpenAI provider includes a proactive throttle that reads `x-ratelimit-remaining-*` headers and pauses requests before capacity is exhausted.

## API Endpoints

| Endpoint                    | Method | Description                     |
|-----------------------------|--------|---------------------------------|
| `/v1/chat/completions`      | POST   | Chat Completions (streaming + non-streaming) |
| `/v1/responses`             | POST   | Responses API (streaming + non-streaming)    |
| `/v1/embeddings`            | POST   | Text embeddings                 |
| `/v1/models`                | GET    | List available models           |
| `/v1/models/{id}`           | GET    | Get model details               |
| `/v1/extract`               | POST   | Document text extraction        |
| `/v1/segment`               | POST   | Text segmentation               |

All endpoints follow the [OpenAI API specification](https://platform.openai.com/docs/api-reference).

## Architecture

```
src/maestro/
├── core/               # Domain models, protocols, error types
│   ├── models.py       #   Completion, Message, Content, Usage, Accumulator
│   ├── protocols.py    #   Completer, Embedder, Renderer, ... (Protocol classes)
│   ├── types.py        #   Role, Status, Effort, ToolChoice (enums)
│   └── errors.py       #   ProviderError, status_code_from_error()
│
├── providers/          # LLM provider implementations
│   ├── registry.py     #   @provider decorator, create(), detect_capability()
│   ├── openai/         #   OpenAI (+ throttle transport)
│   ├── anthropic/      #   Anthropic (Claude)
│   ├── google/         #   Google Gemini
│   ├── ollama/         #   Ollama (local models)
│   ├── openrouter/     #   OpenRouter (completer, renderer, synthesizer, transcriber)
│   ├── chroma/         #   ChromaDB vector store
│   └── grpc/           #   Custom gRPC extractor/segmenter
│
├── server/             # FastAPI HTTP layer
│   ├── app.py          #   Application factory, middleware wiring
│   ├── auth.py         #   Bearer token authentication middleware
│   └── openai/         #   OpenAI-compatible API handlers
│       ├── chat.py     #     /v1/chat/completions
│       ├── responses/  #     /v1/responses
│       ├── embeddings.py#    /v1/embeddings
│       ├── models_handler.py# /v1/models
│       ├── extract.py  #     /v1/extract
│       └── segment.py  #     /v1/segment
│
├── config/             # YAML config loader + wiring
│   ├── loader.py       #   load() → Config (the wired object graph)
│   └── models.py       #   Pydantic models for config validation
│
├── extractors/         # Document extraction backends
│   ├── text.py         #   Local text/markdown reader
│   ├── tika.py         #   Apache Tika
│   ├── azure.py        #   Azure Document Intelligence
│   ├── mistral.py      #   Mistral OCR
│   ├── kreuzberg.py    #   Kreuzberg API
│   ├── docling.py      #   IBM Docling
│   ├── unstructured.py #   Unstructured.io
│   └── multi.py        #   Multi-extractor fallback chain
│
├── segmenters/         # Text chunking backends
│   ├── text.py         #   Markdown/code-aware splitter
│   ├── code.py         #   Tree-sitter AST-based code splitter
│   ├── jina.py         #   Jina Segment API
│   ├── kreuzberg.py    #   Kreuzberg chunking
│   └── unstructured.py #   Unstructured.io chunking
│
├── scrapers/           # Web scraping
│   └── fetch.py        #   HTML-to-text scraper
│
├── chains/             # Agent chains
│   └── agent.py        #   Tool-calling agent loop
│
├── middleware/          # Provider middleware (rate limiting, tracing)
│   └── wrappers.py     #   wrap() — rate limiter + OTel tracing
│
├── policy/             # Access control
│   ├── policy.py       #   PolicyProvider protocol
│   └── noop.py         #   NoOpPolicy (allow all)
│
└── tools/              # Tool definitions
    └── __init__.py     #   Tool schema normalization
```

### Design Principles

- **Protocol-based interfaces** — All capabilities (`Completer`, `Embedder`, `Extractor`, ...) are Python `Protocol` classes. Providers implement them without inheritance.
- **Registry pattern** — The `@provider("openai", "completer")` decorator registers implementations. Config wiring uses `create("openai", "completer", ...)` to instantiate.
- **Middleware stack** — Every provider instance is wrapped with rate limiting and tracing middleware via `wrap()`.
- **Streaming-first** — All completers are async generators. The `Accumulator` collects chunks for non-streaming responses.
- **Error propagation** — `ProviderError` carries HTTP status codes and retry-after durations from upstream through to the client, with semantic error type mapping per protocol.

## Examples

### LibreChat Integration

Full-stack setup with LibreChat as the web UI:

```bash
cd examples/librechat
cp .env.example .env
# Edit .env with your secrets
docker compose up
# Open http://localhost:3080
```

See [examples/librechat/README.md](examples/librechat/README.md) for details.

### OpenCode CLI

Use Maestro as the backend for OpenCode:

```bash
cd examples/opencode
docker compose up
# Copy opencode.json to your project, then run: opencode
```

See [examples/opencode/README.md](examples/opencode/README.md) for details.

### curl

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# List models
curl http://localhost:8080/v1/models

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

## Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Task](https://taskfile.dev) (optional, for task runner)
- Docker (optional, for containerized deployment)

### Setup

```bash
git clone https://github.com/altipard/maestro.git
cd maestro
uv pip install -e ".[all,dev]"
```

### Running Tests

```bash
# All tests
task test

# Unit tests only (no API keys needed)
task test-unit

# Specific test file
uv run pytest tests/maestro/core/test_errors.py -v
```

### Linting & Formatting

```bash
task lint       # Check for issues
task fmt        # Auto-fix formatting
```

### Adding a New Provider

1. Create a new directory under `src/maestro/providers/yourprovider/`
2. Implement the relevant protocol (`Completer`, `Embedder`, etc.)
3. Register with the `@provider` decorator:

```python
from maestro.providers.registry import provider

@provider("yourprovider", "completer")
class Completer:
    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        ...

    async def complete(self, messages, options=None):
        ...
        yield completion_chunk
```

4. Auto-import in `src/maestro/providers/__init__.py`
5. Add a case in `config/loader.py` if needed (or it works automatically via the registry)

### Docker

```bash
# Build
docker build -t maestro .

# Build and push multi-arch
task publish

# Run
docker run -p 8080:8080 -v ./config.yaml:/config.yaml maestro
```

## Available Task Commands

| Command                | Description                                        |
|------------------------|----------------------------------------------------|
| `task server`          | Start Maestro server locally (reads `.env`)         |
| `task run`             | Start via Docker Compose                            |
| `task test`            | Run full test suite                                 |
| `task test-unit`       | Run unit tests (no API keys needed)                 |
| `task lint`            | Check code with ruff                                |
| `task fmt`             | Auto-format code                                    |
| `task publish`         | Build and push multi-arch Docker image              |
| `task proto`           | Regenerate protobuf Python files                    |
| `task install`         | Install all dependencies                            |
| `task example-librechat` | Start Maestro + LibreChat stack                  |
| `task example-opencode`  | Start Maestro backend for OpenCode               |
| `task lgtm`            | Start Grafana OTEL stack for observability          |

## License

MIT
