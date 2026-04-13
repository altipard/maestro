# Maestro + LibreChat

Run [LibreChat](https://librechat.ai) as a full-featured web UI powered by Maestro.

## Architecture

```
┌──────────────┐         ┌────────────────┐         ┌──────────────┐
│  LibreChat   │────────▶│    Maestro     │────────▶│   Ollama     │
│  (Web UI)    │  :8080  │   (Platform)   │         │  OpenRouter  │
│  :3080       │         │                │         │  OpenAI ...  │
└──────────────┘         └────────────────┘         └──────────────┘
       │
       ├── MongoDB
       ├── Meilisearch
       ├── VectorDB (pgvector)
       └── RAG API
```

LibreChat provides the web UI with chat, agents, RAG, and artifacts. Maestro routes all LLM requests to your configured providers.

## Quick Start

```bash
# 1. Create your .env from the example
cp .env.example .env
# Edit .env — fill in CREDS_KEY, JWT_SECRET, JWT_REFRESH_SECRET

# 2. Start the full stack
docker compose up

# 3. Open LibreChat
open http://localhost:3080
```

On first launch, create an account in the LibreChat UI. All LLM traffic goes through Maestro.

## Configuration

### Maestro Providers

Edit the `maestro-config` section in `compose.yaml` to add providers:

```yaml
# Ollama is enabled by default (local models)
- type: ollama
  url: http://host.docker.internal:11434
  models:
    gemma3:1b:
    qwen3.5:latest:

# Uncomment to add OpenRouter, OpenAI, Anthropic, etc.
```

### LibreChat Models

When adding providers to Maestro, update the `models.default` list in `librechat.yaml` so they appear in the UI:

```yaml
models:
  default:
    - 'gemma3:1b'
    - 'qwen3.5:latest'
    - 'deepseek-r1'       # added via OpenRouter
  fetch: true              # or use fetch to auto-discover all models
```

### Environment Variables

Provider API keys go in `.env`:

```bash
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Ports

| Service     | Port  | Purpose                      |
|-------------|-------|------------------------------|
| LibreChat   | 3080  | Web UI                       |
| Maestro     | 8080  | OpenAI-compatible API        |
| MongoDB     | 27017 | LibreChat database (internal)|
| Meilisearch | 7700  | Search index (internal)      |
| VectorDB    | 5432  | RAG embeddings (internal)    |
| Extractor   | 50051 | Document extraction (internal)|

## Combining with OpenCode

You can use OpenCode CLI alongside LibreChat — both connect to the same Maestro instance:

```bash
# While docker compose is running, OpenCode can connect to the same Maestro
# Copy the opencode.json from examples/opencode/ into your project
cp ../opencode/opencode.json /path/to/your/project/
cd /path/to/your/project && opencode
```
