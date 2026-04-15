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

## Using RAG (File Search)

LibreChat's RAG pipeline (`rag_api` → Maestro → embedding model) is **only available through Agents**, not in regular chats. Attaching a file to a normal conversation does *not* index it.

**Setup:**

1. **Embedding model** — already wired in `compose.yaml`'s `maestro-config`:
   ```yaml
   - type: ollama
     models:
       nomic-embed-text:        # used by rag_api for embeddings
   ```
   Make sure it's pulled: `ollama pull nomic-embed-text`.

2. **Create an agent** — Sidebar → *Agenten-Marktplatz* → *Agenten erstellen*:
   - Endpoint: `Maestro`
   - Model: any chat model (e.g. `qwen3.5:latest`)
   - Tools: enable **File Search**

3. **Upload files to that agent** — they go through `rag_api /embed` → Maestro `/v1/embeddings` → Ollama, and land in the pgvector store.

4. **Chat with the agent** — questions trigger `rag_api /query` for retrieval, and the chunks are injected into the prompt.

**Verify the pipeline is working:**
```bash
docker compose logs -f rag_api maestro
# On upload: POST /embed → POST /v1/embeddings 200
# On query : POST /query → POST /v1/embeddings 200 → POST /v1/chat/completions 200
```

> **Note on `librechat.yaml`:** the `endpoints.openAI` block is redundant when `endpoints.custom.Maestro` is present — both point to the same backend, and LibreChat routes through the custom one. Removing the `openAI` block keeps the model picker uncluttered.

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
