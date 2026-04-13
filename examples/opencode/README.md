# Maestro + OpenCode

Run [OpenCode](https://opencode.ai) CLI against Maestro as an OpenAI-compatible backend.

## Architecture

```
┌──────────────┐         ┌────────────────┐         ┌──────────────┐
│   OpenCode   │────────▶│    Maestro     │────────▶│   Ollama     │
│   (CLI)      │  :8080  │   (Platform)   │         │  OpenRouter  │
└──────────────┘         │                │         │  OpenAI ...  │
                         └────────────────┘         └──────────────┘
```

OpenCode runs natively on your machine. Maestro runs in Docker and routes LLM requests to any configured provider.

## Quick Start

```bash
# 1. Start the Maestro backend
docker compose up

# 2. Copy the OpenCode config into your project
cp opencode.json /path/to/your/project/

# 3. Start OpenCode in your project
cd /path/to/your/project
opencode
```

## Configuration

### Maestro Providers

Edit the `platform` config section in `compose.yaml` to add providers:

```yaml
# Ollama (default, local)
- type: openai-compatible
  url: ${LLM_URL}
  models:
    qwen3:
      id: ${LLM_MODEL}

# OpenRouter (200+ cloud models)
- type: openrouter
  token: ${OPENROUTER_API_KEY}
  models:
    deepseek-r1:
      id: deepseek/deepseek-r1

# Direct OpenAI
- type: openai
  token: ${OPENAI_API_KEY}
  models:
    gpt-4o:
```

### OpenCode Models

When adding providers to Maestro, also add matching models in `opencode.json`:

```json
{
  "deepseek-r1": {
    "id": "deepseek-r1",
    "name": "DeepSeek R1 (via OpenRouter)",
    "tool_call": true,
    "temperature": true,
    "limit": { "context": 65536, "output": 8192 }
  }
}
```

## Ports

| Service   | Port | Purpose              |
|-----------|------|----------------------|
| Maestro   | 8080 | OpenAI-compatible API |
| Extractor | 50051| Document extraction (gRPC, internal) |
