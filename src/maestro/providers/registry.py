"""Provider registry with auto-discovery via decorators.

Instead of Go's factory switch statements:

    switch p.Type {
    case "openai":  return newOpenAICompleter(...)
    case "anthropic": return newAnthropicCompleter(...)
    }

Python uses declarative registration:

    @provider("openai", "completer")
    class OpenAICompleter:
        def __init__(self, url: str, token: str, model: str): ...
        async def complete(self, messages, options=None): ...

    # Config wiring becomes:
    instance = create("openai", "completer", url=..., token=..., model=...)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, TypeVar

T = TypeVar("T")

_registry: dict[str, dict[str, type]] = defaultdict(dict)


def provider(vendor: str, capability: str):
    """Register a class as a provider implementation.

    Args:
        vendor: Provider name as used in config (e.g. "openai", "anthropic").
        capability: What it provides (e.g. "completer", "embedder", "renderer").
    """

    def decorator(cls: type[T]) -> type[T]:
        _registry[vendor][capability] = cls
        return cls

    return decorator


def create(vendor: str, capability: str, **kwargs: Any) -> Any:
    """Instantiate a registered provider.

    Raises ValueError if the vendor/capability combination isn't registered.
    """
    try:
        cls = _registry[vendor][capability]
    except KeyError:
        available = list(_registry.get(vendor, {}).keys())
        raise ValueError(
            f"No '{capability}' provider for '{vendor}'. Available: {available or 'none'}"
        ) from None
    return cls(**kwargs)


def list_providers() -> dict[str, list[str]]:
    """Return all registered vendors and their capabilities."""
    return {v: sorted(c.keys()) for v, c in sorted(_registry.items())}


# ── Model type detection ────────────────────────────────────────────

_MODEL_PATTERNS: dict[str, list[str]] = {
    "completer": [
        "gpt",
        "o1",
        "o3",
        "o4",
        "chatgpt",
        "claude",
        "sonnet",
        "opus",
        "haiku",
        "gemini",
        "gemma",
        "mistral",
        "mixtral",
        "codestral",
        "devstral",
        "llama",
        "command",
        "qwen",
        "deepseek",
        "phi",
    ],
    "embedder": ["embed", "embedding"],
    "renderer": ["dall-e", "flux", "stable-diffusion", "imagen"],
    "synthesizer": ["tts"],
    "transcriber": ["whisper", "stt"],
    "reranker": ["rerank"],
    "vectorstore": ["vectorstore", "vector-store", "vecdb", "collection"],
}


def detect_capability(model_id: str) -> str:
    """Guess a model's capability from its ID. Falls back to 'completer'."""
    model_lower = model_id.lower()

    # Check non-completer patterns first (they're more specific)
    for capability in [
        "embedder",
        "renderer",
        "synthesizer",
        "transcriber",
        "reranker",
        "vectorstore",
    ]:
        if any(p in model_lower for p in _MODEL_PATTERNS[capability]):
            return capability

    return "completer"
