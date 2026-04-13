"""Ollama embedder — thin wrapper reusing the OpenAI provider."""

from __future__ import annotations

from maestro.providers.openai.embedder import Embedder as OpenAIEmbedder
from maestro.providers.registry import provider


@provider("ollama", "embedder")
class Embedder(OpenAIEmbedder):
    """Ollama embedder — delegates to OpenAI provider with URL normalization."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        url = (url or "http://localhost:11434").rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        super().__init__(url=url + "/v1", token=token or "ollama", model=model)
