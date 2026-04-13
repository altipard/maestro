"""Ollama completer — thin wrapper reusing the OpenAI provider.

Ollama exposes an OpenAI-compatible API at /v1, so we just normalize
the URL and delegate to the OpenAI completer.
"""

from __future__ import annotations

from maestro.providers.openai.completer import Completer as OpenAICompleter
from maestro.providers.registry import provider


@provider("ollama", "completer")
class Completer(OpenAICompleter):
    """Ollama completer — delegates to OpenAI provider with URL normalization."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        url = (url or "http://localhost:11434").rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        super().__init__(url=url + "/v1", token=token or "ollama", model=model)
