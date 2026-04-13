"""OpenRouter completer — delegates to the OpenAI-compatible provider.

OpenRouter speaks the OpenAI Chat Completions API at https://openrouter.ai/api/v1.
The completer simply reuses the OpenAI provider with the correct base URL.
"""

from __future__ import annotations

from maestro.providers.openai.completer import Completer as OpenAICompleter
from maestro.providers.registry import provider

_DEFAULT_URL = "https://openrouter.ai/api/v1"


@provider("openrouter", "completer")
class Completer(OpenAICompleter):
    """OpenRouter completer — OpenAI-compatible with openrouter.ai base URL."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        super().__init__(url=url or _DEFAULT_URL, token=token, model=model)
