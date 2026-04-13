"""OpenRouter image renderer.

Uses the chat completions endpoint with modalities=["image", "text"].
Extracts base64-encoded image from the response.
"""

from __future__ import annotations

import base64
import re

from maestro.core.errors import ProviderError
from maestro.core.models import FileData
from maestro.providers.registry import provider

from ._client import _DEFAULT_URL, extract_message, post_chat

_DATA_URL_RE = re.compile(r"data:[a-zA-Z]+/[a-zA-Z0-9.+_-]+;base64,\s*(.+)")


@provider("openrouter", "renderer")
class Renderer:
    """OpenRouter image renderer via chat completions multimodal API."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        self._url = url or _DEFAULT_URL
        self._token = token
        self._model = model

    async def render(self, prompt: str) -> FileData:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image", "text"],
            "stream": False,
        }

        result = await post_chat(self._url, self._token, body)
        message = extract_message(result)

        images = message.get("images", [])
        if not images:
            raise ProviderError(500, "no images in response")

        img = images[0]
        if not isinstance(img, dict):
            raise ProviderError(500, "invalid image in response")

        image_url = img.get("image_url", {})
        url_str = image_url.get("url", "") if isinstance(image_url, dict) else ""

        if not url_str or not url_str.startswith("data:"):
            raise ProviderError(500, "unsupported image URL format: expected data URL")

        match = _DATA_URL_RE.match(url_str)
        if not match:
            raise ProviderError(500, "invalid data URL")

        data = base64.b64decode(match.group(1))

        return FileData(name="image.png", content=data, content_type="image/png")
