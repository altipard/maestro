"""OpenRouter TTS synthesizer.

Uses chat completions with modalities=["text", "audio"] to generate speech.
"""

from __future__ import annotations

import base64

from maestro.core.errors import ProviderError
from maestro.core.models import FileData
from maestro.providers.registry import provider

from ._client import _DEFAULT_URL, extract_message, post_chat


@provider("openrouter", "synthesizer")
class Synthesizer:
    """OpenRouter TTS synthesizer via chat completions multimodal API."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        self._url = url or _DEFAULT_URL
        self._token = token
        self._model = model

    async def synthesize(self, text: str) -> FileData:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": text}],
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
            "stream": False,
        }

        result = await post_chat(self._url, self._token, body)
        message = extract_message(result)

        audio = message.get("audio", {})
        if not isinstance(audio, dict):
            raise ProviderError(500, "no audio data in response")

        audio_data = audio.get("data", "")
        if not audio_data:
            raise ProviderError(500, "no audio data in response")

        content = base64.b64decode(audio_data)
        return FileData(name="speech.wav", content=content, content_type="audio/wav")
