"""OpenRouter transcriber.

Uses chat completions with input_audio content type for speech-to-text.
"""

from __future__ import annotations

import base64
import os

from maestro.core.errors import ProviderError
from maestro.core.models import FileData
from maestro.providers.registry import provider

from ._client import _DEFAULT_URL, extract_message, post_chat


@provider("openrouter", "transcriber")
class Transcriber:
    """OpenRouter STT transcriber via chat completions multimodal API."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        self._url = url or _DEFAULT_URL
        self._token = token
        self._model = model

    async def transcribe(self, file: FileData) -> str:
        audio_b64 = base64.b64encode(file.content).decode()
        audio_format = _detect_audio_format(file)

        body = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please transcribe this audio file."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format,
                            },
                        },
                    ],
                }
            ],
            "stream": False,
        }

        result = await post_chat(self._url, self._token, body)
        message = extract_message(result)

        text = message.get("content", "")
        return text.strip() if isinstance(text, str) else ""


def _detect_audio_format(file: FileData) -> str:
    """Detect audio format from content type or filename."""
    if file.content_type:
        _, _, subtype = file.content_type.partition("/")
        _SUBTYPE_MAP = {
            "x-wav": "wav",
            "mpeg": "mp3",
            "x-aiff": "aiff",
            "mp4": "m4a",
            "x-m4a": "m4a",
        }
        return _SUBTYPE_MAP.get(subtype, subtype)

    if file.name:
        ext = os.path.splitext(file.name)[1].lstrip(".").lower()
        _EXT_MAP = {"aif": "aiff", "weba": "webm"}
        return _EXT_MAP.get(ext, ext) if ext else "wav"

    return "wav"
