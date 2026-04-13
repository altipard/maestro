"""Mistral OCR extractor.

Sends a base64-encoded document as JSON to ``/ocr`` and concatenates
the per-page markdown from the response.
"""

from __future__ import annotations

import base64
import os

import httpx

from maestro.core.models import FileData

SUPPORTED_EXTENSIONS: set[str] = {".pdf"}
SUPPORTED_MIME_TYPES: set[str] = {"application/pdf"}

_DEFAULT_URL = "https://api.mistral.ai/v1/"
_DEFAULT_MODEL = "mistral-ocr-latest"


class MistralExtractor:
    """Extracts text from PDFs using the Mistral OCR API."""

    def __init__(
        self,
        url: str = "",
        token: str = "",
        model: str = "",
        **_: object,
    ) -> None:
        self._url = (url or _DEFAULT_URL).rstrip("/")
        self._token = token
        self._model = model or _DEFAULT_MODEL

    async def extract(self, file: FileData) -> str:
        if not _is_supported(file):
            raise ValueError("unsupported file type")

        content_type = file.content_type or "application/pdf"
        data_url = f"data:{content_type};base64,{base64.b64encode(file.content).decode()}"

        body = {
            "model": self._model,
            "document": {
                "type": "document_url",
                "document_name": "test.pdf",
                "document_url": data_url,
            },
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._url}/ocr",
                json=body,
                headers=headers,
            )

        if resp.status_code != 200:
            raise RuntimeError(resp.text or httpx.codes(resp.status_code).name)

        response = resp.json()
        pages = response.get("pages", [])

        parts: list[str] = []
        for page in pages:
            md = page.get("markdown", "")
            if md:
                parts.append(md)

        return "\n\n".join(parts).strip()


def _is_supported(file: FileData) -> bool:
    if file.name:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return True

    if file.content_type:
        if file.content_type in SUPPORTED_MIME_TYPES:
            return True

    return False
