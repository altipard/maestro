"""Apache Tika extractor.

Sends binary content via PUT to ``/tika/text`` and parses the JSON response.
"""

from __future__ import annotations

import os

import httpx

from maestro.core.models import FileData

SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
}

SUPPORTED_MIME_TYPES: set[str] = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class TikaExtractor:
    """Extracts text using an Apache Tika server."""

    def __init__(self, url: str = "", **_: object) -> None:
        if not url:
            raise ValueError("tika extractor requires a url")
        self._url = url.rstrip("/")

    async def extract(self, file: FileData) -> str:
        if not _is_supported(file):
            raise ValueError("unsupported file type")

        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{self._url}/tika/text",
                content=file.content,
            )

        if resp.status_code != 200:
            raise RuntimeError(resp.text or httpx.codes(resp.status_code).name)

        data = resp.json()
        content = data.get("X-TIKA:content", "")
        return content.strip()


def _is_supported(file: FileData) -> bool:
    if file.name:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return True

    if file.content_type:
        if file.content_type in SUPPORTED_MIME_TYPES:
            return True

    return False
