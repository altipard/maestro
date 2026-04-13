"""Unstructured.io extractor.

Sends files as multipart POST and concatenates the text from each
returned element.
"""

from __future__ import annotations

import os

import httpx

from maestro.core.models import FileData

_DEFAULT_URL = "https://api.unstructured.io/general/v0/general"

SUPPORTED_EXTENSIONS: set[str] = {
    ".bmp",
    ".csv",
    ".doc",
    ".docx",
    ".eml",
    ".epub",
    ".heic",
    ".html",
    ".jpeg",
    ".png",
    ".md",
    ".msg",
    ".odt",
    ".org",
    ".p7s",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rst",
    ".rtf",
    ".tiff",
    ".txt",
    ".tsv",
    ".xls",
    ".xlsx",
    ".xml",
}

SUPPORTED_MIME_TYPES: set[str] = {
    "image/bmp",
    "text/csv",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "message/rfc822",
    "application/epub+zip",
    "image/heic",
    "text/html",
    "image/jpeg",
    "image/png",
    "text/markdown",
    "application/vnd.ms-outlook",
    "application/vnd.oasis.opendocument.text",
    "text/org",
    "application/pkcs7-signature",
    "application/pdf",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/rst",
    "application/rtf",
    "image/tiff",
    "text/plain",
    "text/tab-separated-values",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/xml",
}


class UnstructuredExtractor:
    """Extracts text using the Unstructured.io API."""

    def __init__(
        self,
        url: str = "",
        token: str = "",
        model: str = "",
        **_: object,
    ) -> None:
        self._url = url or _DEFAULT_URL
        self._token = token
        self._strategy = model or "fast"

    async def extract(self, file: FileData) -> str:
        if not _is_supported(file):
            raise ValueError("unsupported file type")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._url,
                data={
                    "strategy": self._strategy,
                    "include_page_breaks": "true",
                },
                files={"files": (file.name, file.content)},
            )

        if resp.status_code != 200:
            raise RuntimeError(resp.text or httpx.codes(resp.status_code).name)

        elements = resp.json()

        parts: list[str] = []
        for element in elements:
            text = element.get("text", "")
            if text:
                parts.append(text)

        return "\n".join(parts).strip()


def _is_supported(file: FileData) -> bool:
    if file.name:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return True

    if file.content_type:
        if file.content_type in SUPPORTED_MIME_TYPES:
            return True

    return False
