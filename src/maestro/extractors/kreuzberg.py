"""Kreuzberg API extractor.

Sends files as multipart POST to ``/extract`` and parses the JSON array response.
"""

from __future__ import annotations

import mimetypes
import os
import uuid

import httpx

from maestro.core.models import FileData

SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".epub",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".html",
    ".htm",
    ".xml",
    ".json",
    ".csv",
    ".txt",
    ".md",
    ".eml",
    ".msg",
}

SUPPORTED_MIME_TYPES: set[str] = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/epub+zip",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "text/html",
    "application/xhtml+xml",
    "application/xml",
    "text/xml",
    "application/json",
    "text/csv",
    "text/plain",
    "text/markdown",
    "message/rfc822",
    "application/vnd.ms-outlook",
}


class KreuzbergExtractor:
    """Extracts text using a Kreuzberg API server."""

    def __init__(self, url: str = "", token: str = "", **_: object) -> None:
        self._url = url.rstrip("/") if url else ""
        self._token = token

    async def extract(self, file: FileData) -> str:
        if not _is_supported(file):
            raise ValueError("unsupported file type")

        content_type = file.content_type or "application/octet-stream"
        name = file.name
        if not name:
            exts = mimetypes.guess_all_extensions(content_type)
            ext = exts[0] if exts else ""
            name = uuid.uuid4().hex + ext

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._url}/extract",
                files={"files": (name, file.content, content_type)},
            )

        if resp.status_code != 200:
            raise RuntimeError(resp.text or httpx.codes(resp.status_code).name)

        result = resp.json()
        return result[0]["content"].strip()


def _is_supported(file: FileData) -> bool:
    if file.name:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return True

    if file.content_type:
        if file.content_type in SUPPORTED_MIME_TYPES:
            return True

    return False
