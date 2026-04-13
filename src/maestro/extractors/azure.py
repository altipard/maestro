"""Azure Document Intelligence extractor.

Submits a document for analysis, polls until the operation completes,
and returns the extracted markdown content.
"""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlencode

import httpx

from maestro.core.models import FileData

SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".jpeg",
    ".jpg",
    ".png",
    ".bmp",
    ".tiff",
    ".heif",
    ".docx",
    ".pptx",
    ".xlsx",
}

SUPPORTED_MIME_TYPES: set[str] = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/heif",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

_POLL_INTERVAL = 5  # seconds, matches Go implementation
_API_VERSION = "2024-11-30"
_MODEL = "prebuilt-layout"


class AzureExtractor:
    """Extracts text using Azure Document Intelligence."""

    def __init__(self, url: str = "", token: str = "", **_: object) -> None:
        if not url:
            raise ValueError("azure extractor requires a url")
        self._url = url.rstrip("/")
        self._token = token

    async def extract(self, file: FileData) -> str:
        if not _is_supported(file):
            raise ValueError("unsupported file type")

        params = urlencode(
            {
                "api-version": _API_VERSION,
                "outputContentFormat": "markdown",
            }
        )
        analyze_url = f"{self._url}/documentintelligence/documentModels/{_MODEL}:analyze?{params}"

        headers = {
            "Content-Type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": self._token,
        }

        async with httpx.AsyncClient() as client:
            # Submit document for analysis
            resp = await client.post(
                analyze_url,
                content=file.content,
                headers=headers,
            )

            if resp.status_code != 202:
                raise RuntimeError(resp.text or httpx.codes(resp.status_code).name)

            operation_url = resp.headers.get("Operation-Location", "")
            if not operation_url:
                raise RuntimeError("missing operation location")

            # Poll until analysis completes
            while True:
                resp = await client.get(
                    operation_url,
                    headers={"Ocp-Apim-Subscription-Key": self._token},
                )

                if resp.status_code != 200:
                    raise RuntimeError(resp.text or httpx.codes(resp.status_code).name)

                operation = resp.json()
                status = operation.get("status", "")

                if status in ("running", "notStarted"):
                    await asyncio.sleep(_POLL_INTERVAL)
                    continue

                if status != "succeeded":
                    raise RuntimeError(f"operation {status}")

                content = operation.get("analyzeResult", {}).get("content", "")
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
