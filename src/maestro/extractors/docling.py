"""IBM Docling extractor.

Submits a file for async conversion, polls for completion, then
retrieves the extracted document content.
"""

from __future__ import annotations

import asyncio
import os

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

_POLL_INTERVAL = 4  # seconds, matches Go implementation


class DoclingExtractor:
    """Extracts text using an IBM Docling server."""

    def __init__(self, url: str = "", token: str = "", **_: object) -> None:
        if not url:
            raise ValueError("docling extractor requires a url")
        self._url = url.rstrip("/")
        self._token = token

    async def extract(self, file: FileData) -> str:
        if not _is_supported(file):
            raise ValueError("unsupported file type")

        async with httpx.AsyncClient() as client:
            # Submit async conversion
            resp = await client.post(
                f"{self._url}/v1/convert/file/async",
                files={"files": (file.name, file.content)},
            )

            if resp.status_code != 200:
                raise RuntimeError(resp.text or httpx.codes(resp.status_code).name)

            task_id = resp.json()["task_id"]

            # Poll until task completes
            await self._await_task(client, task_id)

            # Retrieve the result
            return await self._read_document(client, task_id)

    async def _await_task(self, client: httpx.AsyncClient, task_id: str) -> None:
        while True:
            await asyncio.sleep(_POLL_INTERVAL)

            resp = await client.get(f"{self._url}/v1/status/poll/{task_id}")
            task = resp.json()
            status = task.get("task_status", "")

            if status == "started":
                continue
            if status == "success":
                return

            raise RuntimeError("task failed")

    async def _read_document(self, client: httpx.AsyncClient, task_id: str) -> str:
        resp = await client.get(f"{self._url}/v1/result/{task_id}")
        task = resp.json()

        if task.get("task_status") != "success":
            raise RuntimeError("task not successful")

        doc = task.get("document", {})

        # Priority order: markdown > text > json > html (last non-empty wins
        # in Go, which checks html, json, text, markdown in sequence)
        text = ""
        if doc.get("html_content"):
            text = doc["html_content"]
        if doc.get("json_content"):
            text = doc["json_content"]
        if doc.get("text_content"):
            text = doc["text_content"]
        if doc.get("md_content"):
            text = doc["md_content"]

        if not text:
            raise RuntimeError("no document content")

        return text


def _is_supported(file: FileData) -> bool:
    if file.name:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return True

    if file.content_type:
        if file.content_type in SUPPORTED_MIME_TYPES:
            return True

    return False
