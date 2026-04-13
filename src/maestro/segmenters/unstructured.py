"""Unstructured.io chunking API segmenter."""

from __future__ import annotations

import httpx

from maestro.core.models import Segment

_DEFAULT_URL = "https://api.unstructured.io"


class UnstructuredSegmenter:
    """Segments text via the Unstructured.io partition API (POST multipart)."""

    def __init__(
        self,
        url: str = "",
        token: str = "",
        segment_length: int = 1000,
        segment_overlap: int = 0,
        **_: object,
    ) -> None:
        self._url = url or _DEFAULT_URL
        self._token = token
        self._segment_length = segment_length
        self._segment_overlap = segment_overlap

    async def segment(self, text: str) -> list[Segment]:
        if not text:
            return []

        # Build multipart form data matching the Go implementation exactly.
        data: dict[str, str] = {
            "strategy": "fast",
            "chunking_strategy": "by_title",
        }

        data["max_characters"] = str(self._segment_length)

        if self._segment_overlap:
            data["overlap"] = str(self._segment_overlap)

        files = {
            "files": ("content.txt", text.encode("utf-8"), "text/plain"),
        }

        headers: dict[str, str] = {}
        if self._token:
            headers["unstructured-api-key"] = self._token

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._url,
                data=data,
                files=files,
                headers=headers,
            )

        if resp.status_code != 200:
            detail = resp.text or httpx.codes.get_reason_phrase(resp.status_code)
            raise RuntimeError(
                f"Unstructured API error ({resp.status_code}): {detail}"
            )

        elements: list[dict] = resp.json()

        return [Segment(text=el["text"]) for el in elements if el.get("text")]
