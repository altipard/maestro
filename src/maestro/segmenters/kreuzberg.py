"""Kreuzberg chunking API segmenter."""

from __future__ import annotations

import httpx

from maestro.core.models import Segment

_DEFAULT_SEGMENT_LENGTH = 2000


class KreuzbergSegmenter:
    """Segments text via the Kreuzberg extract/chunk API (POST multipart)."""

    def __init__(
        self,
        url: str = "",
        token: str = "",
        segment_length: int = _DEFAULT_SEGMENT_LENGTH,
        segment_overlap: int = 0,
        **_: object,
    ) -> None:
        if not url:
            raise ValueError("kreuzberg segmenter requires a url")
        self._url = url.rstrip("/")
        self._token = token
        self._segment_length = segment_length
        self._segment_overlap = segment_overlap

    async def segment(self, text: str) -> list[Segment]:
        if not text:
            return []

        # Build query parameters (matches Go: chunk_content, max_chars, max_overlap)
        params: dict[str, str] = {"chunk_content": "true"}
        params["max_chars"] = str(self._segment_length)

        if self._segment_overlap:
            params["max_overlap"] = str(self._segment_overlap)

        # Multipart file upload with explicit content type
        files = {
            "files": ("content.txt", text.encode("utf-8"), "text/plain"),
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._url}/extract",
                params=params,
                files=files,
            )

        # Kreuzberg returns 201 Created on success
        if resp.status_code != 201:
            detail = resp.text or httpx.codes.get_reason_phrase(resp.status_code)
            raise RuntimeError(
                f"Kreuzberg extract API error ({resp.status_code}): {detail}"
            )

        results: list[dict] = resp.json()

        if not results:
            return []

        # The first result contains chunks for our uploaded file.
        chunks: list[str] = results[0].get("chunks", [])

        return [Segment(text=c) for c in chunks if c]
