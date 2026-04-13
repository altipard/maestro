"""Jina Segment API segmenter."""

from __future__ import annotations

import httpx

from maestro.core.models import Segment

_DEFAULT_URL = "https://segment.jina.ai/"


class JinaSegmenter:
    """Segments text via the Jina Segment API (POST JSON)."""

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

        body = {
            "content": text,
            "return_chunks": True,
            "max_chunk_length": self._segment_length,
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        async with httpx.AsyncClient() as client:
            resp = await client.post(self._url, json=body, headers=headers)

        if resp.status_code != 200:
            detail = resp.text or httpx.codes.get_reason_phrase(resp.status_code)
            raise RuntimeError(f"Jina segment API error ({resp.status_code}): {detail}")

        data = resp.json()
        chunks: list[str] = data.get("chunks", [])

        return [Segment(text=c) for c in chunks if c]
