"""Multi-provider fallback extractor.

Tries each extractor in order and returns the first successful result.
"""

from __future__ import annotations

from maestro.core.models import FileData
from maestro.core.protocols import Extractor


class MultiExtractor:
    """Tries multiple extractors in sequence, returning the first success."""

    def __init__(self, extractors: list[Extractor]) -> None:
        self._extractors = extractors

    async def extract(self, file: FileData) -> str:
        for extractor in self._extractors:
            try:
                return await extractor.extract(file)
            except Exception:
                continue

        raise ValueError("unsupported file type")
