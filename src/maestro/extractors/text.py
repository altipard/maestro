"""Local text extractor.

Detects text files by extension, MIME type, or a printability heuristic,
then returns the content as normalised text.
"""

from __future__ import annotations

import os
import re
import unicodedata

from maestro.core.models import FileData

SUPPORTED_EXTENSIONS: set[str] = {
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".ini",
    ".log",
    ".md",
    ".rst",
}

SUPPORTED_MIME_TYPES: set[str] = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/tab-separated-values",
    "application/json",
    "application/xml",
    "application/yaml",
}


class TextExtractor:
    """Extracts text from plaintext files without any external service."""

    def __init__(self, **_: object) -> None:
        pass

    async def extract(self, file: FileData) -> str:
        if not _is_text(file):
            raise ValueError("unsupported file type")

        return _normalize(file.content.decode("utf-8", errors="replace"))


# ── Detection ──────────────────────────────────────────────────────────


def _is_text(file: FileData) -> bool:
    if _is_supported(file):
        return True
    return _is_printable(file.content)


def _is_supported(file: FileData) -> bool:
    if file.name:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return True

    if file.content_type:
        if file.content_type in SUPPORTED_MIME_TYPES:
            return True

    return False


def _is_printable(data: bytes) -> bool:
    if not data:
        return False

    printable_count = 0

    for b in data:
        if b == 0:
            return False
        ch = chr(b)
        if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t":
            printable_count += 1

    return printable_count > (len(data) * 90 // 100)


# ── Normalisation (mirrors Go text.Normalize) ─────────────────────────

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n\s*")
_SINGLE_BREAK = re.compile(r"\n\s*")


def _normalize(text: str) -> str:
    text = text.strip()
    text = text.replace("\a", "")
    text = text.replace("\r\n", "\n")

    # Use \a as temporary marker for paragraph breaks
    text = _PARAGRAPH_BREAK.sub("\a\a", text)
    # Use \a as temporary marker for single line breaks
    text = _SINGLE_BREAK.sub("\a", text)

    # Collapse multiple spaces into single space
    text = " ".join(text.split())

    # Restore line breaks from temporary markers
    text = text.replace("\a", "\n")

    return text.strip()
