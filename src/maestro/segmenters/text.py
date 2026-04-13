"""Local text segmenter with markdown-aware and code-language-aware splitting."""

from __future__ import annotations

import os
import re

from maestro.core.models import Segment

# ---------------------------------------------------------------------------
# Language-specific separators (mirrors the Go text_separators.go)
# ---------------------------------------------------------------------------

_LANGUAGE_SEPARATORS: dict[str, list[str]] = {
    ".cs": [
        "\ninterface ",
        "\nenum ",
        "\nimplements ",
        "\ndelegate ",
        "\nevent ",
        "\nclass ",
        "\nabstract ",
        "\npublic ",
        "\nprotected ",
        "\nprivate ",
        "\nstatic ",
        "\nreturn ",
        "\nif ",
        "\ncontinue ",
        "\nfor ",
        "\nforeach ",
        "\nwhile ",
        "\nswitch ",
        "\nbreak ",
        "\ncase ",
        "\nelse ",
        "\ntry ",
        "\nthrow ",
        "\nfinally ",
        "\ncatch ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".cpp": [
        "\nclass ",
        "\nvoid ",
        "\nint ",
        "\nfloat ",
        "\ndouble ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".go": [
        "\nfunc ",
        "\nvar ",
        "\nconst ",
        "\ntype ",
        "\nif ",
        "\nfor ",
        "\nswitch ",
        "\ncase ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".java": [
        "\nclass ",
        "\npublic ",
        "\nprotected ",
        "\nprivate ",
        "\nstatic ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".kt": [
        "\nclass ",
        "\npublic ",
        "\nprotected ",
        "\nprivate ",
        "\ninternal ",
        "\ncompanion ",
        "\nfun ",
        "\nval ",
        "\nvar ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nwhen ",
        "\ncase ",
        "\nelse ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".js": [
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\nclass ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\ndefault ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".jsm": [
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\nclass ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\ndefault ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".ts": [
        "\nenum ",
        "\ninterface ",
        "\nnamespace ",
        "\ntype ",
        "\nclass ",
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\ndefault ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".tsx": [
        "\nenum ",
        "\ninterface ",
        "\nnamespace ",
        "\ntype ",
        "\nclass ",
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nswitch ",
        "\ncase ",
        "\ndefault ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".py": [
        "\nclass ",
        "\ndef ",
        "\n\tdef ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".rb": [
        "\ndef ",
        "\nclass ",
        "\nif ",
        "\nunless ",
        "\nwhile ",
        "\nfor ",
        "\ndo ",
        "\nbegin ",
        "\nrescue ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".rs": [
        "\nfn ",
        "\nconst ",
        "\nlet ",
        "\nif ",
        "\nwhile ",
        "\nfor ",
        "\nloop ",
        "\nmatch ",
        "\nconst ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".sc": [
        "\nclass ",
        "\nobject ",
        "\ndef ",
        "\nval ",
        "\nvar ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nmatch ",
        "\ncase ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".scala": [
        "\nclass ",
        "\nobject ",
        "\ndef ",
        "\nval ",
        "\nvar ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\nmatch ",
        "\ncase ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    ".swift": [
        "\nfunc ",
        "\nclass ",
        "\nstruct ",
        "\nenum ",
        "\nif ",
        "\nfor ",
        "\nwhile ",
        "\ndo ",
        "\nswitch ",
        "\ncase ",
        "\n\n",
        "\n",
        " ",
        "",
    ],
}

# ---------------------------------------------------------------------------
# Markdown detection (mirrors the Go markdown.go heuristics)
# ---------------------------------------------------------------------------

_RE_ATX_HEADING = re.compile(r"(?m)^#{1,6}\s+.+$")
_RE_CODE_FENCE = re.compile(r"(?m)^```|^~~~")
_RE_UNORDERED_LIST = re.compile(r"(?m)^\s*[-*+]\s+.+$")
_RE_ORDERED_LIST = re.compile(r"(?m)^\s*\d+\.\s+.+$")
_RE_LINK = re.compile(r"!?\[([^\]]+)\]\(([^)]+)\)")
_RE_BLOCKQUOTE = re.compile(r"(?m)^>\s+.+$")
_RE_HORIZONTAL_RULE = re.compile(r"(?m)^\s*(-{3,}|\*{3,}|_{3,})\s*$")


def _is_markdown(text: str) -> bool:
    """Return True when *text* looks like markdown (>=2 distinct indicators)."""
    if not text:
        return False
    indicators = sum(
        [
            bool(_RE_ATX_HEADING.search(text)),
            bool(_RE_CODE_FENCE.search(text)),
            bool(_RE_UNORDERED_LIST.search(text) or _RE_ORDERED_LIST.search(text)),
            bool(_RE_LINK.search(text)),
            bool(_RE_BLOCKQUOTE.search(text)),
            bool(_RE_HORIZONTAL_RULE.search(text)),
        ]
    )
    return indicators >= 2


# ---------------------------------------------------------------------------
# Recursive character text splitter (pure-Python fallback)
# ---------------------------------------------------------------------------


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split *text* recursively along *separators*, largest first."""
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Pick the first separator that appears in the text.
    chosen_sep = ""
    remaining_seps = separators
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            chosen_sep = sep
            remaining_seps = separators[i + 1 :]
            break

    # Split with the chosen separator.
    if chosen_sep == "":
        # Character-level split.
        parts: list[str] = []
        for i in range(0, len(text), max(chunk_size - chunk_overlap, 1)):
            chunk = text[i : i + chunk_size].strip()
            if chunk:
                parts.append(chunk)
        return parts

    pieces = text.split(chosen_sep)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = len(piece) + (len(chosen_sep) if current else 0)
        if current and current_len + piece_len > chunk_size:
            merged = chosen_sep.join(current).strip()
            if merged:
                chunks.append(merged)

            # Handle overlap by keeping trailing pieces.
            if chunk_overlap > 0:
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) > chunk_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + len(chosen_sep)
                current = overlap_parts
                current_len = sum(len(p) for p in current) + len(chosen_sep) * max(
                    len(current) - 1, 0
                )
            else:
                current = []
                current_len = 0

        current.append(piece)
        current_len += piece_len

    if current:
        merged = chosen_sep.join(current).strip()
        if merged:
            # If the last chunk is still too large, recurse with finer separators.
            if len(merged) > chunk_size and remaining_seps:
                chunks.extend(_recursive_split(merged, remaining_seps, chunk_size, chunk_overlap))
            else:
                chunks.append(merged)

    return chunks


# ---------------------------------------------------------------------------
# Public segmenter class
# ---------------------------------------------------------------------------


class TextSegmenter:
    """Local text segmenter with markdown and code-language awareness.

    When ``semantic_text_splitter`` is installed, uses its MarkdownSplitter /
    TextSplitter for higher-quality semantic splitting.  Otherwise falls back
    to a recursive character-based splitter.
    """

    def __init__(
        self,
        url: str = "",
        token: str = "",
        segment_length: int = 1000,
        segment_overlap: int = 0,
        **_: object,
    ) -> None:
        self._segment_length = segment_length
        self._segment_overlap = segment_overlap

        # Attempt to import semantic_text_splitter at init time.
        try:
            from semantic_text_splitter import (  # noqa: N814
                MarkdownSplitter,
                TextSplitter,
            )

            self._semantic_md = MarkdownSplitter
            self._semantic_txt = TextSplitter
            self._has_semantic = True
        except ImportError:
            self._has_semantic = False

    # ------------------------------------------------------------------

    async def segment(self, text: str) -> list[Segment]:
        if not text:
            return []

        chunks = self._split(text)
        return [Segment(text=c) for c in chunks if c]

    # ------------------------------------------------------------------

    def _split(self, text: str) -> list[str]:
        """Choose the best splitter and return chunks."""
        if self._has_semantic:
            return self._split_semantic(text)
        return self._split_fallback(text)

    def _split_semantic(self, text: str) -> list[str]:
        """Use the Rust-backed semantic_text_splitter library."""
        if _is_markdown(text):
            splitter = self._semantic_md(self._segment_length)
        else:
            splitter = self._semantic_txt(self._segment_length)

        return splitter.chunks(text, self._segment_overlap)

    def _split_fallback(self, text: str) -> list[str]:
        """Pure-Python recursive character splitter."""
        if _is_markdown(text):
            separators = [
                "\n# ",
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n##### ",
                "\n###### ",
                "\n---",
                "\n***",
                "\n___",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        else:
            separators = ["\n\n", "\n", " ", ""]

        return _recursive_split(
            text,
            separators,
            self._segment_length,
            self._segment_overlap,
        )


def _get_language_separators(filename: str) -> list[str] | None:
    """Return language-specific separators based on file extension."""
    if not filename:
        return None
    ext = os.path.splitext(filename)[1].lower()
    return _LANGUAGE_SEPARATORS.get(ext)
