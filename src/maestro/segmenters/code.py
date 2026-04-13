"""Code-aware segmenter using tree-sitter AST.

Uses tree-sitter to parse source code and splits at AST node boundaries
(function definitions, class declarations, etc.) for syntax-aware chunking.
Falls back to TextSegmenter if tree-sitter is not available or the language
is not supported.
"""

from __future__ import annotations

import os
from typing import Any

from maestro.core.models import Segment


class CodeSegmenter:
    """Syntax-aware code segmenter using tree-sitter.

    Requires ``tree-sitter`` and language grammars (e.g. ``tree-sitter-python``).
    Falls back to text-based splitting when tree-sitter is unavailable.
    """

    def __init__(
        self,
        segment_length: int = 1500,
        segment_overlap: int = 0,
        **_: object,
    ) -> None:
        self._segment_length = segment_length
        self._segment_overlap = segment_overlap
        self._has_tree_sitter = _check_tree_sitter()

    async def segment(self, text: str, filename: str = "") -> list[Segment]:
        if not text:
            return []

        chunks = self._split(text, filename)
        return [Segment(text=c) for c in chunks if c]

    def _split(self, text: str, filename: str) -> list[str]:
        if self._has_tree_sitter and filename:
            lang = _detect_language(filename)
            if lang is not None:
                return self._split_ast(text, lang)

        # Fallback: use language-aware separators
        return self._split_by_separators(text, filename)

    def _split_ast(self, text: str, language: Any) -> list[str]:
        """Split using tree-sitter AST node boundaries."""
        try:
            import tree_sitter
        except ImportError:
            return self._split_by_separators(text, "")

        parser = tree_sitter.Parser(language)
        tree = parser.parse(bytes(text, "utf-8"))
        root = tree.root_node

        # Collect named nodes at depth 1-2 (top-level declarations)
        sections = _collect_sections(root, max_depth=4)

        if not sections:
            return [text.strip()] if len(text) <= self._segment_length else self._split_by_separators(text, "")

        return self._build_chunks(text, sections)

    def _build_chunks(self, text: str, sections: list[tuple[int, int, int]]) -> list[str]:
        """Merge adjacent sections into chunks that fit within size limit."""
        chunks: list[str] = []
        cursor = 0

        for start, end, _depth in sections:
            if start < cursor:
                continue

            # If current section alone exceeds limit, split it
            section_text = text[cursor:end]
            if len(section_text) <= self._segment_length:
                # Try to merge with existing chunk
                if chunks and len(chunks[-1]) + len(section_text) <= self._segment_length:
                    chunks[-1] += section_text
                else:
                    chunk = section_text.strip()
                    if chunk:
                        chunks.append(chunk)
            else:
                # Large section — add what we have, then split the section
                remaining = section_text
                while remaining:
                    if len(remaining) <= self._segment_length:
                        chunk = remaining.strip()
                        if chunk:
                            chunks.append(chunk)
                        break

                    # Find a line break near the limit
                    split_at = remaining.rfind("\n", 0, self._segment_length)
                    if split_at <= 0:
                        split_at = self._segment_length

                    chunk = remaining[:split_at].strip()
                    if chunk:
                        chunks.append(chunk)
                    remaining = remaining[split_at:]

            cursor = end

        # Remaining text after last section
        if cursor < len(text):
            remaining = text[cursor:].strip()
            if remaining:
                if chunks and len(chunks[-1]) + len(remaining) + 1 <= self._segment_length:
                    chunks[-1] += "\n" + remaining
                else:
                    chunks.append(remaining)

        return chunks

    def _split_by_separators(self, text: str, filename: str) -> list[str]:
        """Fallback: split using language-aware separators."""
        from .text import TextSegmenter

        seg = TextSegmenter(
            segment_length=self._segment_length,
            segment_overlap=self._segment_overlap,
        )
        return seg._split(text)


def _collect_sections(
    node: Any, max_depth: int = 4, depth: int = 0,
) -> list[tuple[int, int, int]]:
    """Walk AST and collect named node boundaries as (start_byte, end_byte, depth)."""
    sections: list[tuple[int, int, int]] = []

    if depth > 0 and node.is_named:
        start = node.start_byte
        end = node.end_byte
        if start < end:
            sections.append((start, end, depth))

    if depth < max_depth:
        for child in node.children:
            sections.extend(_collect_sections(child, max_depth, depth + 1))

    return sections


def _check_tree_sitter() -> bool:
    """Check if tree-sitter is available."""
    try:
        import tree_sitter  # noqa: F401
        return True
    except ImportError:
        return False


# Extension → tree-sitter language module mapping
_LANG_MAP: dict[str, str] = {
    ".py": "tree_sitter_python",
    ".js": "tree_sitter_javascript",
    ".jsx": "tree_sitter_javascript",
    ".ts": "tree_sitter_typescript",
    ".tsx": "tree_sitter_typescript",
    ".go": "tree_sitter_go",
    ".rs": "tree_sitter_rust",
    ".rb": "tree_sitter_ruby",
    ".java": "tree_sitter_java",
    ".cs": "tree_sitter_c_sharp",
    ".cpp": "tree_sitter_cpp",
    ".c": "tree_sitter_c",
}


def _detect_language(filename: str) -> Any | None:
    """Detect tree-sitter language from filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    module_name = _LANG_MAP.get(ext)

    if not module_name:
        return None

    try:
        import importlib
        mod = importlib.import_module(module_name)
        return mod.language()
    except (ImportError, AttributeError):
        return None
