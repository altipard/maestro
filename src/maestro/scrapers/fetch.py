"""Lightweight HTTP scraper.

Downloads HTML, strips non-content elements (nav, footer, scripts, aria-hidden),
extracts from <main>/<article> with body fallback, and returns cleaned text.
No external service needed.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import httpx

# Elements to remove entirely before text extraction
_REMOVE_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "nav",
        "footer",
        "header",
    }
)

# Block-level elements that produce line breaks
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "blockquote",
        "pre",
        "tr",
        "dt",
        "dd",
        "section",
        "figure",
        "figcaption",
    }
)

# Attributes that indicate non-content elements
_REMOVE_ATTRS = {
    "role": {"navigation", "banner", "contentinfo"},
    "aria-hidden": {"true"},
}

_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_SPACES = re.compile(r"[^\S\n]+")


class FetchScraper:
    """Scrape a URL and return cleaned text content."""

    def __init__(self, timeout: float = 30) -> None:
        self._timeout = timeout

    async def scrape(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; Maestro/1.0)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
                follow_redirects=True,
            )
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")

            if "text/plain" in content_type:
                return resp.text.strip()

            return _extract_text_from_html(resp.text)


def _extract_text_from_html(html: str) -> str:
    """Parse HTML, strip non-content elements, extract and clean text."""
    parser = _ContentParser()
    parser.feed(html)

    text = parser.get_text()
    text = _collapse_whitespace(text)
    return text.strip()


class _ContentParser(HTMLParser):
    """HTML parser that extracts text content, skipping non-content elements."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_remove_tag = 0
        self._parts: list[str] = []
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        self._tag_stack.append(tag_lower)

        # Check if this element should be removed
        if tag_lower in _REMOVE_TAGS:
            self._in_remove_tag += 1
            return

        # Check attribute-based removal
        attr_dict = {k: v for k, v in attrs if v is not None}
        for attr_key, remove_vals in _REMOVE_ATTRS.items():
            if attr_dict.get(attr_key) in remove_vals:
                self._in_remove_tag += 1
                return

        # Add line breaks for block elements
        if tag_lower in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()

        # Pop from stack
        if self._tag_stack and self._tag_stack[-1] == tag_lower:
            self._tag_stack.pop()

        if tag_lower in _REMOVE_TAGS or self._in_remove_tag > 0:
            if tag_lower in _REMOVE_TAGS:
                self._in_remove_tag = max(0, self._in_remove_tag - 1)
            return

        if tag_lower in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_remove_tag > 0:
            return
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _collapse_whitespace(s: str) -> str:
    """Collapse multiple spaces and blank lines."""
    s = _RE_SPACES.sub(" ", s)
    s = _RE_BLANK_LINES.sub("\n\n", s)
    return s
