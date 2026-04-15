"""Tokenizer service for the OpenAI-compatible API edge.

The OpenAI Embeddings and (legacy) Completions APIs accept pre-tokenized input
as integer arrays. Maestro's internal embedder protocol is text-only — so the
HTTP edge decodes token IDs back to strings before dispatching to providers.

This module is the single place that knows about tiktoken; the rest of the
codebase stays free of transport-layer concerns.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def decode_tokens(ids: list[int], encoding: str = DEFAULT_ENCODING) -> str:
    """Decode a token-ID sequence to its string representation."""
    return _encoding(encoding).decode(ids)


def normalize_embedding_input(
    inp: str | list[str] | list[int] | list[list[int]],
    encoding: str = DEFAULT_ENCODING,
) -> tuple[list[str], int | None]:
    """Normalize any OpenAI-spec embedding input shape to a list of strings.

    Returns ``(strings, known_token_count)`` where ``known_token_count`` is set
    only when the input was pre-tokenized — in which case the count is exact
    and should be preferred over re-tokenizing for usage accounting.

    Accepts all four shapes from the OpenAI Embeddings API spec:
        - ``str``              — single text
        - ``list[str]``        — batch of texts
        - ``list[int]``        — single pre-tokenized input (token IDs)
        - ``list[list[int]]``  — batch of pre-tokenized inputs
    """
    if isinstance(inp, str):
        return [inp], None

    if not isinstance(inp, list) or not inp:
        return [], None

    first = inp[0]

    if isinstance(first, str):
        return inp, None  # type: ignore[return-value]

    if isinstance(first, int):
        ids: list[int] = inp  # type: ignore[assignment]
        return [decode_tokens(ids, encoding)], len(ids)

    if isinstance(first, list):
        batches: list[list[int]] = inp  # type: ignore[assignment]
        for row in batches:
            if not all(isinstance(x, int) for x in row):
                raise ValueError("nested arrays must contain only integer token IDs")
        total = sum(len(row) for row in batches)
        return [decode_tokens(row, encoding) for row in batches], total

    raise ValueError(f"unsupported input element type: {type(first).__name__}")
