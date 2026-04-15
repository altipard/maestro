"""Coverage for all four OpenAI-spec embedding input shapes.

The Embeddings API accepts ``string | string[] | int[] | int[][]``.
The int-array variants are token IDs produced by tiktoken-based clients
(OpenAI SDK, LangChain ``OpenAIEmbeddings``). Maestro decodes these back
to text at the HTTP edge before reaching any provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import tiktoken
from fastapi.testclient import TestClient

from maestro.config import Config
from maestro.core.models import EmbedOptions, Embedding, Usage
from maestro.server import create_app
from maestro.server.openai.tokens import (
    DEFAULT_ENCODING,
    normalize_embedding_input,
)


class _RecordingEmbedder:
    """Captures the texts it was asked to embed, for assertion."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(
        self, texts: list[str], options: EmbedOptions | None = None
    ) -> Embedding:
        self.calls.append(list(texts))
        return Embedding(
            model="mock-embed",
            embeddings=[[0.1, 0.2, 0.3] for _ in texts],
            usage=Usage(input_tokens=99, output_tokens=0),
        )


def _client(embedder: _RecordingEmbedder) -> TestClient:
    cfg = Config()
    cfg.register("embedder", "mock-embed", embedder)
    return TestClient(create_app(config=cfg))


# ── normalize_embedding_input (unit) ───────────────────────────────


class TestNormalizeEmbeddingInput:
    def test_string(self) -> None:
        strs, count = normalize_embedding_input("hello")
        assert strs == ["hello"]
        assert count is None

    def test_list_of_strings(self) -> None:
        strs, count = normalize_embedding_input(["a", "b"])
        assert strs == ["a", "b"]
        assert count is None

    def test_token_ids_single(self) -> None:
        enc = tiktoken.get_encoding(DEFAULT_ENCODING)
        text = "hello world"
        ids = enc.encode(text)

        strs, count = normalize_embedding_input(ids)

        assert strs == [text]
        assert count == len(ids)

    def test_token_ids_batched(self) -> None:
        enc = tiktoken.get_encoding(DEFAULT_ENCODING)
        texts = ["hello", "goodbye"]
        batches = [enc.encode(t) for t in texts]

        strs, count = normalize_embedding_input(batches)

        assert strs == texts
        assert count == sum(len(b) for b in batches)

    def test_empty_string(self) -> None:
        strs, count = normalize_embedding_input("")
        assert strs == [""]
        assert count is None

    def test_empty_list(self) -> None:
        strs, count = normalize_embedding_input([])
        assert strs == []
        assert count is None

    def test_invalid_nested_mixed(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            normalize_embedding_input([[1, "not-an-int"]])  # type: ignore[list-item]

    def test_round_trip(self) -> None:
        """decode(encode(text)) should reproduce the original text."""
        enc = tiktoken.get_encoding(DEFAULT_ENCODING)
        original = "The quick brown fox jumps over the lazy dog."
        strs, _ = normalize_embedding_input(enc.encode(original))
        assert strs == [original]


# ── /v1/embeddings (integration) ───────────────────────────────────


class TestEmbeddingsEndpoint:
    def test_accepts_token_ids_single(self) -> None:
        """Regression: LangChain OpenAIEmbeddings sends pre-tokenized input.

        Before the fix, Maestro returned 422 because the Pydantic model
        rejected ``list[int]``. This is the bug that broke LibreChat's
        RAG pipeline when pointed at Maestro as OpenAI-compat backend.
        """
        enc = tiktoken.get_encoding(DEFAULT_ENCODING)
        text = "Maestro accepts pre-tokenized input"
        ids = enc.encode(text)

        embedder = _RecordingEmbedder()
        resp = _client(embedder).post(
            "/v1/embeddings",
            json={"model": "mock-embed", "input": ids},
        )

        assert resp.status_code == 200, resp.text
        assert embedder.calls == [[text]]
        assert resp.json()["usage"]["prompt_tokens"] == len(ids)

    def test_accepts_token_ids_batched(self) -> None:
        """LangChain batches chunks as list[list[int]] — same as OpenAI SDK."""
        enc = tiktoken.get_encoding(DEFAULT_ENCODING)
        texts = ["first chunk", "second chunk", "third chunk"]
        batches = [enc.encode(t) for t in texts]

        embedder = _RecordingEmbedder()
        resp = _client(embedder).post(
            "/v1/embeddings",
            json={"model": "mock-embed", "input": batches},
        )

        assert resp.status_code == 200, resp.text
        assert embedder.calls == [texts]
        data = resp.json()
        assert len(data["data"]) == 3
        assert data["usage"]["prompt_tokens"] == sum(len(b) for b in batches)

    def test_still_accepts_plain_strings(self) -> None:
        """Existing string / list[str] clients must keep working."""
        embedder = _RecordingEmbedder()
        resp = _client(embedder).post(
            "/v1/embeddings",
            json={"model": "mock-embed", "input": ["alpha", "beta"]},
        )

        assert resp.status_code == 200
        assert embedder.calls == [["alpha", "beta"]]

    def test_invalid_input_shape_returns_400(self) -> None:
        """Mixed/invalid nested arrays produce a 400, not a 500."""
        # Pydantic will reject most malformed shapes at the model layer (422),
        # which is fine. This guards the specific case where the model passes
        # but normalization still catches structural problems.
        embedder = _RecordingEmbedder()
        resp = _client(embedder).post(
            "/v1/embeddings",
            json={"model": "mock-embed", "input": 42},  # wrong top-level type
        )
        assert resp.status_code in (400, 422)
