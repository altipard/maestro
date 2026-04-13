from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from .models import (
    CompleteOptions,
    Completion,
    Embedding,
    EmbedOptions,
    FileData,
    Message,
    RankedDocument,
    SearchResult,
    Segment,
)

# ── Core AI capabilities ────────────────────────────────────────────


@runtime_checkable
class Completer(Protocol):
    def complete(
        self,
        messages: list[Message],
        options: CompleteOptions | None = None,
    ) -> AsyncIterator[Completion]: ...


@runtime_checkable
class Embedder(Protocol):
    async def embed(
        self,
        texts: list[str],
        options: EmbedOptions | None = None,
    ) -> Embedding: ...


@runtime_checkable
class Renderer(Protocol):
    async def render(self, prompt: str) -> FileData: ...


@runtime_checkable
class Synthesizer(Protocol):
    async def synthesize(self, text: str) -> FileData: ...


@runtime_checkable
class Transcriber(Protocol):
    async def transcribe(self, file: FileData) -> str: ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[RankedDocument]: ...


# ── Pipeline capabilities (satellite project interfaces) ────────────
#
# These protocols define the contracts that maestro-extract,
# maestro-segment, and other satellite packages implement.


@runtime_checkable
class Extractor(Protocol):
    async def extract(self, file: FileData) -> str: ...


@runtime_checkable
class Segmenter(Protocol):
    async def segment(self, text: str) -> list[Segment]: ...


@runtime_checkable
class Searcher(Protocol):
    async def search(self, query: str, *, limit: int = 10) -> list[SearchResult]: ...


@runtime_checkable
class Summarizer(Protocol):
    async def summarize(self, text: str) -> str: ...


@runtime_checkable
class Translator(Protocol):
    async def translate(self, text: str, target: str) -> str: ...


# ── Storage capabilities ──────────────────────────────────────────


@runtime_checkable
class VectorStore(Protocol):
    """Persistent vector storage for document embeddings.

    Constructor parameter semantics (following provider convention):
      - url:   connection URL (empty = embedded/local mode)
      - token: auth token (empty = no authentication)
      - model: collection/index name
    """

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None: ...

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    async def delete(self, ids: list[str]) -> None: ...

    async def count(self) -> int: ...
