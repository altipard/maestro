"""ChromaDB vector store provider.

Supports two modes via the ``url`` constructor parameter:
- **Embedded** (url empty): ``chromadb.PersistentClient`` — no extra service needed.
- **Client/Server** (url set): ``chromadb.HttpClient`` — connects to a running Chroma instance.

The ``model`` parameter is used as the collection name (default: ``"default"``).
"""

from __future__ import annotations

import asyncio
from typing import Any

from maestro.core.models import SearchResult
from maestro.providers.registry import provider


@provider("chroma", "vectorstore")
class VectorStore:
    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        import chromadb

        if url:
            self._client = chromadb.HttpClient(host=url)
        else:
            self._client = chromadb.PersistentClient(path=".vectorstore")

        self._collection = self._client.get_or_create_collection(
            name=model or "default",
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {"ids": ids, "embeddings": embeddings}
        if documents is not None:
            kwargs["documents"] = documents
        if metadatas is not None:
            kwargs["metadatas"] = metadatas

        await asyncio.to_thread(self._collection.upsert, **kwargs)

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": limit,
        }
        if filters:
            kwargs["where"] = filters

        results = await asyncio.to_thread(self._collection.query, **kwargs)

        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        return [
            SearchResult(
                title=meta.get("source", "") if meta else "",
                content=doc or "",
                score=round(1.0 - dist, 4),
            )
            for doc, dist, meta in zip(docs, distances, metas)
        ]

    async def delete(self, ids: list[str]) -> None:
        await asyncio.to_thread(self._collection.delete, ids=ids)

    async def count(self) -> int:
        return await asyncio.to_thread(self._collection.count)
