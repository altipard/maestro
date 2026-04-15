#!/usr/bin/env python3
"""Minimal RAG CLI on top of Maestro.

Uses only Maestro's HTTP API (extract, segment, embeddings, chat) plus
a ChromaDB HTTP client. No framework, no magic — the full pipeline fits
on one screen.

Flow:
    ingest  : file -> /v1/extract -> /v1/segment -> /v1/embeddings -> chroma
    ask     : question -> /v1/embeddings -> chroma.query -> /v1/chat/completions
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import chromadb
import httpx

MAESTRO = "http://localhost:8080/v1"
CHROMA_HOST, CHROMA_PORT = "localhost", 8000
COLLECTION = "docs"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3.5:latest"


def chroma():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT).get_or_create_collection(
        COLLECTION
    )


def ingest(path: str) -> None:
    p = pathlib.Path(path)
    if not p.is_file():
        sys.exit(f"not a file: {p}")

    with httpx.Client(timeout=300) as h:
        with p.open("rb") as f:
            r = h.post(f"{MAESTRO}/extract", files={"file": (p.name, f)})
            r.raise_for_status()
            text = r.json()["text"]

        r = h.post(f"{MAESTRO}/segment", json={"text": text})
        r.raise_for_status()
        chunks = [s["text"] for s in r.json()["segments"]]
        if not chunks:
            sys.exit("segmenter returned no chunks")

        r = h.post(
            f"{MAESTRO}/embeddings",
            json={"model": EMBED_MODEL, "input": chunks},
        )
        r.raise_for_status()
        vectors = [d["embedding"] for d in r.json()["data"]]

    ids = [f"{p.name}#{i}" for i in range(len(chunks))]
    chroma().upsert(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=[{"source": p.name, "chunk": i} for i in range(len(chunks))],
    )
    print(f"indexed {len(chunks)} chunks from {p.name}")


def ask(question: str, k: int = 4) -> None:
    with httpx.Client(timeout=120) as h:
        r = h.post(
            f"{MAESTRO}/embeddings",
            json={"model": EMBED_MODEL, "input": [question]},
        )
        r.raise_for_status()
        qv = r.json()["data"][0]["embedding"]

        hits = chroma().query(query_embeddings=[qv], n_results=k)
        docs = hits["documents"][0]
        metas = hits["metadatas"][0]
        if not docs:
            sys.exit("index is empty — run `ingest` first")

        context = "\n\n---\n\n".join(
            f"[{m['source']}#{m['chunk']}]\n{d}" for d, m in zip(docs, metas)
        )

        r = h.post(
            f"{MAESTRO}/chat/completions",
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer the user using only the context below. "
                            "Cite sources inline as [filename#chunk]. "
                            "If the context does not contain the answer, say so."
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            },
        )
        r.raise_for_status()
        print(r.json()["choices"][0]["message"]["content"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal RAG CLI on top of Maestro")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="index a file")
    ing.add_argument("file")

    q = sub.add_parser("ask", help="ask a question against the index")
    q.add_argument("question")
    q.add_argument("-k", type=int, default=4, help="number of chunks to retrieve")

    args = ap.parse_args()
    if args.cmd == "ingest":
        ingest(args.file)
    elif args.cmd == "ask":
        ask(args.question, args.k)


if __name__ == "__main__":
    main()
