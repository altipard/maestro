"""POST /v1/embeddings — embedding generation endpoint."""

from __future__ import annotations

import base64
import struct

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from maestro.config import Config
from maestro.core.models import EmbedOptions

from .models import (
    ChatCompletionUsage,
    EmbeddingData,
    EmbeddingList,
    EmbeddingsRequest,
    ErrorDetail,
    ErrorResponse,
)

router = APIRouter()


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=ErrorDetail(message=message)).model_dump(),
    )


@router.post("/embeddings")
async def create_embeddings(req: EmbeddingsRequest, request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config

    try:
        embedder = cfg.embedder(req.model)
    except KeyError:
        return _error(400, f"model not found: {req.model}")

    # Normalize input to list
    if isinstance(req.input, str):
        inputs = [req.input]
    else:
        inputs = req.input

    if not inputs:
        return _error(400, "no input provided")

    options: EmbedOptions | None = None
    if req.dimensions is not None:
        options = EmbedOptions(dimensions=req.dimensions)

    embedding = await embedder.embed(inputs, options)

    result = EmbeddingList(
        model=embedding.model or req.model,
    )

    use_base64 = req.encoding_format == "base64"

    for i, e in enumerate(embedding.embeddings):
        data = EmbeddingData(index=i)

        if use_base64:
            data.embedding = _floats_to_base64(e)
        else:
            data.embedding = e

        result.data.append(data)

    if embedding.usage:
        result.usage = ChatCompletionUsage(
            prompt_tokens=embedding.usage.input_tokens,
            total_tokens=embedding.usage.input_tokens + embedding.usage.output_tokens,
        )

    return JSONResponse(content=result.model_dump(exclude_none=True))


def _floats_to_base64(floats: list[float]) -> str:
    """Encode float list as little-endian base64 (matches Go's floatsToBase64)."""
    buf = struct.pack(f"<{len(floats)}f", *floats)
    return base64.b64encode(buf).decode()
