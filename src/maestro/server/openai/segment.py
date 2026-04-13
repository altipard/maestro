"""POST /v1/segment — text segmentation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from maestro.config import Config

from .models import ErrorDetail, ErrorResponse

router = APIRouter()


class SegmentRequest(BaseModel):
    text: str
    model: str = ""


class SegmentItem(BaseModel):
    text: str


class SegmentResponse(BaseModel):
    segments: list[SegmentItem] = Field(default_factory=list)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=ErrorDetail(message=message)).model_dump(),
    )


@router.post("/segment")
async def segment(req: SegmentRequest, request: Request) -> JSONResponse:
    """Segment text into chunks for RAG pipelines.

    Chunk size and overlap are configured via the YAML config file.
    """
    cfg: Config = request.app.state.config

    try:
        segmenter = cfg.segmenter(req.model)
    except KeyError:
        return _error(400, f"segmenter not found: {req.model or '(default)'}")

    if not req.text:
        return _error(400, "no text provided")

    segments = await segmenter.segment(req.text)

    return JSONResponse(
        content=SegmentResponse(
            segments=[SegmentItem(text=s.text) for s in segments],
        ).model_dump(),
    )
