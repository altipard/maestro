"""POST /v1/extract — document text extraction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse

from maestro.config import Config
from maestro.core.models import FileData

from .models import ErrorDetail, ErrorResponse

router = APIRouter()


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=ErrorDetail(message=message)).model_dump(),
    )


@router.post("/extract")
async def extract(request: Request, file: UploadFile, model: str = "") -> JSONResponse:
    """Extract text from an uploaded document.

    Accepts multipart file upload. Returns extracted text as JSON.
    """
    cfg: Config = request.app.state.config

    try:
        extractor = cfg.extractor(model)
    except KeyError:
        return _error(400, f"extractor not found: {model or '(default)'}")

    content = await file.read()

    file_data = FileData(
        name=file.filename or "upload",
        content=content,
        content_type=file.content_type or "",
    )

    text = await extractor.extract(file_data)

    return JSONResponse(content={"text": text})
