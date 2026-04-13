"""GET /v1/models and GET /v1/models/{model_id} — model listing."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from maestro.config import Config

from .models import ErrorDetail, ErrorResponse, ModelList, ModelObject

router = APIRouter()


@router.get("/models")
async def list_models(request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config

    now = int(time.time())
    result = ModelList()

    for model_id in cfg.models():
        result.data.append(ModelObject(id=model_id, created=now, owned_by="maestro"))

    return JSONResponse(content=result.model_dump())


@router.get("/models/{model_id}")
async def get_model(model_id: str, request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config

    if model_id not in cfg.models():
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(type="not_found", message=f"model not found: {model_id}"),
            ).model_dump(),
        )

    return JSONResponse(
        content=ModelObject(
            id=model_id,
            created=int(time.time()),
            owned_by="maestro",
        ).model_dump(),
    )
