"""FastAPI application — Maestro LLM platform server.

- CORS middleware (allow all)
- /v1 prefix for OpenAI-compatible endpoints
- Config wiring via app.state
- FastAPI handles JSON serialization, validation, OpenAPI docs
- Dependency injection via app.state
- SSE streaming via StreamingResponse
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from maestro.config import Config, load

from .auth import AuthMiddleware
from .openai.chat import router as chat_router
from .openai.embeddings import router as embeddings_router
from .openai.extract import router as extract_router
from .openai.models_handler import router as models_router
from .openai.responses import router as responses_router
from .openai.segment import router as segment_router


def create_app(config: Config | None = None, config_path: str | Path | None = None) -> FastAPI:
    """Create the FastAPI application with all routes wired.

    Either pass a pre-built Config or a path to a YAML config file.
    """
    if config is None and config_path is not None:
        config = load(config_path)
    elif config is None:
        config = Config()

    app = FastAPI(title="Maestro", version="0.1.0")

    # Store config for route handlers
    app.state.config = config

    # Auth middleware — if tokens configured, require Bearer auth
    auth_tokens = getattr(config, "auth_tokens", None)
    if auth_tokens:
        app.add_middleware(AuthMiddleware, tokens=auth_tokens)

    # CORS — match Go's cors.Options{AllowedOrigins: []string{"*"}}
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=300,
    )

    # Mount OpenAI-compatible routes under /v1
    app.include_router(chat_router, prefix="/v1")
    app.include_router(embeddings_router, prefix="/v1")
    app.include_router(extract_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(responses_router, prefix="/v1")
    app.include_router(segment_router, prefix="/v1")

    return app
