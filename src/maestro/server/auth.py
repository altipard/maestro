"""Bearer token authentication middleware.

If no tokens are configured, all requests pass through (open access).
Otherwise, requests must include a valid Bearer token.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


class AuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that validates Bearer tokens."""

    def __init__(self, app, tokens: list[str] | None = None) -> None:
        super().__init__(app)
        self._tokens = set(tokens) if tokens else set()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # No tokens configured = open access (same as Go: len(s.Authorizers) == 0)
        if not self._tokens:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token in self._tokens:
                return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid or missing authentication token",
                }
            },
        )
