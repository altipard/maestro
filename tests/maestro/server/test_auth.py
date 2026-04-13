"""Tests for auth middleware."""

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from maestro.server.auth import AuthMiddleware


def _make_app(tokens: list[str] | None = None) -> FastAPI:
    app = FastAPI()
    if tokens is not None:
        app.add_middleware(AuthMiddleware, tokens=tokens)

    @app.get("/test")
    async def test_endpoint():
        return JSONResponse({"ok": True})

    return app


@pytest.fixture
def open_app():
    return _make_app(tokens=[])


@pytest.fixture
def secured_app():
    return _make_app(tokens=["secret-token-123", "another-token"])


class TestAuthMiddleware:
    async def test_no_tokens_allows_all(self, open_app):
        async with AsyncClient(
            transport=ASGITransport(app=open_app), base_url="http://test"
        ) as client:
            resp = await client.get("/test")
            assert resp.status_code == 200

    async def test_valid_bearer_token(self, secured_app):
        async with AsyncClient(
            transport=ASGITransport(app=secured_app), base_url="http://test"
        ) as client:
            resp = await client.get("/test", headers={"Authorization": "Bearer secret-token-123"})
            assert resp.status_code == 200

    async def test_invalid_token_returns_401(self, secured_app):
        async with AsyncClient(
            transport=ASGITransport(app=secured_app), base_url="http://test"
        ) as client:
            resp = await client.get("/test", headers={"Authorization": "Bearer wrong-token"})
            assert resp.status_code == 401
            body = resp.json()
            assert body["error"]["type"] == "authentication_error"

    async def test_missing_auth_header_returns_401(self, secured_app):
        async with AsyncClient(
            transport=ASGITransport(app=secured_app), base_url="http://test"
        ) as client:
            resp = await client.get("/test")
            assert resp.status_code == 401

    async def test_non_bearer_auth_returns_401(self, secured_app):
        async with AsyncClient(
            transport=ASGITransport(app=secured_app), base_url="http://test"
        ) as client:
            resp = await client.get("/test", headers={"Authorization": "Basic abc123"})
            assert resp.status_code == 401

    async def test_second_valid_token(self, secured_app):
        async with AsyncClient(
            transport=ASGITransport(app=secured_app), base_url="http://test"
        ) as client:
            resp = await client.get("/test", headers={"Authorization": "Bearer another-token"})
            assert resp.status_code == 200
