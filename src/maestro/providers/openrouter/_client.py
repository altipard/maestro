"""Shared HTTP client for OpenRouter multimodal endpoints.

A thin HTTP helper that POSTs JSON to the chat completions endpoint
with error handling.
"""

from __future__ import annotations

from typing import Any

import httpx

from maestro.core.errors import ProviderError

_DEFAULT_URL = "https://openrouter.ai/api/v1"


async def post_chat(
    url: str,
    token: str,
    body: dict[str, Any],
    *,
    timeout: float = 120,
) -> dict[str, Any]:
    """POST to /chat/completions and return the parsed JSON response."""
    endpoint = url.rstrip("/") + "/chat/completions"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, json=body, headers=headers)

    if resp.status_code != 200:
        raise ProviderError(resp.status_code, resp.text)

    return resp.json()


def extract_message(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the first choice's message from a chat completion response."""
    choices = result.get("choices", [])
    if not choices:
        raise ProviderError(500, "no choices in response")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProviderError(500, "invalid choice in response")

    message = choice.get("message", {})
    if not isinstance(message, dict):
        raise ProviderError(500, "no message in response")

    return message
