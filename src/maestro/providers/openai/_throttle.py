"""Proactive rate-limit throttle transport.

Wraps an httpx transport, reads rate-limit headers from every response,
and delays subsequent requests when remaining capacity drops below 5%.
Prevents 429s proactively instead of retrying after the fact.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass


class ThrottleTransport(httpx.AsyncBaseTransport):
    """Async transport wrapper that throttles based on rate-limit headers."""

    def __init__(self, base: httpx.AsyncBaseTransport | None = None) -> None:
        self._base = base or httpx.AsyncHTTPTransport()
        self._wait_until: float = 0  # monotonic timestamp
        self._lock = asyncio.Lock()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Wait if we need to throttle
        async with self._lock:
            wait = self._wait_until - time.monotonic()

        if wait > 0:
            await asyncio.sleep(wait)

        response = await self._base.handle_async_request(request)
        self._observe(response.headers)
        return response

    def _observe(self, headers: httpx.Headers) -> None:
        delay = 0.0

        for dimension in ("requests", "tokens"):
            d = _check_limit(headers, dimension)
            if d > delay:
                delay = d

        if delay <= 0:
            return

        # Cap at 60 seconds (same as Go)
        if delay > 60:
            delay = 60

        until = time.monotonic() + delay
        if until > self._wait_until:
            self._wait_until = until

    async def aclose(self) -> None:
        await self._base.aclose()


def _check_limit(headers: httpx.Headers, dimension: str) -> float:
    """Check a single rate-limit dimension, return delay in seconds if low."""
    remaining = _header_int(headers.get(f"x-ratelimit-remaining-{dimension}", ""))

    if remaining < 0:
        return 0

    limit = _header_int(headers.get(f"x-ratelimit-limit-{dimension}", ""))

    if not _is_low(remaining, limit):
        return 0

    delay = _header_duration(headers.get(f"x-ratelimit-reset-{dimension}", ""))

    if delay <= 0:
        delay = 1.0

    return delay


def _is_low(remaining: int, limit: int) -> bool:
    """True if remaining capacity is critically low relative to limit."""
    if remaining <= 1:
        return True
    if limit > 0:
        return remaining < limit // 20  # less than 5%
    return False


def _header_int(value: str) -> int:
    """Parse an integer header, returning -1 on failure."""
    if not value:
        return -1
    try:
        return int(value)
    except ValueError:
        return -1


def _header_duration(value: str) -> float:
    """Parse a duration header (Go-style '1s', '6m0s' or plain seconds)."""
    if not value:
        return 0

    # Try parsing as Go-style duration (e.g. "1s", "6m0s", "500ms")
    total = 0.0
    remaining = value
    parsed_any = False

    while remaining:
        # Find first non-digit/non-dot character
        i = 0
        while i < len(remaining) and (remaining[i].isdigit() or remaining[i] == '.'):
            i += 1

        if i == 0:
            break

        num_str = remaining[:i]
        unit = remaining[i:]

        try:
            num = float(num_str)
        except ValueError:
            break

        if unit.startswith("ms"):
            total += num / 1000
            remaining = unit[2:]
        elif unit.startswith("m") and not unit.startswith("ms"):
            total += num * 60
            remaining = unit[1:]
        elif unit.startswith("s"):
            total += num
            remaining = unit[1:]
        elif unit.startswith("h"):
            total += num * 3600
            remaining = unit[1:]
        else:
            # Plain number — treat as seconds
            total += num
            remaining = unit
            parsed_any = True
            break

        parsed_any = True

    if parsed_any:
        return total

    # Fallback: try plain float seconds
    try:
        return float(value)
    except ValueError:
        return 0
