"""Generic middleware wrappers for any provider type.

This is where Python beats Go. In Go, you need separate wrapper types
for each capability:

    limiter.NewCompleter(limiter, completer)
    limiter.NewEmbedder(limiter, embedder)
    limiter.NewRenderer(limiter, renderer)
    otel.NewCompleter("openai", id, completer)
    otel.NewEmbedder("openai", id, embedder)
    ...

In Python, ONE wrapper handles ALL provider types via __getattr__ proxying.
It auto-detects async generators (streaming) vs coroutines and wraps both.

Usage:
    completer = OpenAICompleter(url=..., token=..., model=...)
    completer = RateLimited(completer, rate=10)
    completer = Traced(completer, vendor="openai", model="gpt-4o")

    # Or use the convenience function:
    completer = wrap(completer, vendor="openai", model="gpt-4o", rate_limit=10)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from typing import Any


class _TokenBucket:
    """Simple async token-bucket rate limiter."""

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate)
            self._last = now

            if self._tokens < 1:
                await asyncio.sleep((1 - self._tokens) / self._rate)
                self._tokens = 0
            else:
                self._tokens -= 1


class RateLimited:
    """Rate-limit any provider. Works with completers, embedders, renderers — anything.

    Transparently proxies all attributes. Async methods and async generators
    are automatically wrapped with rate limiting.
    """

    def __init__(self, inner: Any, rate: float) -> None:
        self._inner = inner
        self._bucket = _TokenBucket(rate)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)

        if inspect.ismethod(attr) or inspect.isfunction(attr):
            if inspect.isasyncgenfunction(attr):

                @functools.wraps(attr)
                async def gen_wrapper(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
                    await self._bucket.acquire()
                    async for item in attr(*args, **kwargs):
                        yield item

                return gen_wrapper

            if asyncio.iscoroutinefunction(attr):

                @functools.wraps(attr)
                async def coro_wrapper(*args: Any, **kwargs: Any) -> Any:
                    await self._bucket.acquire()
                    return await attr(*args, **kwargs)

                return coro_wrapper

        return attr


class Traced:
    """Add observability to any provider. Plugs into OpenTelemetry when available.

    Same generic proxy approach as RateLimited.
    """

    def __init__(self, inner: Any, vendor: str, model: str) -> None:
        self._inner = inner
        self._vendor = vendor
        self._model = model

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)

        if inspect.ismethod(attr) or inspect.isfunction(attr):
            if inspect.isasyncgenfunction(attr):

                @functools.wraps(attr)
                async def gen_wrapper(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
                    # TODO: create otel span when configured
                    async for item in attr(*args, **kwargs):
                        yield item

                return gen_wrapper

            if asyncio.iscoroutinefunction(attr):

                @functools.wraps(attr)
                async def coro_wrapper(*args: Any, **kwargs: Any) -> Any:
                    # TODO: create otel span when configured
                    return await attr(*args, **kwargs)

                return coro_wrapper

        return attr


def wrap(provider: Any, *, vendor: str = "", model: str = "", rate_limit: float = 0) -> Any:
    """Apply standard middleware stack to any provider instance.

    Order: inner provider → rate limit → tracing (outermost).
    """
    if rate_limit > 0:
        provider = RateLimited(provider, rate=rate_limit)
    if vendor:
        provider = Traced(provider, vendor=vendor, model=model)
    return provider
