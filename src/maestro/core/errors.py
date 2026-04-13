"""Provider error types.

ProviderError wraps upstream API errors with HTTP status codes and
rate-limit info, enabling proper error propagation through the server layer.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Upstream API error with HTTP status and optional retry-after."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        retry_after: float = 0,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        if cause is not None:
            self.__cause__ = cause


def status_code_from_error(err: BaseException, fallback: int = 500) -> int:
    """Extract HTTP status from a ProviderError, mapping 5xx to 502 (Bad Gateway).

    Returns fallback if err is not a ProviderError.
    """
    if isinstance(err, ProviderError) and err.status_code > 0:
        if err.status_code >= 500:
            return 502
        return err.status_code
    return fallback


def retry_after_from_error(err: BaseException) -> float:
    """Extract retry-after seconds from a ProviderError, or 0."""
    if isinstance(err, ProviderError):
        return err.retry_after
    return 0


def retry_after_header_value(seconds: float) -> str:
    """Format retry-after as an HTTP header value (integer seconds)."""
    if seconds <= 0:
        return ""
    secs = max(1, int(seconds))
    return str(secs)
