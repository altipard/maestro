"""Tests for ProviderError and helper functions."""

from maestro.core.errors import (
    ProviderError,
    retry_after_from_error,
    retry_after_header_value,
    status_code_from_error,
)


class TestProviderError:
    def test_basic_construction(self):
        err = ProviderError(429, "rate limited")
        assert err.status_code == 429
        assert str(err) == "rate limited"
        assert err.retry_after == 0

    def test_with_retry_after(self):
        err = ProviderError(429, "slow down", retry_after=30.0)
        assert err.retry_after == 30.0

    def test_with_cause(self):
        cause = ConnectionError("upstream down")
        err = ProviderError(502, "bad gateway", cause=cause)
        assert err.__cause__ is cause

    def test_is_exception(self):
        err = ProviderError(500, "internal")
        assert isinstance(err, Exception)


class TestStatusCodeFromError:
    def test_provider_error_4xx_passthrough(self):
        assert status_code_from_error(ProviderError(401, "unauthorized")) == 401
        assert status_code_from_error(ProviderError(403, "forbidden")) == 403
        assert status_code_from_error(ProviderError(404, "not found")) == 404
        assert status_code_from_error(ProviderError(429, "rate limited")) == 429

    def test_provider_error_5xx_maps_to_502(self):
        assert status_code_from_error(ProviderError(500, "internal")) == 502
        assert status_code_from_error(ProviderError(502, "bad gateway")) == 502
        assert status_code_from_error(ProviderError(503, "unavailable")) == 502

    def test_non_provider_error_returns_fallback(self):
        assert status_code_from_error(ValueError("oops")) == 500
        assert status_code_from_error(ValueError("oops"), fallback=503) == 503

    def test_zero_status_code_returns_fallback(self):
        assert status_code_from_error(ProviderError(0, "unknown")) == 500


class TestRetryAfterFromError:
    def test_provider_error_with_retry(self):
        assert retry_after_from_error(ProviderError(429, "wait", retry_after=60)) == 60

    def test_provider_error_without_retry(self):
        assert retry_after_from_error(ProviderError(500, "err")) == 0

    def test_non_provider_error(self):
        assert retry_after_from_error(RuntimeError("nope")) == 0


class TestRetryAfterHeaderValue:
    def test_positive_value(self):
        assert retry_after_header_value(30.0) == "30"

    def test_fractional_rounds_up_to_minimum_1(self):
        assert retry_after_header_value(0.5) == "1"

    def test_zero_returns_empty(self):
        assert retry_after_header_value(0) == ""

    def test_negative_returns_empty(self):
        assert retry_after_header_value(-5) == ""
