"""Tests for the rate-limit throttle transport."""

import httpx

from maestro.providers.openai._throttle import (
    ThrottleTransport,
    _check_limit,
    _header_duration,
    _header_int,
    _is_low,
)


class TestIsLow:
    def test_remaining_zero(self):
        assert _is_low(0, 100) is True

    def test_remaining_one(self):
        assert _is_low(1, 100) is True

    def test_remaining_below_five_percent(self):
        assert _is_low(4, 100) is True  # 4 < 100//20 = 5

    def test_remaining_at_five_percent(self):
        assert _is_low(5, 100) is False  # 5 is not < 5

    def test_remaining_above_five_percent(self):
        assert _is_low(50, 100) is False

    def test_unknown_limit_remaining_high(self):
        assert _is_low(1000, 0) is False

    def test_unknown_limit_remaining_one(self):
        assert _is_low(1, 0) is True


class TestHeaderInt:
    def test_valid(self):
        assert _header_int("59") == 59

    def test_empty(self):
        assert _header_int("") == -1

    def test_invalid(self):
        assert _header_int("abc") == -1


class TestHeaderDuration:
    def test_seconds(self):
        assert _header_duration("2s") == 2.0

    def test_minutes_and_seconds(self):
        assert _header_duration("6m0s") == 360.0

    def test_milliseconds(self):
        assert _header_duration("500ms") == 0.5

    def test_plain_float(self):
        assert _header_duration("1.5") == 1.5

    def test_empty(self):
        assert _header_duration("") == 0

    def test_hours(self):
        assert _header_duration("1h") == 3600.0

    def test_complex(self):
        assert _header_duration("1m30s") == 90.0


class TestCheckLimit:
    def test_headers_absent(self):
        headers = httpx.Headers({})
        assert _check_limit(headers, "requests") == 0

    def test_remaining_high(self):
        headers = httpx.Headers(
            {
                "x-ratelimit-remaining-requests": "100",
                "x-ratelimit-limit-requests": "1000",
            }
        )
        assert _check_limit(headers, "requests") == 0

    def test_remaining_low_with_reset(self):
        headers = httpx.Headers(
            {
                "x-ratelimit-remaining-requests": "1",
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-reset-requests": "2s",
            }
        )
        assert _check_limit(headers, "requests") == 2.0

    def test_remaining_low_no_reset_defaults_to_1s(self):
        headers = httpx.Headers(
            {
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-limit-requests": "60",
            }
        )
        assert _check_limit(headers, "requests") == 1.0

    def test_negative_remaining_ignored(self):
        headers = httpx.Headers(
            {
                "x-ratelimit-remaining-requests": "-1",
            }
        )
        assert _check_limit(headers, "requests") == 0


class TestThrottleTransportObserve:
    def test_no_headers_no_delay(self):
        transport = ThrottleTransport.__new__(ThrottleTransport)
        transport._wait_until = 0
        transport._observe(httpx.Headers({}))
        assert transport._wait_until == 0

    def test_low_remaining_sets_delay(self):
        transport = ThrottleTransport.__new__(ThrottleTransport)
        transport._wait_until = 0
        transport._observe(
            httpx.Headers(
                {
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-limit-requests": "60",
                    "x-ratelimit-reset-requests": "5s",
                }
            )
        )
        assert transport._wait_until > 0

    def test_delay_capped_at_60s(self):
        import time

        transport = ThrottleTransport.__new__(ThrottleTransport)
        transport._wait_until = 0
        transport._observe(
            httpx.Headers(
                {
                    "x-ratelimit-remaining-tokens": "0",
                    "x-ratelimit-limit-tokens": "100000",
                    "x-ratelimit-reset-tokens": "5m0s",  # 300s > 60s cap
                }
            )
        )
        # wait_until should be at most ~60s from now
        assert transport._wait_until <= time.monotonic() + 61
