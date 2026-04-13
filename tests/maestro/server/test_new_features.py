"""Tests for new server features: policy, tool options, verbosity, effort, incomplete status."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from maestro.config import Config
from maestro.core.models import (
    CompleteOptions,
    Completion,
    Content,
    Message,
    Usage,
)
from maestro.core.types import Role, Status
from maestro.policy.policy import AccessDeniedError
from maestro.server import create_app

# ── Helpers ───────────────────────────────────────────────────────


class CapturingCompleter:
    """Completer that captures options and returns canned response."""

    def __init__(self, chunks: list[Completion] | None = None) -> None:
        self.captured_options: CompleteOptions | None = None
        self._chunks = chunks or [
            Completion(
                id="test-123",
                model="test-model",
                message=Message(role=Role.ASSISTANT, content=[Content(text="ok")]),
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        ]

    async def complete(
        self, messages: list[Message], options: CompleteOptions | None = None,
    ) -> AsyncIterator[Completion]:
        self.captured_options = options
        for chunk in self._chunks:
            yield chunk


def _make_client(
    completer: CapturingCompleter | None = None,
    policy=None,
) -> tuple[TestClient, CapturingCompleter]:
    completer = completer or CapturingCompleter()
    cfg = Config()
    cfg.register("completer", "test-model", completer)
    if policy is not None:
        cfg.policy = policy
    app = create_app(config=cfg)
    return TestClient(app), completer


# ── Policy integration ────────────────────────────────────────────


class TestPolicyIntegration:
    def test_noop_policy_allows(self) -> None:
        client, _ = _make_client()
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200

    def test_deny_policy_blocks_chat(self) -> None:
        class DenyAll:
            async def verify(self, resource, resource_id, action, *, user="", email=""):
                raise AccessDeniedError("nope")

        client, _ = _make_client(policy=DenyAll())
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"]["message"]

    def test_deny_policy_blocks_responses(self) -> None:
        class DenyAll:
            async def verify(self, resource, resource_id, action, *, user="", email=""):
                raise AccessDeniedError("nope")

        client, _ = _make_client(policy=DenyAll())
        resp = client.post("/v1/responses", json={
            "model": "test-model",
            "input": "Hi",
        })
        assert resp.status_code == 404

    def test_selective_policy(self) -> None:
        class SelectivePolicy:
            async def verify(self, resource, resource_id, action, *, user="", email=""):
                if resource_id == "blocked-model":
                    raise AccessDeniedError("blocked")

        client, _ = _make_client(policy=SelectivePolicy())

        # Allowed model works
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200


# ── Verbosity ─────────────────────────────────────────────────────


class TestVerbosity:
    def test_chat_verbosity_forwarded(self) -> None:
        client, completer = _make_client()
        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "verbosity": "low",
        })
        assert completer.captured_options is not None
        assert completer.captured_options.verbosity is not None
        assert completer.captured_options.verbosity.value == "low"

    def test_responses_verbosity_forwarded(self) -> None:
        client, completer = _make_client()
        client.post("/v1/responses", json={
            "model": "test-model",
            "input": "Hi",
            "text": {"verbosity": "high"},
        })
        assert completer.captured_options is not None
        assert completer.captured_options.verbosity is not None
        assert completer.captured_options.verbosity.value == "high"


# ── Effort levels ─────────────────────────────────────────────────


class TestEffortLevels:
    def test_chat_xhigh_effort(self) -> None:
        client, completer = _make_client()
        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "xhigh",
        })
        assert completer.captured_options is not None
        assert completer.captured_options.effort.value == "max"

    def test_chat_none_effort(self) -> None:
        client, completer = _make_client()
        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "none",
        })
        assert completer.captured_options is not None
        assert completer.captured_options.effort.value == "none"

    def test_responses_xhigh_effort(self) -> None:
        client, completer = _make_client()
        client.post("/v1/responses", json={
            "model": "test-model",
            "input": "Hi",
            "reasoning": {"effort": "xhigh"},
        })
        assert completer.captured_options is not None
        assert completer.captured_options.effort.value == "max"

    def test_responses_minimal_effort(self) -> None:
        client, completer = _make_client()
        client.post("/v1/responses", json={
            "model": "test-model",
            "input": "Hi",
            "reasoning": {"effort": "minimal"},
        })
        assert completer.captured_options is not None
        assert completer.captured_options.effort.value == "minimal"


# ── Parallel tool calls ──────────────────────────────────────────


class TestParallelToolCalls:
    def test_chat_parallel_disabled(self) -> None:
        client, completer = _make_client()
        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "parallel_tool_calls": False,
        })
        assert completer.captured_options is not None
        assert completer.captured_options.tool_options is not None
        assert completer.captured_options.tool_options.disable_parallel_tool_calls is True

    def test_responses_parallel_disabled(self) -> None:
        client, completer = _make_client()
        client.post("/v1/responses", json={
            "model": "test-model",
            "input": "Hi",
            "tools": [{"type": "function", "name": "f", "parameters": {}}],
            "parallel_tool_calls": False,
        })
        assert completer.captured_options is not None
        assert completer.captured_options.tool_options is not None
        assert completer.captured_options.tool_options.disable_parallel_tool_calls is True


# ── Incomplete status ─────────────────────────────────────────────


class TestIncompleteStatus:
    def test_responses_non_streaming_incomplete(self) -> None:
        chunks = [
            Completion(
                id="r1",
                model="test-model",
                status=Status.INCOMPLETE,
                message=Message(role=Role.ASSISTANT, content=[Content(text="partial")]),
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        ]
        client, _ = _make_client(completer=CapturingCompleter(chunks=chunks))
        resp = client.post("/v1/responses", json={
            "model": "test-model",
            "input": "Hi",
        })
        data = resp.json()
        assert data["status"] == "incomplete"

    def test_responses_streaming_incomplete(self) -> None:
        chunks = [
            Completion(
                id="r1",
                model="test-model",
                status=Status.INCOMPLETE,
                message=Message(role=Role.ASSISTANT, content=[Content(text="partial")]),
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        ]
        client, _ = _make_client(completer=CapturingCompleter(chunks=chunks))
        resp = client.post("/v1/responses", json={
            "model": "test-model",
            "input": "Hi",
            "stream": True,
        })

        # Parse SSE events
        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("event: "):
                events.append(line[7:])

        assert "response.incomplete" in events
        assert "response.completed" not in events


# ── Structured output (json_object) ──────────────────────────────


class TestStructuredOutput:
    def test_responses_json_object_format(self) -> None:
        client, completer = _make_client()
        client.post("/v1/responses", json={
            "model": "test-model",
            "input": "Hi",
            "text": {"format": {"type": "json_object"}},
        })
        assert completer.captured_options is not None
        assert completer.captured_options.structured_output is not None
        assert completer.captured_options.structured_output.name == "json_object"

    def test_responses_json_schema_format(self) -> None:
        client, completer = _make_client()
        client.post("/v1/responses", json={
            "model": "test-model",
            "input": "Hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "my_schema",
                    "description": "A schema",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
        })
        opts = completer.captured_options
        assert opts is not None
        assert opts.structured_output is not None
        assert opts.structured_output.name == "my_schema"
        assert opts.structured_output.description == "A schema"
        assert opts.structured_output.strict is True
        assert opts.response_schema == {"type": "object"}


# ── Cache tokens in usage ────────────────────────────────────────


class TestCacheTokensInServer:
    def test_responses_usage_includes_cache_tokens(self) -> None:
        chunks = [
            Completion(
                id="r1",
                model="test-model",
                message=Message(role=Role.ASSISTANT, content=[Content(text="ok")]),
                usage=Usage(
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_input_tokens=80,
                    cache_creation_input_tokens=20,
                ),
            )
        ]
        client, _ = _make_client(completer=CapturingCompleter(chunks=chunks))
        resp = client.post("/v1/responses", json={
            "model": "test-model",
            "input": "Hi",
        })
        data = resp.json()
        assert data["usage"]["input_tokens"] == 100
        assert data["usage"]["output_tokens"] == 50
