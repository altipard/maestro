"""Tests for the FastAPI server layer.

These tests do NOT require API keys — they use a mock completer/embedder
to validate request parsing, response formatting, SSE streaming, and routing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from maestro.config import Config
from maestro.core.models import (
    CompleteOptions,
    Completion,
    Content,
    Embedding,
    EmbedOptions,
    Message,
    ToolCall,
    Usage,
)
from maestro.core.types import Role
from maestro.server import create_app

# ── Mock providers ─────────────────────────────────────────────────


class MockCompleter:
    """Mock completer that returns canned responses."""

    def __init__(self, chunks: list[Completion] | None = None) -> None:
        self._chunks = chunks or [
            Completion(
                id="chatcmpl-123",
                model="mock-model",
                message=Message(role=Role.ASSISTANT, content=[Content(text="Hello!")]),
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        ]

    async def complete(
        self,
        messages: list[Message],
        options: CompleteOptions | None = None,
    ) -> AsyncIterator[Completion]:
        for chunk in self._chunks:
            yield chunk


class MockEmbedder:
    """Mock embedder that returns canned embeddings."""

    async def embed(
        self,
        texts: list[str],
        options: EmbedOptions | None = None,
    ) -> Embedding:
        return Embedding(
            model="mock-embed",
            embeddings=[[0.1, 0.2, 0.3] for _ in texts],
            usage=Usage(input_tokens=len(texts) * 5, output_tokens=0),
        )


def _make_client(
    completer: MockCompleter | None = None,
    embedder: MockEmbedder | None = None,
) -> TestClient:
    """Build a test client with mock providers registered."""
    cfg = Config()

    if completer:
        cfg.register("completer", "mock-model", completer)
    if embedder:
        cfg.register("embedder", "mock-embed", embedder)

    app = create_app(config=cfg)
    return TestClient(app)


# ── Chat completions (non-streaming) ──────────────────────────────


class TestChatCompletions:
    def test_simple_completion(self) -> None:
        client = _make_client(completer=MockCompleter())

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["id"] == "chatcmpl-123"
        assert data["model"] == "mock-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_usage_included(self) -> None:
        client = _make_client(completer=MockCompleter())

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        data = resp.json()
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 5
        assert data["usage"]["total_tokens"] == 15

    def test_tool_call_response(self) -> None:
        chunks = [
            Completion(
                id="chatcmpl-456",
                model="mock-model",
                message=Message(
                    role=Role.ASSISTANT,
                    content=[
                        Content(
                            tool_call=ToolCall(
                                id="call_abc",
                                name="get_weather",
                                arguments='{"city":"Berlin"}',
                            )
                        )
                    ],
                ),
                usage=Usage(input_tokens=20, output_tokens=10),
            )
        ]
        client = _make_client(completer=MockCompleter(chunks=chunks))

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "Weather?"}],
            },
        )

        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city":"Berlin"}'

    def test_model_not_found(self) -> None:
        client = _make_client()  # No providers registered

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert resp.status_code == 400
        assert "not found" in resp.json()["error"]["message"]

    def test_system_message_passed(self) -> None:
        """System and developer messages should be passed through."""
        received: list[list[Message]] = []

        class CapturingCompleter:
            async def complete(self, messages, options=None):
                received.append(messages)
                yield Completion(
                    id="test",
                    message=Message(role=Role.ASSISTANT, content=[Content(text="ok")]),
                )

        cfg = Config()
        cfg.register("completer", "test-model", CapturingCompleter())
        client = TestClient(create_app(config=cfg))

        client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
            },
        )

        assert len(received) == 1
        assert received[0][0].role == Role.SYSTEM
        assert received[0][0].text == "You are helpful."
        assert received[0][1].role == Role.USER

    def test_tool_result_message(self) -> None:
        """Tool results should be converted to internal format."""
        received: list[list[Message]] = []

        class CapturingCompleter:
            async def complete(self, messages, options=None):
                received.append(messages)
                yield Completion(
                    id="test",
                    message=Message(role=Role.ASSISTANT, content=[Content(text="ok")]),
                )

        cfg = Config()
        cfg.register("completer", "test-model", CapturingCompleter())
        client = TestClient(create_app(config=cfg))

        client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 22°C"},
                ],
            },
        )

        assert len(received) == 1
        # Tool result message: role=USER, content has tool_result
        tool_msg = received[0][2]
        assert tool_msg.role == Role.USER
        assert tool_msg.content[0].tool_result is not None
        assert tool_msg.content[0].tool_result.id == "call_1"
        assert tool_msg.content[0].tool_result.data == "Sunny, 22°C"


# ── Chat completions (streaming) ──────────────────────────────────


class TestChatCompletionsStream:
    def test_streaming_sse_format(self) -> None:
        client = _make_client(completer=MockCompleter())

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = resp.text.strip().split("\n")
        # Should have data lines and a [DONE] at the end
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) >= 2  # At least one chunk + [DONE]
        assert data_lines[-1] == "data: [DONE]"

    def test_streaming_chunk_structure(self) -> None:
        client = _make_client(completer=MockCompleter())

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        lines = resp.text.strip().split("\n")
        data_lines = [
            line for line in lines if line.startswith("data: ") and line != "data: [DONE]"
        ]

        # First data chunk should have the content
        first = json.loads(data_lines[0][6:])
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"]["role"] == "assistant"

    def test_streaming_multi_chunk(self) -> None:
        """Multiple chunks should produce multiple SSE events."""
        chunks = [
            Completion(
                id="c1",
                model="m",
                message=Message(role=Role.ASSISTANT, content=[Content(text="Hel")]),
            ),
            Completion(
                id="c1",
                model="m",
                message=Message(role=Role.ASSISTANT, content=[Content(text="lo!")]),
            ),
            Completion(id="c1", model="m", usage=Usage(input_tokens=5, output_tokens=3)),
        ]
        client = _make_client(completer=MockCompleter(chunks=chunks))

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

        lines = resp.text.strip().split("\n")
        data_lines = [
            line for line in lines if line.startswith("data: ") and line != "data: [DONE]"
        ]

        # 2 content chunks + 1 finish + 1 usage = 4
        assert len(data_lines) >= 3


# ── Embeddings ─────────────────────────────────────────────────────


class TestEmbeddings:
    def test_simple_embedding(self) -> None:
        client = _make_client(embedder=MockEmbedder())

        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "mock-embed",
                "input": "Hello world",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["model"] == "mock-embed"
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    def test_batch_embeddings(self) -> None:
        client = _make_client(embedder=MockEmbedder())

        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "mock-embed",
                "input": ["Hello", "World"],
            },
        )

        data = resp.json()
        assert len(data["data"]) == 2
        assert data["data"][0]["index"] == 0
        assert data["data"][1]["index"] == 1

    def test_usage_reported(self) -> None:
        client = _make_client(embedder=MockEmbedder())

        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "mock-embed",
                "input": "test",
            },
        )

        data = resp.json()
        assert data["usage"]["prompt_tokens"] == 5

    def test_model_not_found(self) -> None:
        client = _make_client()

        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "nonexistent",
                "input": "test",
            },
        )

        assert resp.status_code == 400

    def test_base64_encoding(self) -> None:
        client = _make_client(embedder=MockEmbedder())

        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "mock-embed",
                "input": "test",
                "encoding_format": "base64",
            },
        )

        data = resp.json()
        assert isinstance(data["data"][0]["embedding"], str)


# ── Models ─────────────────────────────────────────────────────────


class TestModels:
    def test_list_models(self) -> None:
        client = _make_client(
            completer=MockCompleter(),
            embedder=MockEmbedder(),
        )

        resp = client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "mock-model" in model_ids
        assert "mock-embed" in model_ids

    def test_list_empty(self) -> None:
        client = _make_client()

        resp = client.get("/v1/models")

        data = resp.json()
        assert data["data"] == []

    def test_get_model(self) -> None:
        client = _make_client(completer=MockCompleter())

        resp = client.get("/v1/models/mock-model")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "mock-model"
        assert data["object"] == "model"

    def test_get_model_not_found(self) -> None:
        client = _make_client()

        resp = client.get("/v1/models/nonexistent")

        assert resp.status_code == 404
