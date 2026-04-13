"""Integration tests for the Google Gemini completer.

Requires GOOGLE_API_KEY environment variable.
Tests are skipped if the key is not set.
"""

from __future__ import annotations

import os

import pytest

from maestro.core.models import Accumulator, CompleteOptions, Completion, Message, Tool
from maestro.core.protocols import Completer as CompleterProtocol
from maestro.providers.registry import create

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)

MODEL = "gemini-2.0-flash"


class TestGoogleRegistry:
    def test_completer_registered(self) -> None:
        c = create("google", "completer", token="test", model=MODEL)
        assert c is not None

    def test_completer_protocol(self) -> None:
        c = create("google", "completer", token="test", model=MODEL)
        assert isinstance(c, CompleterProtocol)


class TestGoogleCompleter:
    async def test_simple_completion(self) -> None:
        c = create(
            "google", "completer",
            token=os.environ["GOOGLE_API_KEY"],
            model=MODEL,
        )

        result = await Accumulator.collect(
            c.complete([Message.user("Say 'hello' and nothing else.")])
        )

        assert result.message is not None
        assert "hello" in result.message.text.lower()

    async def test_streaming_chunks(self) -> None:
        c = create(
            "google", "completer",
            token=os.environ["GOOGLE_API_KEY"],
            model=MODEL,
        )

        chunks: list[Completion] = []
        async for chunk in c.complete([Message.user("Count to 3.")]):
            chunks.append(chunk)

        assert len(chunks) > 1

    async def test_system_message(self) -> None:
        c = create(
            "google", "completer",
            token=os.environ["GOOGLE_API_KEY"],
            model=MODEL,
        )

        result = await Accumulator.collect(
            c.complete([
                Message.system("You are a pirate. Always say 'Arr'."),
                Message.user("Greet me."),
            ])
        )

        assert result.message is not None
        assert len(result.message.text) > 0

    async def test_max_tokens(self) -> None:
        c = create(
            "google", "completer",
            token=os.environ["GOOGLE_API_KEY"],
            model=MODEL,
        )

        result = await Accumulator.collect(
            c.complete(
                [Message.user("Write a long essay about the universe.")],
                CompleteOptions(max_tokens=50),
            )
        )

        assert result.message is not None

    async def test_usage_reported(self) -> None:
        c = create(
            "google", "completer",
            token=os.environ["GOOGLE_API_KEY"],
            model=MODEL,
        )

        result = await Accumulator.collect(
            c.complete([Message.user("Hi")])
        )

        assert result.usage is not None
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0

    async def test_tool_call(self) -> None:
        c = create(
            "google", "completer",
            token=os.environ["GOOGLE_API_KEY"],
            model=MODEL,
        )

        tools = [
            Tool(
                name="get_weather",
                description="Get the weather for a city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            )
        ]

        result = await Accumulator.collect(
            c.complete(
                [Message.user("What's the weather in Berlin?")],
                CompleteOptions(tools=tools),
            )
        )

        assert result.message is not None
        assert len(result.message.tool_calls) > 0
        assert result.message.tool_calls[0].name == "get_weather"

    async def test_multi_turn(self) -> None:
        c = create(
            "google", "completer",
            token=os.environ["GOOGLE_API_KEY"],
            model=MODEL,
        )

        result = await Accumulator.collect(
            c.complete([
                Message.user("My name is TestBot."),
                Message.assistant("Hello TestBot!"),
                Message.user("What's my name?"),
            ])
        )

        assert result.message is not None
        assert "testbot" in result.message.text.lower()
