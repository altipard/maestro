"""Integration tests for the OpenAI Completer.

These tests require a running OpenAI-compatible API.
Set OPENAI_API_KEY environment variable to run.

Following the Go project's testing philosophy:
- Integration tests against real APIs
- Table-driven, multi-scenario
- Fuzzy assertions (LLM outputs vary)
"""

from __future__ import annotations

import os

import pytest

from maestro.core.models import Accumulator, CompleteOptions, Message, Tool
from maestro.core.types import Role
from maestro.providers.openai import Completer
from maestro.providers.registry import create

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_TEST_MODEL", "gpt-4o-mini")

skip_no_key = pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")


@skip_no_key
class TestCompleterRegistry:
    def test_create_via_registry(self) -> None:
        c = create("openai", "completer", token=OPENAI_API_KEY, model=MODEL)
        assert isinstance(c, Completer)

    def test_protocol_compliance(self) -> None:
        from maestro.core.protocols import Completer as CompleterProto

        c = create("openai", "completer", token=OPENAI_API_KEY, model=MODEL)
        assert isinstance(c, CompleterProto)


@skip_no_key
class TestCompleterStreaming:
    async def test_simple_completion(self) -> None:
        """Basic streaming: send a message, get text back."""
        completer = Completer(token=OPENAI_API_KEY, model=MODEL)
        messages = [Message.user("Say 'hello' and nothing else.")]

        result = await Accumulator.collect(completer.complete(messages))

        assert result.id
        assert result.model == MODEL
        assert result.message is not None
        assert result.message.role == Role.ASSISTANT
        assert "hello" in result.message.text.lower()

    async def test_streaming_yields_chunks(self) -> None:
        """Verify we get multiple chunks, not just one."""
        completer = Completer(token=OPENAI_API_KEY, model=MODEL)
        messages = [Message.user("Count from 1 to 5.")]

        chunk_count = 0
        async for _chunk in completer.complete(messages):
            chunk_count += 1

        assert chunk_count > 1

    async def test_system_message(self) -> None:
        """System messages influence the response."""
        completer = Completer(token=OPENAI_API_KEY, model=MODEL)
        messages = [
            Message.system("You are a pirate. Always say 'arr'."),
            Message.user("Greet me."),
        ]

        result = await Accumulator.collect(completer.complete(messages))
        assert result.message is not None
        assert "arr" in result.message.text.lower()

    async def test_max_tokens(self) -> None:
        """max_tokens limits output length."""
        completer = Completer(token=OPENAI_API_KEY, model=MODEL)
        messages = [Message.user("Write a very long essay about the universe.")]
        options = CompleteOptions(max_tokens=10)

        result = await Accumulator.collect(completer.complete(messages, options))
        assert result.message is not None
        # With max_tokens=10, response should be short
        assert len(result.message.text.split()) <= 30

    async def test_temperature_zero(self) -> None:
        """Temperature 0 should give deterministic-ish results."""
        completer = Completer(token=OPENAI_API_KEY, model=MODEL)
        messages = [Message.user("What is 2+2? Answer with just the number.")]
        options = CompleteOptions(temperature=0.0)

        result = await Accumulator.collect(completer.complete(messages, options))
        assert result.message is not None
        assert "4" in result.message.text

    async def test_usage_reported(self) -> None:
        """Usage tokens should be present on the final result."""
        completer = Completer(token=OPENAI_API_KEY, model=MODEL)
        messages = [Message.user("Say 'test'.")]

        result = await Accumulator.collect(completer.complete(messages))
        assert result.usage is not None
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0


@skip_no_key
class TestCompleterToolCalls:
    async def test_tool_call_response(self) -> None:
        """Model should invoke a tool when given one."""
        completer = Completer(token=OPENAI_API_KEY, model=MODEL)
        messages = [Message.user("What's the weather in Berlin?")]
        options = CompleteOptions(
            tools=[
                Tool(
                    name="get_weather",
                    description="Get current weather for a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                )
            ],
        )

        result = await Accumulator.collect(completer.complete(messages, options))
        assert result.message is not None

        tool_calls = result.message.tool_calls
        assert len(tool_calls) >= 1
        assert tool_calls[0].name == "get_weather"
        assert "berlin" in tool_calls[0].arguments.lower()

    async def test_multi_turn_with_tool_result(self) -> None:
        """Multi-turn: tool call → tool result → final response."""
        completer = Completer(token=OPENAI_API_KEY, model=MODEL)

        weather_tool = Tool(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        )

        # Turn 1: user asks, model should call tool
        messages = [Message.user("What's the weather in Berlin?")]
        options = CompleteOptions(tools=[weather_tool])

        result = await Accumulator.collect(completer.complete(messages, options))
        assert result.message is not None
        assert result.message.tool_calls

        tool_call = result.message.tool_calls[0]

        # Turn 2: provide tool result, model should respond with text
        messages.append(result.message)
        messages.append(Message.tool(tool_call.id, '{"temperature": 18, "condition": "sunny"}'))

        result2 = await Accumulator.collect(completer.complete(messages, options))
        assert result2.message is not None
        # Model should mention the weather data
        text = result2.message.text.lower()
        assert "18" in text or "sunny" in text or "berlin" in text
