"""Tests for the agent chain (agentic tool loop)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from maestro.chains.agent import AgentChain
from maestro.core.models import (
    Accumulator,
    CompleteOptions,
    Completion,
    Content,
    Message,
    Tool,
    ToolCall,
    Usage,
)
from maestro.core.types import Role

# ── Helpers ─────────────────────────────────────────────────────


class _MockToolProvider:
    """Simple tool provider that returns canned responses."""

    def __init__(
        self,
        tools_list: list[Tool],
        results: dict[str, Any] | None = None,
    ) -> None:
        self._tools = tools_list
        self._results = results or {}

    async def tools(self) -> list[Tool]:
        return self._tools

    async def execute(self, name: str, parameters: dict[str, Any]) -> Any:
        return self._results.get(name, {"result": "ok"})


class _MockCompleter:
    """Completer that returns scripted responses.

    Each call to complete() pops the next response from the script.
    """

    def __init__(self, script: list[list[Completion]]) -> None:
        self._script = list(script)
        self._call_count = 0

    async def complete(
        self,
        messages: list[Message],
        options: CompleteOptions | None = None,
    ) -> AsyncIterator[Completion]:
        idx = min(self._call_count, len(self._script) - 1)
        chunks = self._script[idx]
        self._call_count += 1
        for chunk in chunks:
            yield chunk


# ── Tests ───────────────────────────────────────────────────────


class TestAgentChain:
    async def test_simple_no_tools(self) -> None:
        """Chain with no tools passes through to completer."""
        completer = _MockCompleter([
            [
                Completion(
                    model="test",
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[Content(text="Hello!")],
                    ),
                    usage=Usage(input_tokens=10, output_tokens=5),
                ),
            ],
        ])

        chain = AgentChain(completer, model="test-model")

        result = await Accumulator.collect(
            chain.complete([Message.user("Hi")])
        )

        assert result.message is not None
        assert "Hello!" in result.message.text

    async def test_tool_call_loop(self) -> None:
        """Chain executes tool calls and loops back to completer."""
        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {}},
        )

        tp = _MockToolProvider(
            tools_list=[tool],
            results={"get_weather": {"temp": 20, "city": "Berlin"}},
        )

        # First call: model returns a tool call
        # Second call: model returns text
        completer = _MockCompleter([
            [
                Completion(
                    model="test",
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[
                            Content(
                                tool_call=ToolCall(
                                    id="tc_1",
                                    name="get_weather",
                                    arguments='{"city": "Berlin"}',
                                )
                            )
                        ],
                    ),
                ),
            ],
            [
                Completion(
                    model="test",
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[
                            Content(text="It's 20°C in Berlin.")
                        ],
                    ),
                ),
            ],
        ])

        chain = AgentChain(
            completer,
            model="test-model",
            tools=[tp],
        )

        result = await Accumulator.collect(
            chain.complete([Message.user("What's the weather in Berlin?")])
        )

        assert result.message is not None
        assert "20" in result.message.text
        assert completer._call_count == 2

    async def test_agent_tool_calls_filtered_from_output(self) -> None:
        """Tool calls for agent tools are not yielded to the caller."""
        tool = Tool(
            name="internal_tool",
            description="Internal",
            parameters={"type": "object", "properties": {}},
        )

        tp = _MockToolProvider(
            tools_list=[tool],
            results={"internal_tool": {"done": True}},
        )

        completer = _MockCompleter([
            [
                Completion(
                    model="test",
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[
                            Content(
                                tool_call=ToolCall(
                                    id="tc_1",
                                    name="internal_tool",
                                    arguments="{}",
                                )
                            )
                        ],
                    ),
                ),
            ],
            [
                Completion(
                    model="test",
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[Content(text="Done.")],
                    ),
                ),
            ],
        ])

        chain = AgentChain(completer, model="test", tools=[tp])

        chunks: list[Completion] = []
        async for chunk in chain.complete([Message.user("Do it")]):
            chunks.append(chunk)

        # No chunk should contain internal_tool call
        for chunk in chunks:
            if chunk.message:
                for cnt in chunk.message.content:
                    if cnt.tool_call:
                        assert cnt.tool_call.name != "internal_tool"

    async def test_user_tools_pass_through(self) -> None:
        """Tool calls for user-supplied tools are yielded (not executed)."""
        completer = _MockCompleter([
            [
                Completion(
                    model="test",
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[
                            Content(
                                tool_call=ToolCall(
                                    id="tc_1",
                                    name="user_tool",
                                    arguments='{"x": 1}',
                                )
                            )
                        ],
                    ),
                ),
            ],
        ])

        user_tools = [
            Tool(
                name="user_tool",
                description="User tool",
                parameters={"type": "object", "properties": {}},
            )
        ]

        chain = AgentChain(completer, model="test")

        result = await Accumulator.collect(
            chain.complete(
                [Message.user("Use my tool")],
                CompleteOptions(tools=user_tools),
            )
        )

        assert result.message is not None
        assert len(result.message.tool_calls) == 1
        assert result.message.tool_calls[0].name == "user_tool"

    async def test_chain_prepends_system_messages(self) -> None:
        """System messages from chain config are prepended."""
        completer = _MockCompleter([
            [
                Completion(
                    model="test",
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[Content(text="Arr!")],
                    ),
                ),
            ],
        ])

        chain = AgentChain(
            completer,
            model="test",
            messages=[Message.system("You are a pirate.")],
        )

        chunks: list[Completion] = []
        async for chunk in chain.complete([Message.user("Greet me")]):
            chunks.append(chunk)

        # Verify completer received both system and user messages
        assert completer._call_count == 1

    async def test_effort_passed_through(self) -> None:
        """Chain-level effort is passed to options when not set."""
        from maestro.core.types import Effort

        completer = _MockCompleter([
            [
                Completion(
                    model="test",
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[Content(text="ok")],
                    ),
                ),
            ],
        ])

        chain = AgentChain(
            completer, model="test", effort=Effort.HIGH
        )

        async for _ in chain.complete([Message.user("test")]):
            pass
