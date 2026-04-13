"""Agent chain — agentic tool loop with streaming.

The agent chain wraps a completer and a set of tool providers.
It streams completion chunks, intercepts tool calls for registered
tools, executes them, feeds results back, and loops until the
model stops calling tools.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from maestro.core.models import (
    Accumulator,
    CompleteOptions,
    Completion,
    Content,
    Message,
    Tool,
    ToolResult,
)
from maestro.core.protocols import Completer
from maestro.core.types import Effort, Role, ToolChoice
from maestro.tools import ToolProvider


class AgentChain:
    """Agentic tool loop — completer + tool providers.

    Streams completion deltas, automatically executes tool calls
    for registered tool providers, and re-invokes the completer
    until no more tool calls remain.
    """

    def __init__(
        self,
        completer: Completer,
        *,
        model: str = "",
        tools: list[ToolProvider] | None = None,
        messages: list[Message] | None = None,
        effort: Effort | None = None,
        temperature: float | None = None,
    ) -> None:
        self._completer = completer
        self._model = model
        self._tool_providers = tools or []
        self._messages = messages or []
        self._effort = effort
        self._temperature = temperature

    async def complete(
        self,
        messages: list[Message],
        options: CompleteOptions | None = None,
    ) -> AsyncIterator[Completion]:
        options = options or CompleteOptions()

        if self._effort and not options.effort:
            options = options.model_copy(update={"effort": self._effort})
        if self._temperature is not None and options.temperature is None:
            options = options.model_copy(update={"temperature": self._temperature})

        # Prepend system/context messages
        input_msgs = list(self._messages) + list(messages)

        # Resolve tool providers → tool map
        agent_tools: dict[str, ToolProvider] = {}
        input_tools: dict[str, Tool] = {}

        for tp in self._tool_providers:
            for t in await tp.tools():
                agent_tools[t.name] = tp
                input_tools[t.name] = t

        # Also include any tools the caller passed via options
        if options.tools:
            for t in options.tools:
                input_tools[t.name] = t

        input_options = options.model_copy(update={"tools": list(input_tools.values())})

        # Merge tool_choice with agent tools
        input_options = _merge_tool_options(input_options, list(agent_tools))

        tool_names_by_id: dict[str, str] = {}

        while True:
            acc = Accumulator()

            async for chunk in self._completer.complete(input_msgs, input_options):
                acc.add(chunk)

                delta = Completion(
                    model=self._model or chunk.model,
                    usage=chunk.usage,
                )

                # Filter out agent-handled tool calls from streamed output
                if chunk.message:
                    filtered_content: list[Content] = []
                    for cnt in chunk.message.content:
                        if cnt.tool_call:
                            tc_id = cnt.tool_call.id
                            tc_name = cnt.tool_call.name

                            if tc_id and tc_name:
                                tool_names_by_id[tc_id] = tc_name
                            elif tc_id and not tc_name:
                                tc_name = tool_names_by_id.get(tc_id, "")

                            if tc_name in agent_tools:
                                continue

                        filtered_content.append(cnt)

                    delta.message = Message(
                        role=chunk.message.role,
                        content=filtered_content,
                    )

                yield delta

            result = acc.result

            if result.message is None:
                return

            # Append assistant message to history
            input_msgs.append(result.message)

            # Execute tool calls
            loop = False

            for cnt in result.message.content:
                if cnt.tool_call is None:
                    continue

                tp = agent_tools.get(cnt.tool_call.name)
                if tp is None:
                    continue

                params: dict[str, Any] = {}
                if cnt.tool_call.arguments:
                    try:
                        params = json.loads(cnt.tool_call.arguments)
                    except (json.JSONDecodeError, TypeError):
                        pass

                result_data = await tp.execute(cnt.tool_call.name, params)
                data = json.dumps(result_data)

                input_msgs.append(
                    Message(
                        role=Role.USER,
                        content=[
                            Content(
                                tool_result=ToolResult(
                                    id=cnt.tool_call.id,
                                    data=data,
                                )
                            )
                        ],
                    )
                )

                loop = True

            if not loop:
                return


def _merge_tool_options(
    options: CompleteOptions,
    agent_tool_names: list[str],
) -> CompleteOptions:
    """Merge user tool_choice with agent tools.

    Agent tools must always be callable. If the user sets NONE,
    we switch to AUTO but restrict to agent tools only.
    """
    if not agent_tool_names:
        return options

    if options.tool_choice is None:
        return options

    if options.tool_choice == ToolChoice.NONE:
        return options.model_copy(update={"tool_choice": ToolChoice.AUTO})

    return options
