"""Anthropic chat completions provider with streaming support.

- anthropic Python SDK has async streaming via `client.messages.stream()`
- Thinking blocks are natively supported via `thinking` parameter
- Cache control via `cache_control` on message blocks
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from maestro.core.errors import ProviderError
from maestro.core.models import (
    CompleteOptions,
    Completion,
    Content,
    Message,
    Reasoning,
    ToolCall,
    Usage,
)
from maestro.core.types import Effort, Role, ToolChoice
from maestro.providers.registry import provider

from ._constants import THINKING_MODELS


@provider("anthropic", "completer")
class Completer:
    """Anthropic chat completions provider with streaming support."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        self._url = (url or "https://api.anthropic.com/").rstrip("/") + "/"
        self._token = token
        self._model = model
        self._client = anthropic.AsyncAnthropic(
            base_url=self._url,
            api_key=self._token,
        )

    async def complete(
        self,
        messages: list[Message],
        options: CompleteOptions | None = None,
    ) -> AsyncIterator[Completion]:
        options = options or CompleteOptions()
        params = self._build_params(messages, options)

        try:
            async with self._client.messages.stream(**params) as stream:
                async for event in stream:
                    delta = self._handle_event(event, options)
                    if delta is not None:
                        yield delta

                # Final message with usage
                msg = await stream.get_final_message()
                yield Completion(
                    id=msg.id,
                    model=self._model,
                    message=Message(role=Role.ASSISTANT, content=[]),
                    usage=_to_usage(msg.usage),
                )
        except anthropic.APIStatusError as exc:
            retry_after = 0.0
            ra = exc.response.headers.get("retry-after", "")
            if ra:
                try:
                    retry_after = float(ra)
                except ValueError:
                    pass
            raise ProviderError(
                exc.status_code, exc.message, retry_after=retry_after, cause=exc
            ) from exc

    def _handle_event(
        self,
        event: Any,
        options: CompleteOptions,
    ) -> Completion | None:
        """Convert an Anthropic stream event to a Completion chunk."""
        match event.type:
            case "content_block_start":
                block = event.content_block
                match block.type:
                    case "thinking":
                        return Completion(
                            model=self._model,
                            message=Message(
                                role=Role.ASSISTANT,
                                content=[
                                    Content(
                                        reasoning=Reasoning(
                                            text=block.thinking or "",
                                        )
                                    )
                                ],
                            ),
                        )
                    case "text":
                        return Completion(
                            model=self._model,
                            message=Message(
                                role=Role.ASSISTANT,
                                content=[Content(text=block.text or "")],
                            ),
                        )
                    case "tool_use":
                        if options.response_schema is not None:
                            # Schema mode: tool use is treated as text
                            return Completion(
                                model=self._model,
                                message=Message(
                                    role=Role.ASSISTANT,
                                    content=[Content(text="")],
                                ),
                            )
                        return Completion(
                            model=self._model,
                            message=Message(
                                role=Role.ASSISTANT,
                                content=[
                                    Content(
                                        tool_call=ToolCall(
                                            id=block.id,
                                            name=block.name,
                                        )
                                    )
                                ],
                            ),
                        )
            case "content_block_delta":
                delta = event.delta
                match delta.type:
                    case "thinking_delta":
                        return Completion(
                            model=self._model,
                            message=Message(
                                role=Role.ASSISTANT,
                                content=[Content(reasoning=Reasoning(text=delta.thinking))],
                            ),
                        )
                    case "signature_delta":
                        return Completion(
                            model=self._model,
                            message=Message(
                                role=Role.ASSISTANT,
                                content=[
                                    Content(
                                        reasoning=Reasoning(
                                            signature=delta.signature,
                                        )
                                    )
                                ],
                            ),
                        )
                    case "text_delta":
                        return Completion(
                            model=self._model,
                            message=Message(
                                role=Role.ASSISTANT,
                                content=[Content(text=delta.text)],
                            ),
                        )
                    case "input_json_delta":
                        if options.response_schema is not None:
                            return Completion(
                                model=self._model,
                                message=Message(
                                    role=Role.ASSISTANT,
                                    content=[Content(text=delta.partial_json)],
                                ),
                            )
                        return Completion(
                            model=self._model,
                            message=Message(
                                role=Role.ASSISTANT,
                                content=[
                                    Content(
                                        tool_call=ToolCall(
                                            arguments=delta.partial_json,
                                        )
                                    )
                                ],
                            ),
                        )
            case "content_block_stop":
                # Check for empty tool use (same hack as Go)
                pass

        return None

    # ── Request building ──────────────────────────────────────────

    def _build_params(
        self,
        messages: list[Message],
        options: CompleteOptions,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 64000,
        }

        is_thinking = self._model in THINKING_MODELS

        if is_thinking:
            params["max_tokens"] = 128000
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 32000,
            }

            match options.effort:
                case Effort.NONE:
                    params["thinking"]["budget_tokens"] = 1024
                case Effort.MINIMAL:
                    params["thinking"]["budget_tokens"] = 2000
                case Effort.LOW:
                    params["thinking"]["budget_tokens"] = 4000
                case Effort.MEDIUM:
                    params["thinking"]["budget_tokens"] = 16000
                case Effort.HIGH:
                    params["thinking"]["budget_tokens"] = 32000
                case Effort.MAX:
                    params["thinking"]["budget_tokens"] = 64000

        # System messages
        system_parts = self._extract_system(messages)
        if system_parts:
            # Add cache control to last system block
            system_parts[-1]["cache_control"] = {"type": "ephemeral"}
            params["system"] = system_parts

        # Messages (non-system)
        params["messages"] = self._convert_messages(messages)

        # Tools
        tools = self._convert_tools(options.tools)
        if tools:
            # Add cache control to last tool
            tools[-1]["cache_control"] = {"type": "ephemeral"}
            params["tools"] = tools

        # Tool choice (legacy flat field)
        if options.tool_choice is not None:
            params["tool_choice"] = _convert_tool_choice(options.tool_choice)

        # Tool options (structured, overrides tool_choice)
        if options.tool_options is not None:
            tc = _convert_tool_choice(options.tool_options.choice)
            if options.tool_options.disable_parallel_tool_calls:
                if tc.get("type") in ("auto", "any"):
                    tc["disable_parallel_tool_use"] = True
            if options.tool_options.allowed and len(options.tool_options.allowed) == 1:
                tc = {
                    "type": "tool",
                    "name": options.tool_options.allowed[0],
                    "disable_parallel_tool_use": options.tool_options.disable_parallel_tool_calls,
                }
            params["tool_choice"] = tc

        # Stop sequences
        if options.stop:
            params["stop_sequences"] = options.stop

        # Max tokens override
        if options.max_tokens is not None:
            params["max_tokens"] = options.max_tokens

        # Temperature
        if options.temperature is not None:
            params["temperature"] = options.temperature

        # Response schema (structured output)
        if options.response_schema is not None:
            # Anthropic doesn't have native JSON schema mode yet;
            # use tool_use with a schema tool as the Go version does
            pass

        return params

    def _extract_system(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Extract system messages as Anthropic system blocks."""
        parts: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                text = msg.text
                if text:
                    parts.append({"type": "text", "text": text})
        return parts

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Convert internal Messages to Anthropic message format."""
        result: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue  # Handled separately

            match msg.role:
                case Role.USER:
                    blocks: list[dict[str, Any]] = []

                    for c in msg.content:
                        if c.text and c.text.strip():
                            blocks.append(
                                {
                                    "type": "text",
                                    "text": c.text.rstrip(),
                                }
                            )

                        if c.tool_result:
                            blocks.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": c.tool_result.id,
                                    "content": [{"type": "text", "text": c.tool_result.data}],
                                }
                            )

                    if blocks:
                        # Cache control on last block of last user msg
                        result.append(
                            {
                                "role": "user",
                                "content": blocks,
                            }
                        )

                case Role.ASSISTANT:
                    blocks = []

                    for c in msg.content:
                        if c.reasoning:
                            blocks.append(
                                {
                                    "type": "thinking",
                                    "thinking": c.reasoning.text,
                                    "signature": c.reasoning.signature,
                                }
                            )

                        if c.text and c.text.strip():
                            blocks.append(
                                {
                                    "type": "text",
                                    "text": c.text.rstrip(),
                                }
                            )

                        if c.tool_call:
                            input_data: dict[str, Any] = {}
                            if c.tool_call.arguments:
                                try:
                                    input_data = json.loads(c.tool_call.arguments)
                                except (json.JSONDecodeError, TypeError):
                                    pass

                            blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": c.tool_call.id,
                                    "name": c.tool_call.name,
                                    "input": input_data,
                                }
                            )

                    if blocks:
                        result.append(
                            {
                                "role": "assistant",
                                "content": blocks,
                            }
                        )

        # Add cache control to last user message's last block
        for i in range(len(result) - 1, -1, -1):
            if result[i]["role"] == "user":
                content = result[i]["content"]
                if content:
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

        return result

    def _convert_tools(
        self,
        tools: list | None,
    ) -> list[dict[str, Any]]:
        """Convert internal Tool list to Anthropic tool format."""
        if not tools:
            return []

        result: list[dict[str, Any]] = []
        for t in tools:
            if not t.name:
                continue

            tool: dict[str, Any] = {
                "name": t.name,
                "input_schema": t.parameters or {"type": "object"},
            }

            if t.description:
                tool["description"] = t.description

            result.append(tool)

        return result


# ── Helpers ───────────────────────────────────────────────────────


def _convert_tool_choice(choice: ToolChoice) -> dict[str, Any]:
    match choice:
        case ToolChoice.NONE:
            return {"type": "none"}
        case ToolChoice.AUTO:
            return {"type": "auto"}
        case ToolChoice.ANY:
            return {"type": "any"}
    return {"type": "auto"}


def _to_usage(usage: Any) -> Usage | None:
    if not usage:
        return None

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    if not input_tokens and not output_tokens:
        return None

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
