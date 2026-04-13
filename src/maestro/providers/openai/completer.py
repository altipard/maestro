from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import APIStatusError, AsyncOpenAI

from maestro.core.errors import ProviderError
from maestro.core.models import (
    CompleteOptions,
    Completion,
    Content,
    Message,
    ToolCall,
    Usage,
)
from maestro.core.types import Role, Status, ToolChoice
from maestro.providers.registry import provider

from ._constants import REASONING_MODELS
from ._throttle import ThrottleTransport


@provider("openai", "completer")
class Completer:
    """OpenAI chat completions provider with streaming support."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        self._url = url or "https://api.openai.com/v1/"
        self._token = token
        self._model = model

        http_client = httpx.AsyncClient(
            transport=ThrottleTransport(),
            timeout=httpx.Timeout(timeout=300),
        )
        self._client = AsyncOpenAI(
            base_url=self._url.rstrip("/") + "/",
            api_key=self._token,
            http_client=http_client,
        )

    async def complete(
        self,
        messages: list[Message],
        options: CompleteOptions | None = None,
    ) -> AsyncIterator[Completion]:
        options = options or CompleteOptions()
        params = self._build_params(messages, options)

        try:
            stream = await self._client.chat.completions.create(**params, stream=True)
        except APIStatusError as exc:
            retry_after = _parse_retry_after(exc.response.headers)
            raise ProviderError(
                exc.status_code, exc.message, retry_after=retry_after, cause=exc
            ) from exc

        try:
            async for chunk in stream:
                delta = Completion(
                    id=chunk.id or "",
                    model=self._model,
                    message=Message(role=Role.ASSISTANT, content=[]),
                    usage=_to_usage(chunk.usage) if chunk.usage else None,
                )

                if chunk.choices:
                    choice = chunk.choices[0]

                    if choice.delta.content is not None:
                        delta.message.content.append(Content(text=choice.delta.content))

                    if choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            call = ToolCall(
                                id=tc.id or "",
                                name=tc.function.name or "" if tc.function else "",
                                arguments=tc.function.arguments or "" if tc.function else "",
                            )
                            delta.message.content.append(Content(tool_call=call))

                    if choice.finish_reason == "length":
                        delta = delta.model_copy(update={"status": Status.INCOMPLETE})

                yield delta
        except APIStatusError as exc:
            retry_after = _parse_retry_after(exc.response.headers)
            raise ProviderError(
                exc.status_code, exc.message, retry_after=retry_after, cause=exc
            ) from exc

    # ── Request building ──────────────────────────────────────────

    def _build_params(self, messages: list[Message], options: CompleteOptions) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self._model}

        # Stream options — skip for Mistral (same as Go source)
        if "api.mistral.ai" not in self._url:
            params["stream_options"] = {"include_usage": True}

        # Messages
        params["messages"] = self._convert_messages(messages)

        # Tools
        tools = _convert_tools(options.tools)
        if tools:
            params["tools"] = tools

        # Tool choice (legacy flat field)
        if options.tool_choice is not None:
            params["tool_choice"] = _convert_tool_choice(options.tool_choice)

        # Tool options (structured, overrides tool_choice)
        if options.tool_options is not None:
            params["tool_choice"] = _convert_tool_options(options.tool_options)
            if options.tool_options.disable_parallel_tool_calls:
                params["parallel_tool_calls"] = False

        # Reasoning effort
        if options.effort is not None:
            params["reasoning_effort"] = _convert_effort(options.effort)

        # Verbosity
        if options.verbosity is not None:
            params["verbosity"] = options.verbosity.value

        # Response format / structured output
        if options.response_schema is not None:
            params["response_format"] = options.response_schema

        # Stop sequences
        if options.stop:
            params["stop"] = options.stop

        # Max tokens — reasoning models use max_completion_tokens
        if options.max_tokens is not None:
            if self._model in REASONING_MODELS:
                params["max_completion_tokens"] = options.max_tokens
            else:
                params["max_tokens"] = options.max_tokens

        # Temperature — skip for reasoning models (same as Go source)
        if options.temperature is not None and self._model not in REASONING_MODELS:
            params["temperature"] = options.temperature

        return params

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for msg in messages:
            match msg.role:
                case Role.SYSTEM:
                    text = "\n\n".join(c.text for c in msg.content if c.text)
                    role = "developer" if self._model in REASONING_MODELS else "system"
                    result.append({"role": role, "content": text})

                case Role.USER:
                    tool_results = [c.tool_result for c in msg.content if c.tool_result]

                    # Tool results become separate tool messages
                    for tr in tool_results:
                        result.append({
                            "role": "tool",
                            "tool_call_id": tr.id,
                            "content": tr.data,
                        })

                    if not tool_results:
                        parts: list[dict[str, Any]] = []
                        for c in msg.content:
                            if c.text and c.text.strip():
                                parts.append({"type": "text", "text": c.text.rstrip()})
                        # Simple string content when only text
                        if len(parts) == 1 and parts[0]["type"] == "text":
                            result.append({"role": "user", "content": parts[0]["text"]})
                        elif parts:
                            result.append({"role": "user", "content": parts})

                case Role.ASSISTANT:
                    msg_dict: dict[str, Any] = {"role": "assistant"}
                    content_parts: list[dict[str, Any]] = []
                    tool_calls: list[dict[str, Any]] = []

                    for c in msg.content:
                        if c.text and c.text.strip():
                            content_parts.append({"type": "text", "text": c.text.rstrip()})
                        if c.tool_call:
                            tool_calls.append({
                                "id": c.tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": c.tool_call.name,
                                    "arguments": c.tool_call.arguments,
                                },
                            })

                    if content_parts:
                        if len(content_parts) == 1:
                            msg_dict["content"] = content_parts[0]["text"]
                        else:
                            msg_dict["content"] = content_parts
                    if tool_calls:
                        msg_dict["tool_calls"] = tool_calls

                    result.append(msg_dict)

        return result


# ── Helpers ───────────────────────────────────────────────────────


def _convert_tools(tools: list | None) -> list[dict[str, Any]]:
    if not tools:
        return []

    result: list[dict[str, Any]] = []
    for t in tools:
        if not t.name:
            continue

        func: dict[str, Any] = {"name": t.name, "parameters": t.parameters}
        if t.description:
            func["description"] = t.description
        if t.strict is not None:
            func["strict"] = t.strict

        result.append({"type": "function", "function": func})
    return result


def _convert_tool_choice(choice: ToolChoice) -> str:
    mapping = {
        ToolChoice.NONE: "none",
        ToolChoice.AUTO: "auto",
        ToolChoice.ANY: "required",
    }
    return mapping.get(choice, "auto")


def _convert_tool_options(opts) -> str | dict:
    """Convert ToolOptions to OpenAI tool_choice format."""
    from maestro.core.models import ToolOptions

    if not isinstance(opts, ToolOptions):
        return "auto"

    if opts.allowed and len(opts.allowed) == 1:
        return {"type": "function", "function": {"name": opts.allowed[0]}}

    mapping = {
        ToolChoice.NONE: "none",
        ToolChoice.AUTO: "auto",
        ToolChoice.ANY: "required",
    }
    return mapping.get(opts.choice, "auto")


def _convert_effort(effort) -> str:
    """Map internal Effort enum to OpenAI reasoning_effort string."""
    from maestro.core.types import Effort

    mapping = {
        Effort.NONE: "low",      # OpenAI doesn't have none/minimal, map down
        Effort.MINIMAL: "low",
        Effort.LOW: "low",
        Effort.MEDIUM: "medium",
        Effort.HIGH: "high",
        Effort.MAX: "high",      # OpenAI max = high
    }
    return mapping.get(effort, effort.value)


def _parse_retry_after(headers: httpx.Headers) -> float:
    """Extract Retry-After from response headers as seconds."""
    value = headers.get("retry-after", "")
    if not value:
        return 0
    try:
        return float(value)
    except ValueError:
        return 0


def _to_usage(usage: Any) -> Usage | None:
    if not usage or not usage.total_tokens:
        return None
    return Usage(
        input_tokens=usage.prompt_tokens or 0,
        output_tokens=usage.completion_tokens or 0,
    )
