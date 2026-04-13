"""Google Gemini chat completions provider with streaming support.

Uses google-genai SDK for async streaming. System messages are
passed as system_instruction, not inline content.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

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
from maestro.core.types import Role, ToolChoice
from maestro.providers.registry import provider


@provider("google", "completer")
class Completer:
    """Google Gemini streaming completer."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        self._token = token
        self._model = model
        self._client = genai.Client(api_key=self._token)

    async def complete(
        self,
        messages: list[Message],
        options: CompleteOptions | None = None,
    ) -> AsyncIterator[Completion]:
        options = options or CompleteOptions()

        config = self._build_config(messages, options)
        contents = self._convert_messages(messages)

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            )
        except genai_errors.APIError as exc:
            raise ProviderError(
                getattr(exc, "code", 500) or 500,
                str(exc),
                cause=exc,
            ) from exc

        try:
            async for resp in stream:
                delta = Completion(
                    id=getattr(resp, "response_id", "") or "",
                    model=self._model,
                    message=Message(role=Role.ASSISTANT, content=[]),
                    usage=_to_usage(getattr(resp, "usage_metadata", None)),
                )

                if resp.candidates:
                    candidate = resp.candidates[0]
                    if candidate.content and candidate.content.parts:
                        delta.message = Message(
                            role=Role.ASSISTANT,
                            content=_to_content(candidate.content.parts),
                        )

                yield delta
        except genai_errors.APIError as exc:
            raise ProviderError(
                getattr(exc, "code", 500) or 500,
                str(exc),
                cause=exc,
            ) from exc

    # ── Request building ──────────────────────────────────────────

    def _build_config(
        self,
        messages: list[Message],
        options: CompleteOptions,
    ) -> types.GenerateContentConfig:
        config: dict[str, Any] = {}

        # System instruction
        system_parts = self._extract_system(messages)
        if system_parts:
            config["system_instruction"] = system_parts

        # Tools
        tools = self._convert_tools(options.tools)
        if tools:
            config["tools"] = tools

            # Tool choice
            if options.tool_choice is not None:
                fcc: dict[str, Any] = {}
                match options.tool_choice:
                    case ToolChoice.NONE:
                        fcc["mode"] = "NONE"
                    case ToolChoice.AUTO:
                        fcc["mode"] = "AUTO"
                    case ToolChoice.ANY:
                        fcc["mode"] = "ANY"
                config["tool_config"] = {"function_calling_config": fcc}

        # Stop sequences
        if options.stop:
            config["stop_sequences"] = options.stop

        # Max tokens
        if options.max_tokens is not None:
            config["max_output_tokens"] = options.max_tokens

        # Temperature
        if options.temperature is not None:
            config["temperature"] = options.temperature

        # Response schema
        if options.response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = options.response_schema

        return types.GenerateContentConfig(**config)

    def _extract_system(self, messages: list[Message]) -> str | None:
        """Extract system messages as instruction string."""
        parts: list[str] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                text = msg.text
                if text:
                    parts.append(text)
        return "\n\n".join(parts) if parts else None

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> list[types.Content]:
        """Convert internal Messages to Gemini content format."""
        result: list[types.Content] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue  # Handled separately

            match msg.role:
                case Role.USER:
                    parts: list[types.Part] = []
                    for c in msg.content:
                        if c.text and c.text.strip():
                            parts.append(types.Part(text=c.text.rstrip()))

                        if c.tool_result:
                            data: Any
                            try:
                                data = json.loads(c.tool_result.data)
                            except (json.JSONDecodeError, TypeError):
                                data = {"output": c.tool_result.data}

                            if not isinstance(data, dict):
                                data = {"data": data}

                            parts.append(
                                types.Part(
                                    function_response=types.FunctionResponse(
                                        name=c.tool_result.id,
                                        response=data,
                                    )
                                )
                            )

                    if parts:
                        result.append(types.Content(role="user", parts=parts))

                case Role.ASSISTANT:
                    parts = []
                    for c in msg.content:
                        if c.text and c.text.strip():
                            parts.append(types.Part(text=c.text.rstrip()))

                        if c.tool_call:
                            args: dict[str, Any] = {}
                            if c.tool_call.arguments:
                                try:
                                    args = json.loads(c.tool_call.arguments)
                                except (json.JSONDecodeError, TypeError):
                                    pass

                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=c.tool_call.name,
                                        args=args,
                                    )
                                )
                            )

                    if parts:
                        result.append(types.Content(role="model", parts=parts))

        return result

    def _convert_tools(
        self,
        tools: list | None,
    ) -> list[types.Tool] | None:
        """Convert internal Tool list to Gemini tool format."""
        if not tools:
            return None

        declarations: list[types.FunctionDeclaration] = []
        for t in tools:
            if not t.name:
                continue
            declarations.append(
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters=t.parameters or None,
                )
            )

        if not declarations:
            return None

        return [types.Tool(function_declarations=declarations)]


# ── Helpers ───────────────────────────────────────────────────────


def _to_content(parts: list) -> list[Content]:
    """Convert Gemini parts to internal Content list."""
    result: list[Content] = []
    for p in parts:
        if getattr(p, "thought", False) and p.text:
            result.append(Content(reasoning=Reasoning(text=p.text)))
            continue

        if p.text:
            result.append(Content(text=p.text))

        if p.function_call:
            args = json.dumps(p.function_call.args) if p.function_call.args else "{}"
            result.append(
                Content(
                    tool_call=ToolCall(
                        id=getattr(p.function_call, "id", "") or p.function_call.name,
                        name=p.function_call.name,
                        arguments=args,
                    )
                )
            )

    return result


def _to_usage(metadata: Any) -> Usage | None:
    if not metadata:
        return None

    input_tokens = getattr(metadata, "prompt_token_count", 0) or 0
    output_tokens = getattr(metadata, "candidates_token_count", 0) or 0

    if not input_tokens and not output_tokens:
        return None

    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
