"""POST /v1/chat/completions — streaming and non-streaming.

FastAPI handles JSON parsing/validation,
SSE is a simple async generator.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from maestro.config import Config
from maestro.core.errors import ProviderError, retry_after_from_error, status_code_from_error
from maestro.core.models import Accumulator
from maestro.policy.policy import AccessDeniedError, Action, Resource

from .convert import to_complete_options, to_messages, to_oai_tool_calls
from .models import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionUsage,
    ErrorDetail,
    ErrorResponse,
    OAIFunctionCall,
    OAIToolCall,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _oai_error_type(status_code: int) -> str:
    """Map HTTP status to OpenAI-compatible error type string."""
    mapping = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_exceeded",
    }
    return mapping.get(status_code, "server_error")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            error=ErrorDetail(type=_oai_error_type(status), message=message),
        ).model_dump(),
    )


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    req: ChatCompletionRequest, request: Request,
) -> StreamingResponse | JSONResponse:
    cfg: Config = request.app.state.config

    # Policy check
    try:
        await cfg.policy.verify(Resource.MODEL, req.model, Action.ACCESS)
    except AccessDeniedError:
        return _error(404, f"model not found: {req.model}")

    # Look up completer
    try:
        completer = cfg.completer(req.model)
    except KeyError:
        return _error(400, f"model not found: {req.model}")

    # Convert request
    messages = to_messages(req.messages)
    options = to_complete_options(req)

    if req.stream:
        return _stream_response(req, completer, messages, options)

    return await _complete_response(req, completer, messages, options)


async def _complete_response(req, completer, messages, options) -> JSONResponse:
    """Non-streaming: accumulate all chunks, return single JSON response."""
    try:
        result = await Accumulator.collect(completer.complete(messages, options))
    except ProviderError as exc:
        status = status_code_from_error(exc, 500)
        return _error(status, str(exc))

    role = "assistant"
    now = int(time.time())

    resp = ChatCompletion(
        object="chat.completion",
        id=result.id,
        model=result.model or req.model,
        created=now,
    )

    if result.message:
        msg = ChatCompletionMessage(role=role)

        text = result.message.text
        if text:
            msg.content = text

        tool_calls = to_oai_tool_calls(result.message.content)
        finish_reason = "stop"

        if tool_calls:
            finish_reason = "tool_calls"
            msg.content = None
            msg.tool_calls = tool_calls

        resp.choices = [
            ChatCompletionChoice(
                message=msg,
                finish_reason=finish_reason,
            )
        ]

    if result.usage:
        resp.usage = ChatCompletionUsage(
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
            total_tokens=result.usage.input_tokens + result.usage.output_tokens,
        )

    return JSONResponse(content=resp.model_dump(exclude_none=True))


def _stream_response(req, completer, messages, options) -> StreamingResponse:
    """Streaming: yield SSE events as chunks arrive."""

    async def event_generator():
        now = int(time.time())
        model = req.model
        streamed_role = False
        finish_reason = "stop"

        # Track tool call indices across chunks
        tool_call_indices: dict[str, int] = {}
        current_tool_call_id = ""

        acc = Accumulator()

        try:
            async for chunk in completer.complete(messages, options):
                acc.add(chunk)

                # Skip usage-only chunks
                if chunk.usage and (not chunk.message or not chunk.message.content):
                    continue

                delta = ChatCompletionMessage()

                if not streamed_role:
                    streamed_role = True
                    delta.role = "assistant"

                if chunk.message:
                    text = chunk.message.text
                    if text:
                        delta.content = text

                    # Handle tool calls
                    oai_calls: list[OAIToolCall] = []
                    for c in chunk.message.content:
                        if c.tool_call:
                            tc = c.tool_call
                            if tc.id:
                                current_tool_call_id = tc.id
                                if current_tool_call_id not in tool_call_indices:
                                    tool_call_indices[current_tool_call_id] = len(
                                        tool_call_indices
                                    )

                            if current_tool_call_id:
                                oai_calls.append(
                                    OAIToolCall(
                                        id=current_tool_call_id,
                                        index=tool_call_indices.get(current_tool_call_id, 0),
                                        type="function",
                                        function=OAIFunctionCall(
                                            name=tc.name,
                                            arguments=tc.arguments,
                                        ),
                                    )
                                )

                    if oai_calls:
                        finish_reason = "tool_calls"
                        delta.content = None
                        delta.tool_calls = oai_calls

                event = ChatCompletion(
                    object="chat.completion.chunk",
                    id=chunk.id or "",
                    model=chunk.model or model,
                    created=now,
                    choices=[ChatCompletionChoice(delta=delta)],
                )
                yield f"data: {event.model_dump_json(exclude_none=True)}\n\n"

        except Exception as exc:
            # Emit error as SSE event instead of killing the connection
            logger.error("streaming error: %s", exc)
            status = status_code_from_error(exc, 500)
            error_type = _oai_error_type(status)
            err_resp = ErrorResponse(
                error=ErrorDetail(type=error_type, message=str(exc))
            )
            yield f"data: {err_resp.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Finish reason chunk
        result = acc.result
        finish_chunk = ChatCompletion(
            object="chat.completion.chunk",
            id=result.id,
            model=result.model or model,
            created=now,
            choices=[
                ChatCompletionChoice(
                    delta=ChatCompletionMessage(),
                    finish_reason=finish_reason,
                )
            ],
        )
        yield f"data: {finish_chunk.model_dump_json(exclude_none=True)}\n\n"

        # Usage chunk if requested
        include_usage = req.stream_options and req.stream_options.include_usage
        if include_usage and result.usage:
            usage_chunk = ChatCompletion(
                object="chat.completion.chunk",
                id=result.id,
                model=result.model or model,
                created=now,
                choices=[],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.usage.input_tokens,
                    completion_tokens=result.usage.output_tokens,
                    total_tokens=result.usage.input_tokens + result.usage.output_tokens,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
