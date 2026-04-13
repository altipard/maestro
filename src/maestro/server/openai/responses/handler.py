"""Responses API handler — POST /v1/responses.

Supports both streaming (SSE) and non-streaming responses.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from maestro.core.errors import ProviderError, status_code_from_error
from maestro.core.models import Accumulator, CompleteOptions, Schema, ToolOptions
from maestro.core.types import Verbosity

from .convert import to_effort, to_messages, to_response_outputs, to_tool_choice, to_tools
from .models import ResponsesRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/responses", response_model=None)
async def create_response(req: ResponsesRequest, request: Request):
    """POST /v1/responses — OpenAI Responses API."""
    config = request.app.state.config

    # Policy check
    from maestro.policy.policy import AccessDeniedError, Action, Resource

    try:
        await config.policy.verify(Resource.MODEL, req.model, Action.ACCESS)
    except AccessDeniedError:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "type": "invalid_request",
                    "message": f"Model not found: {req.model}",
                }
            },
        )

    try:
        completer = config.completer(req.model)
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "type": "invalid_request",
                    "message": f"Model not found: {req.model}",
                }
            },
        )

    messages = to_messages(req.input, req.instructions)
    tools = to_tools(req.tools)

    options = CompleteOptions()

    if tools:
        options = options.model_copy(update={"tools": tools})

    # Tool choice + parallel tool calls → ToolOptions
    tc = to_tool_choice(req.tool_choice) if req.tool_choice is not None else None
    if req.parallel_tool_calls is not None and not req.parallel_tool_calls:
        from maestro.core.types import ToolChoice as InternalToolChoice

        tool_opts = ToolOptions(
            choice=tc or InternalToolChoice.AUTO,
            disable_parallel_tool_calls=True,
        )
        options = options.model_copy(update={"tool_options": tool_opts})
    elif tc is not None:
        options = options.model_copy(update={"tool_choice": tc})

    if req.max_output_tokens is not None:
        options = options.model_copy(
            update={
                "max_tokens": req.max_output_tokens,
            }
        )

    if req.temperature is not None:
        options = options.model_copy(
            update={
                "temperature": req.temperature,
            }
        )

    if req.reasoning is not None:
        effort = to_effort(req.reasoning)
        if effort:
            options = options.model_copy(update={"effort": effort})

    # Response schema + verbosity from text config
    if req.text:
        if req.text.format:
            fmt = req.text.format
            if fmt.type == "json_object":
                options = options.model_copy(
                    update={
                        "structured_output": Schema(name="json_object"),
                    }
                )
            elif fmt.type == "json_schema" and fmt.schema_:
                options = options.model_copy(
                    update={
                        "response_schema": fmt.schema_,
                        "structured_output": Schema(
                            name=fmt.name,
                            description=fmt.description,
                            schema_=fmt.schema_,
                            strict=fmt.strict,
                        ),
                    }
                )

        if req.text.verbosity:
            verbosity_map = {
                "low": Verbosity.LOW,
                "medium": Verbosity.MEDIUM,
                "high": Verbosity.HIGH,
            }
            v = verbosity_map.get(req.text.verbosity)
            if v:
                options = options.model_copy(update={"verbosity": v})

    if req.stream:
        return StreamingResponse(
            _stream_response(completer, messages, options, req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        return await _complete_response(completer, messages, options, req)


async def _complete_response(
    completer: Any,
    messages: list,
    options: CompleteOptions,
    req: ResponsesRequest,
) -> JSONResponse:
    """Non-streaming: accumulate all chunks and return a Response."""
    try:
        result = await Accumulator.collect(completer.complete(messages, options))
    except ProviderError as exc:
        status = status_code_from_error(exc, 500)
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "type": "server_error" if status >= 500 else "invalid_request_error",
                    "message": str(exc),
                }
            },
        )

    response_id = result.id or f"resp_{uuid.uuid4().hex[:24]}"
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    output = to_response_outputs(result.message, message_id)

    from maestro.core.types import Status

    status = "completed"
    if result.status == Status.INCOMPLETE:
        status = "incomplete"
    elif result.status == Status.FAILED:
        status = "failed"

    response: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": result.model or req.model,
        "status": status,
        "output": output,
    }

    if result.usage:
        response["usage"] = {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": (result.usage.input_tokens + result.usage.output_tokens),
        }

    return JSONResponse(content=response)


async def _stream_response(
    completer: Any,
    messages: list,
    options: CompleteOptions,
    req: ResponsesRequest,
):
    """Streaming: emit SSE events following the Responses streaming protocol."""
    created_at = int(time.time())
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    seq = 0

    def next_seq() -> int:
        nonlocal seq
        n = seq
        seq += 1
        return n

    next_output_index = 0

    def reserve_output_index() -> int:
        nonlocal next_output_index
        idx = next_output_index
        next_output_index += 1
        return idx

    def sse(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def make_response(status: str, output: list) -> dict:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "model": req.model,
            "output": output,
        }

    # Emit response.created + response.in_progress
    yield sse(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": next_seq(),
            "response": make_response("in_progress", []),
        },
    )
    yield sse(
        "response.in_progress",
        {
            "type": "response.in_progress",
            "sequence_number": next_seq(),
            "response": make_response("in_progress", []),
        },
    )

    # State tracking
    has_output_item = False
    has_content_part = False
    message_output_index = 0
    text_parts: list[str] = []

    # Tool call tracking
    tool_call_indices: dict[str, int] = {}
    tool_call_started: set[str] = set()
    last_tool_call_id = ""

    # Reasoning tracking
    reasoning_id = ""
    has_reasoning_item = False
    has_reasoning_text_part = False
    has_reasoning_summary_part = False
    reasoning_output_index = 0
    reasoning_closed = False
    reasoning_text_parts: list[str] = []
    reasoning_summary_parts: list[str] = []
    reasoning_signature = ""

    def close_reasoning_events() -> list[str]:
        """Generate reasoning close events."""
        nonlocal reasoning_closed
        events: list[str] = []
        if not has_reasoning_item or reasoning_closed:
            return events
        reasoning_closed = True

        r_text = "".join(reasoning_text_parts)
        r_summary = "".join(reasoning_summary_parts)

        if reasoning_text_parts:
            events.append(
                sse(
                    "response.reasoning_text.done",
                    {
                        "type": "response.reasoning_text.done",
                        "sequence_number": next_seq(),
                        "item_id": reasoning_id,
                        "output_index": reasoning_output_index,
                        "content_index": 0,
                        "text": r_text,
                    },
                )
            )
            events.append(
                sse(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "sequence_number": next_seq(),
                        "item_id": reasoning_id,
                        "output_index": reasoning_output_index,
                        "content_index": 0,
                        "part": {"type": "reasoning_text", "text": r_text},
                    },
                )
            )

        if reasoning_summary_parts:
            events.append(
                sse(
                    "response.reasoning_summary_text.done",
                    {
                        "type": "response.reasoning_summary_text.done",
                        "sequence_number": next_seq(),
                        "item_id": reasoning_id,
                        "output_index": reasoning_output_index,
                        "summary_index": 0,
                        "text": r_summary,
                    },
                )
            )
            events.append(
                sse(
                    "response.reasoning_summary_part.done",
                    {
                        "type": "response.reasoning_summary_part.done",
                        "sequence_number": next_seq(),
                        "item_id": reasoning_id,
                        "output_index": reasoning_output_index,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": r_summary},
                    },
                )
            )

        # reasoning item done
        item: dict[str, Any] = {
            "id": reasoning_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "content": [],
        }
        if r_summary:
            item["summary"].append({"type": "summary_text", "text": r_summary})
        if r_text:
            item["content"].append({"type": "reasoning_text", "text": r_text})
        if reasoning_signature:
            item["encrypted_content"] = reasoning_signature

        events.append(
            sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": next_seq(),
                    "output_index": reasoning_output_index,
                    "item": item,
                },
            )
        )

        return events

    acc = Accumulator()

    try:
        async for chunk in completer.complete(messages, options):
            acc.add(chunk)

            if chunk.message is None:
                continue

            for cnt in chunk.message.content:
                # Reasoning content
                if cnt.reasoning:
                    r = cnt.reasoning
                    if r.id and not reasoning_id:
                        reasoning_id = r.id

                    if r.signature:
                        reasoning_signature = r.signature
                        if not has_reasoning_item:
                            has_reasoning_item = True
                            reasoning_output_index = reserve_output_index()
                            if not reasoning_id:
                                reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
                            yield sse(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "sequence_number": next_seq(),
                                    "output_index": reasoning_output_index,
                                    "item": {
                                        "id": reasoning_id,
                                        "type": "reasoning",
                                        "status": "in_progress",
                                        "summary": [],
                                        "content": [],
                                    },
                                },
                            )

                    if r.text:
                        if not has_reasoning_item:
                            has_reasoning_item = True
                            reasoning_output_index = reserve_output_index()
                            if not reasoning_id:
                                reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
                            yield sse(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "sequence_number": next_seq(),
                                    "output_index": reasoning_output_index,
                                    "item": {
                                        "id": reasoning_id,
                                        "type": "reasoning",
                                        "status": "in_progress",
                                        "summary": [],
                                        "content": [],
                                    },
                                },
                            )

                        if not has_reasoning_text_part:
                            has_reasoning_text_part = True
                            yield sse(
                                "response.content_part.added",
                                {
                                    "type": "response.content_part.added",
                                    "sequence_number": next_seq(),
                                    "item_id": reasoning_id,
                                    "output_index": reasoning_output_index,
                                    "content_index": 0,
                                    "part": {"type": "reasoning_text", "text": ""},
                                },
                            )

                        reasoning_text_parts.append(r.text)
                        yield sse(
                            "response.reasoning_text.delta",
                            {
                                "type": "response.reasoning_text.delta",
                                "sequence_number": next_seq(),
                                "item_id": reasoning_id,
                                "output_index": reasoning_output_index,
                                "content_index": 0,
                                "delta": r.text,
                            },
                        )

                    if r.summary:
                        if not has_reasoning_item:
                            has_reasoning_item = True
                            reasoning_output_index = reserve_output_index()
                            if not reasoning_id:
                                reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
                            yield sse(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "sequence_number": next_seq(),
                                    "output_index": reasoning_output_index,
                                    "item": {
                                        "id": reasoning_id,
                                        "type": "reasoning",
                                        "status": "in_progress",
                                        "summary": [],
                                        "content": [],
                                    },
                                },
                            )

                        if not has_reasoning_summary_part:
                            has_reasoning_summary_part = True
                            yield sse(
                                "response.reasoning_summary_part.added",
                                {
                                    "type": "response.reasoning_summary_part.added",
                                    "sequence_number": next_seq(),
                                    "item_id": reasoning_id,
                                    "output_index": reasoning_output_index,
                                    "summary_index": 0,
                                    "part": {"type": "summary_text", "text": ""},
                                },
                            )

                        reasoning_summary_parts.append(r.summary)
                        yield sse(
                            "response.reasoning_summary_text.delta",
                            {
                                "type": "response.reasoning_summary_text.delta",
                                "sequence_number": next_seq(),
                                "item_id": reasoning_id,
                                "output_index": reasoning_output_index,
                                "summary_index": 0,
                                "delta": r.summary,
                            },
                        )

                # Text content
                if cnt.text:
                    if not has_output_item:
                        # Close reasoning first
                        for ev in close_reasoning_events():
                            yield ev

                        has_output_item = True
                        message_output_index = reserve_output_index()
                        yield sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "sequence_number": next_seq(),
                                "output_index": message_output_index,
                                "item": {
                                    "id": message_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "content": [],
                                    "role": "assistant",
                                },
                            },
                        )

                    if not has_content_part:
                        has_content_part = True
                        yield sse(
                            "response.content_part.added",
                            {
                                "type": "response.content_part.added",
                                "sequence_number": next_seq(),
                                "item_id": message_id,
                                "output_index": message_output_index,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": ""},
                            },
                        )

                    text_parts.append(cnt.text)
                    yield sse(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "sequence_number": next_seq(),
                            "item_id": message_id,
                            "output_index": message_output_index,
                            "content_index": 0,
                            "delta": cnt.text,
                        },
                    )

                # Tool calls
                if cnt.tool_call:
                    tc = cnt.tool_call
                    tc_id = tc.id or last_tool_call_id

                    if tc.id:
                        if tc.id not in tool_call_indices:
                            tool_call_indices[tc.id] = reserve_output_index()
                        last_tool_call_id = tc.id

                    if tc_id and tc_id not in tool_call_started:
                        tool_call_started.add(tc_id)
                        yield sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "sequence_number": next_seq(),
                                "output_index": tool_call_indices.get(tc_id, 0),
                                "item": {
                                    "id": tc_id,
                                    "type": "function_call",
                                    "status": "in_progress",
                                    "call_id": tc_id,
                                    "name": tc.name,
                                    "arguments": "",
                                },
                            },
                        )

                    if tc_id and tc.arguments:
                        yield sse(
                            "response.function_call_arguments.delta",
                            {
                                "type": "response.function_call_arguments.delta",
                                "sequence_number": next_seq(),
                                "item_id": tc_id,
                                "output_index": tool_call_indices.get(tc_id, 0),
                                "delta": tc.arguments,
                            },
                        )

    except Exception as exc:
        logger.error("streaming error: %s", exc)
        error_type = "server_error"
        if isinstance(exc, ProviderError):
            status = status_code_from_error(exc, 500)
            error_map = {
                401: "authentication_error",
                403: "permission_error",
                404: "not_found_error",
                429: "rate_limit_exceeded",
            }
            error_type = error_map.get(status, "server_error")
        yield sse(
            "response.failed",
            {
                "type": "response.failed",
                "sequence_number": next_seq(),
                "response": {
                    **make_response("failed", []),
                    "error": {"type": error_type, "message": str(exc)},
                },
            },
        )
        yield "data: [DONE]\n\n"
        return

    # ── Final events ───────────────────────────────────────────

    result = acc.result
    full_text = "".join(text_parts)

    # text.done + content_part.done + output_item.done for message
    if text_parts:
        yield sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "sequence_number": next_seq(),
                "item_id": message_id,
                "output_index": message_output_index,
                "content_index": 0,
                "text": full_text,
            },
        )
        yield sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "sequence_number": next_seq(),
                "item_id": message_id,
                "output_index": message_output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": full_text},
            },
        )
        yield sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": next_seq(),
                "output_index": message_output_index,
                "item": {
                    "id": message_id,
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": full_text}],
                    "role": "assistant",
                },
            },
        )

    # Close reasoning if still open
    for ev in close_reasoning_events():
        yield ev

    # function_call done events
    if result.message:
        for tc in result.message.tool_calls:
            oi = tool_call_indices.get(tc.id, 0)
            yield sse(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "sequence_number": next_seq(),
                    "item_id": tc.id,
                    "name": tc.name,
                    "output_index": oi,
                    "arguments": tc.arguments,
                },
            )
            yield sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": next_seq(),
                    "output_index": oi,
                    "item": {
                        "id": tc.id,
                        "type": "function_call",
                        "status": "completed",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                },
            )

    # Determine final status
    from maestro.core.types import Status

    final_status = "completed"
    if result.status == Status.INCOMPLETE:
        final_status = "incomplete"
    elif result.status == Status.FAILED:
        final_status = "failed"

    output = to_response_outputs(result.message, message_id)
    final_response = make_response(final_status, output)
    if result.usage:
        final_response["usage"] = {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": (result.usage.input_tokens + result.usage.output_tokens),
        }

    if final_status == "incomplete":
        yield sse(
            "response.incomplete",
            {
                "type": "response.incomplete",
                "sequence_number": next_seq(),
                "response": final_response,
            },
        )
    else:
        yield sse(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": next_seq(),
                "response": final_response,
            },
        )

    yield "data: [DONE]\n\n"
