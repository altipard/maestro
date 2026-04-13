"""Convert Responses API types to internal maestro types."""

from __future__ import annotations

from typing import Any

from maestro.core.models import (
    Content,
    Message,
    Reasoning,
    Tool,
    ToolCall,
    ToolResult,
)
from maestro.core.types import Effort, Role, ToolChoice
from maestro.tools import normalize_schema

from .models import (
    FunctionTool,
    InputItem,
)


def to_messages(
    input_data: str | list[InputItem],
    instructions: str = "",
) -> list[Message]:
    """Convert Responses API input to internal Message list."""
    result: list[Message] = []

    if instructions:
        result.append(
            Message(
                role=Role.SYSTEM,
                content=[Content(text=instructions)],
            )
        )

    if isinstance(input_data, str):
        result.append(
            Message(
                role=Role.USER,
                content=[Content(text=input_data)],
            )
        )
        return result

    # Pending buffers for batching
    pending_reasoning: list[Content] = []
    pending_calls: list[Content] = []
    pending_results: list[Content] = []

    def flush_calls() -> None:
        nonlocal pending_reasoning, pending_calls
        if not pending_calls and not pending_reasoning:
            return
        result.append(
            Message(
                role=Role.ASSISTANT,
                content=pending_reasoning + pending_calls,
            )
        )
        pending_reasoning = []
        pending_calls = []

    def flush_results() -> None:
        nonlocal pending_results
        if not pending_results:
            return
        result.append(
            Message(
                role=Role.USER,
                content=pending_results,
            )
        )
        pending_results = []

    for item in input_data:
        match item.type:
            case "message" | "":
                content_parts = _to_input_content(item)

                role = _to_role(item.role or "user")

                if role == Role.ASSISTANT:
                    flush_results()
                    content_parts = pending_reasoning + content_parts
                    pending_reasoning = []
                    if content_parts:
                        result.append(
                            Message(
                                role=Role.ASSISTANT,
                                content=content_parts,
                            )
                        )
                else:
                    flush_calls()
                    flush_results()
                    if content_parts:
                        result.append(
                            Message(
                                role=role,
                                content=content_parts,
                            )
                        )

            case "reasoning":
                r = Reasoning(id=item.id or "")

                if item.summary:
                    for part in item.summary:
                        if part.get("type") == "summary_text":
                            r = r.model_copy(
                                update={
                                    "summary": r.summary + part.get("text", ""),
                                }
                            )

                if item.encrypted_content:
                    r = r.model_copy(
                        update={
                            "signature": item.encrypted_content,
                        }
                    )

                pending_reasoning.append(Content(reasoning=r))

            case "function_call":
                flush_results()
                pending_calls.append(
                    Content(
                        tool_call=ToolCall(
                            id=item.call_id or "",
                            name=item.name or "",
                            arguments=item.arguments or "",
                        )
                    )
                )

            case "function_call_output":
                flush_calls()
                pending_results.append(
                    Content(
                        tool_result=ToolResult(
                            id=item.call_id or "",
                            data=item.output or "",
                        )
                    )
                )

    flush_calls()
    flush_results()

    return result


def to_tools(tools: list[FunctionTool]) -> list[Tool]:
    """Convert Responses API tools to internal Tool list."""
    result: list[Tool] = []
    for t in tools:
        if t.type != "function":
            continue
        result.append(
            Tool(
                name=t.name,
                description=t.description,
                strict=t.strict,
                parameters=normalize_schema(t.parameters),
            )
        )
    return result


def to_tool_choice(choice: str | dict | Any | None) -> ToolChoice | None:
    """Convert Responses API tool_choice to internal ToolChoice."""
    if choice is None:
        return None

    if isinstance(choice, str):
        match choice:
            case "none":
                return ToolChoice.NONE
            case "required":
                return ToolChoice.ANY
            case _:
                return ToolChoice.AUTO

    # Object with mode
    if hasattr(choice, "mode"):
        match choice.mode:
            case "none":
                return ToolChoice.NONE
            case "required":
                return ToolChoice.ANY
            case _:
                return ToolChoice.AUTO

    return ToolChoice.AUTO


def to_effort(reasoning: Any) -> Effort | None:
    """Convert Responses API reasoning config to internal Effort (all 6 levels)."""
    if reasoning is None or reasoning.effort is None:
        return None

    mapping = {
        "none": Effort.NONE,
        "minimal": Effort.MINIMAL,
        "low": Effort.LOW,
        "medium": Effort.MEDIUM,
        "high": Effort.HIGH,
        "xhigh": Effort.MAX,
    }
    return mapping.get(reasoning.effort)


def to_response_outputs(message: Message | None, message_id: str) -> list[dict]:
    """Convert internal Message to Responses API output format."""
    if message is None:
        return []

    outputs: list[dict] = []

    # Reasoning output
    for c in message.content:
        if c.reasoning and c.reasoning.id:
            reasoning: dict[str, Any] = {
                "type": "reasoning",
                "id": c.reasoning.id,
                "status": "completed",
                "summary": [],
                "content": [],
            }

            if c.reasoning.summary:
                reasoning["summary"].append(
                    {
                        "type": "summary_text",
                        "text": c.reasoning.summary,
                    }
                )

            if c.reasoning.text:
                reasoning["content"].append(
                    {
                        "type": "reasoning_text",
                        "text": c.reasoning.text,
                    }
                )

            if c.reasoning.signature:
                reasoning["encrypted_content"] = c.reasoning.signature

            outputs.append(reasoning)
            break

    # Tool calls
    for tc in message.tool_calls:
        outputs.append(
            {
                "type": "function_call",
                "id": tc.id,
                "call_id": tc.id,
                "status": "completed",
                "name": tc.name,
                "arguments": tc.arguments,
            }
        )

    # Text output
    text = message.text
    if text:
        outputs.append(
            {
                "type": "message",
                "id": message_id,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        )

    return outputs


# ── Internal helpers ───────────────────────────────────────────


def _to_input_content(item: InputItem) -> list[Content]:
    """Extract content parts from an InputItem."""
    parts: list[Content] = []

    content = item.content
    if content is None and item.text:
        content = item.text

    if isinstance(content, str):
        if content.strip():
            parts.append(Content(text=content))
    elif isinstance(content, list):
        for c in content:
            match c.type:
                case "input_text" | "output_text" | "":
                    if c.text.strip():
                        parts.append(Content(text=c.text))
                # Image/file support can be added later

    return parts


def _to_role(role: str | None) -> Role:
    """Convert Responses API role string to internal Role."""
    match role:
        case "system" | "developer":
            return Role.SYSTEM
        case "assistant":
            return Role.ASSISTANT
        case _:
            return Role.USER
