"""Convert between OpenAI API types and internal maestro types."""

from __future__ import annotations

from maestro.core.models import (
    CompleteOptions,
    Content,
    Message,
    Tool,
    ToolCall,
    ToolOptions,
    ToolResult,
)
from maestro.core.types import Effort, Role, ToolChoice, Verbosity

from .models import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    OAIFunctionCall,
    OAITool,
    OAIToolCall,
)

# ── Messages ───────────────────────────────────────────────────────


def to_messages(oai_messages: list[ChatCompletionMessage]) -> list[Message]:
    """Convert OpenAI ChatCompletionMessage list to internal Messages."""
    result: list[Message] = []

    for m in oai_messages:
        role = _to_role(m.role)
        if role is None:
            continue

        content: list[Content] = []

        if isinstance(m.content, list):
            # Array-of-content-objects format
            for c in m.content:
                if c.type == "text" and c.text:
                    content.append(Content(text=c.text))
        elif m.tool_call_id:
            # Tool result message
            content.append(
                Content(tool_result=ToolResult(id=m.tool_call_id, data=m.content or ""))
            )
        elif m.content is not None:
            # Simple string content
            content.append(Content(text=m.content))

        # Tool calls from assistant messages
        if m.tool_calls:
            for tc in m.tool_calls:
                if tc.type == "function" and tc.function:
                    content.append(
                        Content(
                            tool_call=ToolCall(
                                id=tc.id,
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            )
                        )
                    )

        result.append(Message(role=role, content=content))

    return result


def _to_role(role: str) -> Role | None:
    """Map OpenAI role string to internal Role enum."""
    match role:
        case "system" | "developer":
            return Role.SYSTEM
        case "user" | "tool":
            return Role.USER
        case "assistant":
            return Role.ASSISTANT
        case _:
            return None


# ── Tools ──────────────────────────────────────────────────────────


def to_tools(oai_tools: list[OAITool] | None) -> list[Tool] | None:
    """Convert OpenAI tool definitions to internal Tool list."""
    if not oai_tools:
        return None

    result: list[Tool] = []
    for t in oai_tools:
        if t.type == "function" and t.function:
            result.append(
                Tool(
                    name=t.function.name,
                    description=t.function.description,
                    strict=t.function.strict,
                    parameters=t.function.parameters,
                )
            )
    return result or None


# ── Tool choice ────────────────────────────────────────────────────


def to_tool_choice(raw: str | dict | None) -> ToolChoice | None:
    """Convert OpenAI tool_choice to internal ToolChoice enum."""
    if raw is None:
        return None

    if isinstance(raw, str):
        match raw:
            case "none":
                return ToolChoice.NONE
            case "auto":
                return ToolChoice.AUTO
            case "required":
                return ToolChoice.ANY

    # {"type": "function", "function": {"name": "X"}} → ANY
    if isinstance(raw, dict):
        return ToolChoice.ANY

    return None


# ── Complete options ───────────────────────────────────────────────


def to_complete_options(req: ChatCompletionRequest) -> CompleteOptions | None:
    """Build CompleteOptions from an OpenAI ChatCompletionRequest."""
    stops: list[str] | None = None
    if isinstance(req.stop, str):
        stops = [req.stop]
    elif isinstance(req.stop, list):
        stops = req.stop

    effort = _to_effort(req.reasoning_effort)
    verbosity = _to_verbosity(req.verbosity)

    tools = to_tools(req.tools)
    tool_choice = to_tool_choice(req.tool_choice)

    # Build tool options from tool_choice + parallel_tool_calls
    tool_opts: ToolOptions | None = None
    if req.parallel_tool_calls is not None and not req.parallel_tool_calls:
        tc = tool_choice or ToolChoice.AUTO
        tool_opts = ToolOptions(choice=tc, disable_parallel_tool_calls=True)
        tool_choice = None  # ToolOptions takes precedence

    response_schema: dict | None = None
    if req.response_format and req.response_format.json_schema:
        js = req.response_format.json_schema
        response_schema = js.schema_ or {}

    opts = CompleteOptions(
        effort=effort,
        verbosity=verbosity,
        stop=stops,
        max_tokens=req.max_completion_tokens,
        temperature=req.temperature,
        tools=tools,
        tool_choice=tool_choice,
        tool_options=tool_opts,
        response_schema=response_schema,
    )

    # Only return if at least one option is set
    if opts == CompleteOptions():
        return None

    return opts


def _to_effort(raw: str | None) -> Effort | None:
    """Map effort string to Effort enum (all 6 levels)."""
    if raw is None:
        return None
    mapping = {
        "none": Effort.NONE,
        "minimal": Effort.MINIMAL,
        "low": Effort.LOW,
        "medium": Effort.MEDIUM,
        "high": Effort.HIGH,
        "xhigh": Effort.MAX,
    }
    return mapping.get(raw)


def _to_verbosity(raw: str | None) -> Verbosity | None:
    """Map verbosity string to Verbosity enum."""
    if raw is None:
        return None
    mapping = {
        "low": Verbosity.LOW,
        "medium": Verbosity.MEDIUM,
        "high": Verbosity.HIGH,
    }
    return mapping.get(raw)


# ── Response conversion ────────────────────────────────────────────


def to_oai_tool_calls(content: list[Content]) -> list[OAIToolCall]:
    """Convert internal Content tool calls to OpenAI ToolCall format."""
    result: list[OAIToolCall] = []
    for c in content:
        if c.tool_call:
            result.append(
                OAIToolCall(
                    id=c.tool_call.id,
                    index=len(result),
                    type="function",
                    function=OAIFunctionCall(
                        name=c.tool_call.name,
                        arguments=c.tool_call.arguments,
                    ),
                )
            )
    return result
