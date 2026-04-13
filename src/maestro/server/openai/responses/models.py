"""Pydantic models for the OpenAI Responses API.

https://platform.openai.com/docs/api-reference/responses/create
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Request models ─────────────────────────────────────────────


class ReasoningConfig(BaseModel):
    effort: str | None = None  # none, minimal, low, medium, high, xhigh


class TextFormat(BaseModel):
    type: str = "text"  # text, json_object, json_schema
    name: str = ""
    description: str = ""
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class TextConfig(BaseModel):
    format: TextFormat | None = None
    verbosity: str | None = None  # low, medium, high


class FunctionTool(BaseModel):
    type: str = "function"
    name: str = ""
    description: str = ""
    strict: bool | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolChoice(BaseModel):
    """Flexible tool_choice: can be a string or object."""
    mode: str = "auto"  # none, auto, required


class InputContent(BaseModel):
    type: str = ""  # input_text, input_image, input_file, output_text
    text: str = ""
    image_url: str = ""
    filename: str = ""
    file_url: str = ""
    file_data: str = ""


class InputMessage(BaseModel):
    role: str = "user"
    content: str | list[InputContent] = ""


class InputReasoning(BaseModel):
    id: str = ""
    summary: list[dict[str, str]] = Field(default_factory=list)
    content: Any | None = None  # Can be null — reasoning content from client
    encrypted_content: str = ""


class InputFunctionCall(BaseModel):
    id: str = ""
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    status: str = ""


class InputFunctionCallOutput(BaseModel):
    call_id: str = ""
    output: str = ""


class InputItem(BaseModel):
    type: str = "message"  # message, reasoning, function_call, function_call_output

    # Depending on type, one of these is populated.
    # We use a flat model and pick based on type.
    role: str | None = None
    content: str | list[InputContent] | None = None
    text: str | None = None

    # reasoning
    id: str | None = None
    summary: list[dict[str, str]] | None = None
    encrypted_content: str | None = None

    # function_call
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    status: str | None = None

    # function_call_output
    output: str | None = None


class ResponsesRequest(BaseModel):
    model: str
    stream: bool = False
    instructions: str = ""
    input: str | list[InputItem] = Field(default_factory=list)
    tools: list[FunctionTool] = Field(default_factory=list)
    text: TextConfig | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    reasoning: ReasoningConfig | None = None
    tool_choice: str | ToolChoice | None = None
    parallel_tool_calls: bool | None = None


# ── Response models ────────────────────────────────────────────


class ResponseUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponseError(BaseModel):
    type: str = "server_error"
    code: str = ""
    param: str = ""
    message: str = ""


class OutputContent(BaseModel):
    type: str = "output_text"
    text: str = ""


class OutputMessage(BaseModel):
    id: str = ""
    role: str = "assistant"
    status: str = "completed"
    content: list[OutputContent] = Field(default_factory=list)


class FunctionCallOutput(BaseModel):
    id: str = ""
    type: str = "function_call"
    status: str = "completed"
    name: str = ""
    call_id: str = ""
    arguments: str = ""


class ReasoningSummary(BaseModel):
    type: str = "summary_text"
    text: str = ""


class ReasoningContentPart(BaseModel):
    type: str = "reasoning_text"
    text: str = ""


class ReasoningOutput(BaseModel):
    id: str = ""
    type: str = "reasoning"
    status: str = "completed"
    summary: list[ReasoningSummary] = Field(default_factory=list)
    content: list[ReasoningContentPart] = Field(default_factory=list)
    encrypted_content: str = ""


class ResponseOutput(BaseModel):
    """A single output item — message, function_call, or reasoning."""
    type: str  # message, function_call, reasoning
    # Only one of these will be populated, depending on type.
    message: OutputMessage | None = None
    function_call: FunctionCallOutput | None = None
    reasoning: ReasoningOutput | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Flatten output: merge the active variant's fields into the top level."""
        kwargs.setdefault("exclude_none", True)
        if self.type == "message" and self.message:
            d = {"type": self.type}
            d.update(self.message.model_dump(**kwargs))
            return d
        if self.type == "function_call" and self.function_call:
            d = {"type": self.type}
            d.update(self.function_call.model_dump(**kwargs))
            return d
        if self.type == "reasoning" and self.reasoning:
            d = {"type": self.type}
            d.update(self.reasoning.model_dump(**kwargs))
            return d
        return super().model_dump(**kwargs)


class Response(BaseModel):
    id: str = ""
    object: str = "response"
    created_at: int = 0
    model: str = ""
    status: str = "completed"
    output: list[ResponseOutput] = Field(default_factory=list)
    usage: ResponseUsage | None = None
    error: ResponseError | None = None
