"""OpenAI-compatible API request/response models.

Uses Pydantic v2 for automatic JSON serialization.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Chat Completions ───────────────────────────────────────────────


class FunctionDef(BaseModel):
    name: str
    description: str = ""
    strict: bool | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class OAITool(BaseModel):
    type: str = "function"
    function: FunctionDef | None = None


class OAIFunctionCall(BaseModel):
    name: str = ""
    arguments: str = ""


class OAIToolCall(BaseModel):
    id: str = ""
    type: str = "function"
    index: int = 0
    function: OAIFunctionCall | None = None


class MessageContent(BaseModel):
    type: str = "text"
    text: str = ""


class ChatCompletionMessage(BaseModel):
    role: str | None = None
    content: str | list[MessageContent] | None = None
    tool_calls: list[OAIToolCall] | None = None
    tool_call_id: str | None = None


class ResponseFormatSchema(BaseModel):
    name: str = ""
    description: str = ""
    strict: bool | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")

    model_config = {"populate_by_name": True}


class ChatCompletionResponseFormat(BaseModel):
    type: str = "text"
    json_schema: ResponseFormatSchema | None = None


class StreamOptions(BaseModel):
    include_usage: bool | None = None


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatCompletionMessage] = Field(default_factory=list)

    stream: bool = False
    stop: str | list[str] | None = None
    tools: list[OAITool] | None = None
    tool_choice: str | dict[str, Any] | None = None

    temperature: float | None = None
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None
    verbosity: str | None = None
    parallel_tool_calls: bool | None = None

    response_format: ChatCompletionResponseFormat | None = None
    stream_options: StreamOptions | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage | None = None
    delta: ChatCompletionMessage | None = None
    finish_reason: str | None = None


class ChatCompletion(BaseModel):
    object: str = "chat.completion"
    id: str = ""
    model: str = ""
    created: int = 0
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: ChatCompletionUsage | None = None


# ── Embeddings ─────────────────────────────────────────────────────


class EmbeddingsRequest(BaseModel):
    model: str = ""
    # All four shapes from the OpenAI Embeddings spec. Token-ID variants are
    # decoded to text at the API edge (see server/openai/tokens.py).
    input: str | list[str] | list[int] | list[list[int]] = ""
    dimensions: int | None = None
    encoding_format: str | None = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: list[float] | str = Field(default_factory=list)


class EmbeddingList(BaseModel):
    object: str = "list"
    model: str = ""
    data: list[EmbeddingData] = Field(default_factory=list)
    usage: ChatCompletionUsage | None = None


# ── Models ─────────────────────────────────────────────────────────


class ModelObject(BaseModel):
    object: str = "model"
    id: str = ""
    created: int = 0
    owned_by: str = "maestro"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelObject] = Field(default_factory=list)


# ── Errors ─────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    type: str = "invalid_request_error"
    message: str = ""
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
