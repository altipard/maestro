from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from .types import Effort, Role, Status, ToolChoice, Verbosity

# ── Shared building blocks ──────────────────────────────────────────


class Usage(BaseModel, frozen=True):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class FileData(BaseModel):
    name: str
    content: bytes
    content_type: str = ""


# ── Completion types ────────────────────────────────────────────────


class ToolCall(BaseModel):
    id: str = ""
    name: str = ""
    arguments: str = ""


class ToolResult(BaseModel):
    id: str
    data: str = ""


class Reasoning(BaseModel):
    id: str = ""
    text: str = ""
    summary: str = ""
    signature: str = ""


class Content(BaseModel):
    text: str | None = None
    file: FileData | None = None
    reasoning: Reasoning | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


class Message(BaseModel):
    role: Role
    content: list[Content] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(c.text for c in self.content if c.text)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [c.tool_call for c in self.content if c.tool_call]

    # ── Factory methods ─────────────────────────────────────────

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role=Role.SYSTEM, content=[Content(text=text)])

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role=Role.USER, content=[Content(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role=Role.ASSISTANT, content=[Content(text=text)])

    @classmethod
    def tool(cls, id: str, data: str) -> Message:
        return cls(role=Role.USER, content=[Content(tool_result=ToolResult(id=id, data=data))])


class Tool(BaseModel):
    name: str
    description: str = ""
    strict: bool | None = None
    parameters: dict = Field(default_factory=dict)


class ToolOptions(BaseModel):
    choice: ToolChoice = ToolChoice.AUTO
    allowed: list[str] = Field(default_factory=list)
    disable_parallel_tool_calls: bool = False


class Schema(BaseModel):
    name: str = ""
    description: str = ""
    schema_: dict | None = Field(default=None, alias="schema")
    strict: bool | None = None

    model_config = {"populate_by_name": True}


class CompleteOptions(BaseModel):
    effort: Effort | None = None
    verbosity: Verbosity | None = None
    stop: list[str] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    tool_options: ToolOptions | None = None
    response_schema: dict | None = None
    structured_output: Schema | None = None


class Completion(BaseModel):
    id: str = ""
    model: str = ""
    status: Status = Status.COMPLETED
    message: Message | None = None
    usage: Usage | None = None


# ── Embedding types ─────────────────────────────────────────────────


class EmbedOptions(BaseModel):
    dimensions: int | None = None


class Embedding(BaseModel):
    model: str = ""
    embeddings: list[list[float]] = Field(default_factory=list)
    usage: Usage | None = None


# ── Search types ────────────────────────────────────────────────────


class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    content: str = ""
    score: float = 0.0


# ── Pipeline types (satellite project interfaces) ──────────────────


class Segment(BaseModel):
    text: str
    metadata: dict = Field(default_factory=dict)


class RankedDocument(BaseModel):
    index: int
    text: str
    score: float


# ── Stream accumulator ──────────────────────────────────────────────


class Accumulator:
    """Accumulates streaming completion chunks into a final result.

    Usage:
        acc = Accumulator()
        async for chunk in completer.complete(messages):
            acc.add(chunk)
        result = acc.result

        # Or one-liner:
        result = await Accumulator.collect(completer.complete(messages))
    """

    def __init__(self) -> None:
        self._id = ""
        self._model = ""
        self._status = Status.COMPLETED
        self._role = Role.ASSISTANT
        self._text_parts: list[str] = []
        self._reasoning: Reasoning | None = None
        self._tool_calls: dict[str, ToolCall] = {}
        self._usage = Usage()

    def add(self, chunk: Completion) -> None:
        if chunk.id:
            self._id = chunk.id
        if chunk.model:
            self._model = chunk.model
        if chunk.status:
            self._status = chunk.status

        if chunk.message:
            if chunk.message.role:
                self._role = chunk.message.role

            for c in chunk.message.content:
                if c.text:
                    self._text_parts.append(c.text)

                if c.reasoning:
                    if self._reasoning is None:
                        self._reasoning = Reasoning()
                    if c.reasoning.id:
                        self._reasoning = self._reasoning.model_copy(
                            update={"id": c.reasoning.id}
                        )
                    self._reasoning = self._reasoning.model_copy(
                        update={
                            "text": self._reasoning.text + c.reasoning.text,
                            "summary": self._reasoning.summary + c.reasoning.summary,
                            **(
                                {"signature": c.reasoning.signature}
                                if c.reasoning.signature
                                else {}
                            ),
                        }
                    )

                if c.tool_call:
                    tc = self._tool_calls.get(c.tool_call.id, ToolCall(id=c.tool_call.id))
                    if c.tool_call.name:
                        tc = tc.model_copy(update={"name": c.tool_call.name})
                    tc = tc.model_copy(update={"arguments": tc.arguments + c.tool_call.arguments})
                    self._tool_calls[tc.id] = tc

        if chunk.usage:
            self._usage = Usage(
                input_tokens=max(self._usage.input_tokens, chunk.usage.input_tokens),
                output_tokens=max(self._usage.output_tokens, chunk.usage.output_tokens),
                cache_read_input_tokens=max(
                    self._usage.cache_read_input_tokens,
                    chunk.usage.cache_read_input_tokens,
                ),
                cache_creation_input_tokens=max(
                    self._usage.cache_creation_input_tokens,
                    chunk.usage.cache_creation_input_tokens,
                ),
            )

    @property
    def result(self) -> Completion:
        content: list[Content] = []

        if self._reasoning:
            content.append(Content(reasoning=self._reasoning))
        if self._text_parts:
            content.append(Content(text="".join(self._text_parts)))
        for tc in self._tool_calls.values():
            content.append(Content(tool_call=tc))

        return Completion(
            id=self._id,
            model=self._model,
            status=self._status,
            message=Message(role=self._role, content=content),
            usage=self._usage if self._usage.input_tokens or self._usage.output_tokens else None,
        )

    @classmethod
    async def collect(cls, stream: AsyncIterator[Completion]) -> Completion:
        acc = cls()
        async for chunk in stream:
            acc.add(chunk)
        return acc.result
