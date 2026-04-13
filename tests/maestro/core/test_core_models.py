"""Tests for core model changes: ToolOptions, Verbosity, Effort levels, Usage cache tokens."""

from __future__ import annotations

from maestro.core.models import (
    Accumulator,
    CompleteOptions,
    Completion,
    Content,
    Message,
    Schema,
    ToolOptions,
    Usage,
)
from maestro.core.types import Effort, Role, ToolChoice, Verbosity

# ── Effort enum ──────────────────────────────────────────────────


class TestEffort:
    def test_all_six_levels_exist(self) -> None:
        assert Effort.NONE == "none"
        assert Effort.MINIMAL == "minimal"
        assert Effort.LOW == "low"
        assert Effort.MEDIUM == "medium"
        assert Effort.HIGH == "high"
        assert Effort.MAX == "max"

    def test_effort_count(self) -> None:
        assert len(Effort) == 6


# ── Verbosity enum ───────────────────────────────────────────────


class TestVerbosity:
    def test_all_levels(self) -> None:
        assert Verbosity.LOW == "low"
        assert Verbosity.MEDIUM == "medium"
        assert Verbosity.HIGH == "high"

    def test_verbosity_count(self) -> None:
        assert len(Verbosity) == 3


# ── ToolOptions ──────────────────────────────────────────────────


class TestToolOptions:
    def test_defaults(self) -> None:
        opts = ToolOptions()
        assert opts.choice == ToolChoice.AUTO
        assert opts.allowed == []
        assert opts.disable_parallel_tool_calls is False

    def test_with_allowed_tools(self) -> None:
        opts = ToolOptions(
            choice=ToolChoice.ANY,
            allowed=["get_weather", "search"],
            disable_parallel_tool_calls=True,
        )
        assert opts.choice == ToolChoice.ANY
        assert len(opts.allowed) == 2
        assert opts.disable_parallel_tool_calls is True


# ── Schema ───────────────────────────────────────────────────────


class TestSchema:
    def test_defaults(self) -> None:
        s = Schema()
        assert s.name == ""
        assert s.description == ""
        assert s.schema_ is None
        assert s.strict is None

    def test_with_values(self) -> None:
        s = Schema(
            name="my_schema",
            description="A test schema",
            schema_={"type": "object"},
            strict=True,
        )
        assert s.name == "my_schema"
        assert s.schema_ == {"type": "object"}
        assert s.strict is True


# ── CompleteOptions ──────────────────────────────────────────────


class TestCompleteOptions:
    def test_new_fields_default_none(self) -> None:
        opts = CompleteOptions()
        assert opts.verbosity is None
        assert opts.tool_options is None
        assert opts.structured_output is None

    def test_with_all_new_fields(self) -> None:
        opts = CompleteOptions(
            effort=Effort.MAX,
            verbosity=Verbosity.HIGH,
            tool_options=ToolOptions(disable_parallel_tool_calls=True),
            structured_output=Schema(name="test"),
        )
        assert opts.effort == Effort.MAX
        assert opts.verbosity == Verbosity.HIGH
        assert opts.tool_options.disable_parallel_tool_calls is True
        assert opts.structured_output.name == "test"

    def test_backward_compatible(self) -> None:
        """Old-style usage still works."""
        opts = CompleteOptions(
            effort=Effort.LOW,
            tools=None,
            tool_choice=ToolChoice.AUTO,
            response_schema={"type": "object"},
        )
        assert opts.effort == Effort.LOW
        assert opts.tool_choice == ToolChoice.AUTO


# ── Usage with cache tokens ──────────────────────────────────────


class TestUsage:
    def test_cache_fields_default_zero(self) -> None:
        u = Usage()
        assert u.cache_read_input_tokens == 0
        assert u.cache_creation_input_tokens == 0

    def test_cache_fields_set(self) -> None:
        u = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=20,
        )
        assert u.cache_read_input_tokens == 80
        assert u.cache_creation_input_tokens == 20

    def test_frozen(self) -> None:
        """Usage should be immutable."""
        u = Usage(input_tokens=10)
        try:
            u.input_tokens = 20  # type: ignore
            assert False, "Should have raised"
        except Exception:
            pass


# ── Accumulator with cache tokens ────────────────────────────────


class TestAccumulatorCacheTokens:
    def test_accumulates_cache_tokens(self) -> None:
        acc = Accumulator()
        acc.add(
            Completion(
                id="c1",
                model="m",
                message=Message(role=Role.ASSISTANT, content=[Content(text="hi")]),
                usage=Usage(
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_input_tokens=80,
                    cache_creation_input_tokens=20,
                ),
            )
        )

        result = acc.result
        assert result.usage is not None
        assert result.usage.cache_read_input_tokens == 80
        assert result.usage.cache_creation_input_tokens == 20

    def test_max_cache_tokens_across_chunks(self) -> None:
        acc = Accumulator()
        acc.add(
            Completion(
                id="c1",
                model="m",
                message=Message(role=Role.ASSISTANT, content=[Content(text="a")]),
                usage=Usage(
                    input_tokens=100,
                    output_tokens=20,
                    cache_read_input_tokens=50,
                    cache_creation_input_tokens=10,
                ),
            )
        )
        acc.add(
            Completion(
                id="c1",
                model="m",
                message=Message(role=Role.ASSISTANT, content=[Content(text="b")]),
                usage=Usage(
                    input_tokens=100,
                    output_tokens=40,
                    cache_read_input_tokens=80,
                    cache_creation_input_tokens=5,
                ),
            )
        )

        result = acc.result
        assert result.usage is not None
        assert result.usage.cache_read_input_tokens == 80
        assert result.usage.cache_creation_input_tokens == 10

    async def test_collect_preserves_cache_tokens(self) -> None:
        async def gen():
            yield Completion(
                id="c1",
                model="m",
                message=Message(role=Role.ASSISTANT, content=[Content(text="x")]),
                usage=Usage(
                    input_tokens=10,
                    output_tokens=5,
                    cache_read_input_tokens=7,
                    cache_creation_input_tokens=3,
                ),
            )

        result = await Accumulator.collect(gen())
        assert result.usage.cache_read_input_tokens == 7
