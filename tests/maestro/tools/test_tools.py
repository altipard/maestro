"""Tests for tool provider interface and schema normalization."""

from __future__ import annotations

from typing import Any

from maestro.core.models import Tool
from maestro.tools import ToolProvider, normalize_schema


class TestNormalizeSchema:
    def test_empty_schema(self) -> None:
        result = normalize_schema({})
        assert result == {"type": "object", "properties": {}}

    def test_infer_object_from_properties(self) -> None:
        result = normalize_schema({"properties": {"name": {"type": "string"}}})
        assert result["type"] == "object"

    def test_infer_array_from_items(self) -> None:
        result = normalize_schema({"items": {"type": "number"}})
        assert result["type"] == "array"

    def test_default_to_object(self) -> None:
        result = normalize_schema({"description": "something"})
        assert result["type"] == "object"
        assert result["properties"] == {}

    def test_object_gets_empty_properties(self) -> None:
        result = normalize_schema({"type": "object"})
        assert result["properties"] == {}

    def test_array_gets_default_items(self) -> None:
        result = normalize_schema({"type": "array"})
        assert result["items"] == {"type": "string"}

    def test_complete_schema_unchanged(self) -> None:
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        result = normalize_schema(schema)
        assert result["type"] == "object"
        assert "city" in result["properties"]
        assert result["required"] == ["city"]


class TestToolProviderProtocol:
    def test_protocol_compliance(self) -> None:
        class MockToolProvider:
            async def tools(self) -> list[Tool]:
                return [Tool(name="test", description="test tool")]

            async def execute(
                self, name: str, parameters: dict[str, Any]
            ) -> Any:
                return {"result": "ok"}

        tp = MockToolProvider()
        assert isinstance(tp, ToolProvider)
