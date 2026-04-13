"""Tool provider interface and utilities.

Tool providers expose a list of tools and can execute them by name.
This is the building block for agentic chains.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from maestro.core.models import Tool


@runtime_checkable
class ToolProvider(Protocol):
    """Interface for tool providers."""

    async def tools(self) -> list[Tool]: ...

    async def execute(self, name: str, parameters: dict[str, Any]) -> Any: ...


def normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON schema for tool parameters."""
    if not schema:
        return {"type": "object", "properties": {}}

    if "type" not in schema:
        if "properties" in schema:
            schema["type"] = "object"
        elif "items" in schema:
            schema["type"] = "array"
        else:
            schema["type"] = "object"

    schema_type = schema.get("type")

    if schema_type == "object" and "properties" not in schema:
        schema["properties"] = {}
    elif schema_type == "array" and "items" not in schema:
        schema["items"] = {"type": "string"}

    return schema


__all__ = ["ToolProvider", "normalize_schema"]
