"""Policy interface for access control."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable


class Resource(StrEnum):
    MODEL = "model"
    MCP = "mcp"


class Action(StrEnum):
    ACCESS = "access"


class AccessDeniedError(Exception):
    """Raised when policy denies access."""


@runtime_checkable
class PolicyProvider(Protocol):
    async def verify(
        self,
        resource: Resource,
        resource_id: str,
        action: Action,
        *,
        user: str = "",
        email: str = "",
    ) -> None:
        """Verify access. Raises AccessDeniedError if denied."""
        ...
