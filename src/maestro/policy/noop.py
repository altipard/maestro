"""No-op policy — always allows access."""

from __future__ import annotations

from .policy import Action, Resource


class NoOpPolicy:
    """Default policy that allows all access."""

    async def verify(
        self,
        resource: Resource,
        resource_id: str,
        action: Action,
        *,
        user: str = "",
        email: str = "",
    ) -> None:
        pass
