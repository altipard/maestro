"""Tests for the policy system."""

from __future__ import annotations

from maestro.config import Config
from maestro.policy import NoOpPolicy
from maestro.policy.policy import AccessDeniedError, Action, PolicyProvider, Resource


class TestNoOpPolicy:
    async def test_allows_everything(self) -> None:
        p = NoOpPolicy()
        # Should not raise
        await p.verify(Resource.MODEL, "gpt-4o", Action.ACCESS)
        await p.verify(Resource.MCP, "tool-x", Action.ACCESS, user="alice", email="a@b.com")

    def test_satisfies_protocol(self) -> None:
        assert isinstance(NoOpPolicy(), PolicyProvider)


class TestDenyPolicy:
    """Test a custom policy that denies certain models."""

    async def test_deny_raises(self) -> None:
        class DenyAllPolicy:
            async def verify(self, resource, resource_id, action, *, user="", email=""):
                raise AccessDeniedError(f"Denied: {resource_id}")

        p = DenyAllPolicy()
        try:
            await p.verify(Resource.MODEL, "secret-model", Action.ACCESS)
            assert False, "Should have raised"
        except AccessDeniedError as e:
            assert "secret-model" in str(e)

    async def test_selective_deny(self) -> None:
        blocked = {"internal-model"}

        class SelectivePolicy:
            async def verify(self, resource, resource_id, action, *, user="", email=""):
                if resource_id in blocked:
                    raise AccessDeniedError(f"Access denied: {resource_id}")

        p = SelectivePolicy()
        await p.verify(Resource.MODEL, "public-model", Action.ACCESS)

        try:
            await p.verify(Resource.MODEL, "internal-model", Action.ACCESS)
            assert False, "Should have raised"
        except AccessDeniedError:
            pass


class TestConfigPolicy:
    def test_default_noop(self) -> None:
        cfg = Config()
        assert isinstance(cfg.policy, NoOpPolicy)

    def test_custom_policy(self) -> None:
        cfg = Config()
        custom = NoOpPolicy()
        cfg.policy = custom
        assert cfg.policy is custom
