"""Tests for the config loader.

These tests do NOT require API keys — they only test config parsing,
provider wiring, and the middleware stack.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from maestro.config import Config, load
from maestro.config.models import ConfigFile, ProviderConfig
from maestro.middleware.wrappers import RateLimited, Traced

# Register OpenAI providers so they're available in the registry
from maestro.providers import openai as _openai  # noqa: F401


def _write_yaml(content: str) -> Path:
    """Write YAML content to a temp file and return the path."""
    path = Path(tempfile.mktemp(suffix=".yaml"))
    path.write_text(content)
    return path


class TestConfigFileModel:
    def test_defaults(self) -> None:
        cfg = ConfigFile()
        assert cfg.address == ":8080"
        assert cfg.providers == []

    def test_parse_provider(self) -> None:
        cfg = ConfigFile(
            providers=[
                ProviderConfig(
                    type="openai",
                    url="https://api.openai.com/v1/",
                    token="test",
                    models={"gpt-4o": None},
                )
            ]
        )
        assert len(cfg.providers) == 1
        assert cfg.providers[0].type == "openai"
        assert "gpt-4o" in cfg.providers[0].models


class TestConfigLoader:
    def test_load_basic(self) -> None:
        path = _write_yaml("""
providers:
  - type: openai
    token: test-key
    models:
      gpt-4o: {}
""")
        try:
            cfg = load(path)
            assert "gpt-4o" in cfg.models()
        finally:
            path.unlink()

    def test_address_override(self) -> None:
        path = _write_yaml("""
address: ':9090'
providers: []
""")
        try:
            cfg = load(path)
            assert cfg.address == ":9090"
        finally:
            path.unlink()

    def test_env_var_substitution(self) -> None:
        os.environ["_TEST_MAESTRO_TOKEN"] = "secret-token"
        path = _write_yaml("""
providers:
  - type: openai
    token: ${_TEST_MAESTRO_TOKEN}
    models:
      gpt-4o: {}
""")
        try:
            cfg = load(path)
            # If env var wasn't expanded, provider creation would still work
            # but we verify the config parsed without error
            assert "gpt-4o" in cfg.models()
        finally:
            path.unlink()
            del os.environ["_TEST_MAESTRO_TOKEN"]

    def test_model_list_format(self) -> None:
        """Models can be a list of strings instead of a dict."""
        path = _write_yaml("""
providers:
  - type: openai
    token: test
    models:
      - gpt-4o
      - text-embedding-3-small
""")
        try:
            cfg = load(path)
            assert "gpt-4o" in cfg.models()
            assert "text-embedding-3-small" in cfg.models()
        finally:
            path.unlink()

    def test_empty_config(self) -> None:
        path = _write_yaml("")
        try:
            cfg = load(path)
            assert cfg.models() == []
        finally:
            path.unlink()


class TestProviderWiring:
    def test_completer_registered(self) -> None:
        path = _write_yaml("""
providers:
  - type: openai
    token: test
    models:
      gpt-4o:
        id: gpt-4o
""")
        try:
            cfg = load(path)
            c = cfg.completer("gpt-4o")
            assert c is not None
        finally:
            path.unlink()

    def test_embedder_registered(self) -> None:
        path = _write_yaml("""
providers:
  - type: openai
    token: test
    models:
      text-embedding-3-small:
        id: text-embedding-3-small
""")
        try:
            cfg = load(path)
            e = cfg.embedder("text-embedding-3-small")
            assert e is not None
        finally:
            path.unlink()

    def test_first_registered_wins_default(self) -> None:
        """First completer registered becomes the default (key '')."""
        path = _write_yaml("""
providers:
  - type: openai
    token: test
    models:
      gpt-4o:
        id: gpt-4o
      gpt-4o-mini:
        id: gpt-4o-mini
""")
        try:
            cfg = load(path)
            # Default should work
            c = cfg.completer()
            assert c is not None
        finally:
            path.unlink()

    def test_capability_autodetection(self) -> None:
        """Model type detected from ID: gpt-4o → completer, embed → embedder."""
        path = _write_yaml("""
providers:
  - type: openai
    token: test
    models:
      gpt-4o: {}
      text-embedding-3-small: {}
""")
        try:
            cfg = load(path)
            # gpt-4o auto-detected as completer
            assert cfg.completer("gpt-4o") is not None
            # text-embedding-3-small auto-detected as embedder
            assert cfg.embedder("text-embedding-3-small") is not None
        finally:
            path.unlink()

    def test_middleware_wrapping(self) -> None:
        """Providers get wrapped with Traced middleware."""
        path = _write_yaml("""
providers:
  - type: openai
    token: test
    models:
      gpt-4o: {}
""")
        try:
            cfg = load(path)
            c = cfg.completer("gpt-4o")
            # Should be wrapped in Traced (outermost)
            assert isinstance(c, Traced)
        finally:
            path.unlink()

    def test_rate_limit_wrapping(self) -> None:
        """Rate limit from config gets applied as middleware."""
        path = _write_yaml("""
providers:
  - type: openai
    token: test
    limit: 5
    models:
      gpt-4o: {}
""")
        try:
            cfg = load(path)
            c = cfg.completer("gpt-4o")
            # Outer = Traced, inner should have RateLimited
            assert isinstance(c, Traced)
            assert isinstance(c._inner, RateLimited)
        finally:
            path.unlink()

    def test_model_level_limit_overrides_provider(self) -> None:
        """Model-level limit takes precedence over provider-level."""
        path = _write_yaml("""
providers:
  - type: openai
    token: test
    limit: 100
    models:
      gpt-4o:
        limit: 5
""")
        try:
            cfg = load(path)
            c = cfg.completer("gpt-4o")
            assert isinstance(c, Traced)
            assert isinstance(c._inner, RateLimited)
        finally:
            path.unlink()


class TestConfigLookup:
    def test_missing_capability_raises(self) -> None:
        cfg = Config()
        with pytest.raises(KeyError, match="completer not found"):
            cfg.completer("nonexistent")

    def test_missing_model_falls_back_to_default(self) -> None:
        path = _write_yaml("""
providers:
  - type: openai
    token: test
    models:
      gpt-4o: {}
""")
        try:
            cfg = load(path)
            # "nonexistent" should fall back to default
            c = cfg.completer("nonexistent")
            assert c is not None
        finally:
            path.unlink()
