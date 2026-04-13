"""Tests for Ollama provider registration and URL normalization.

These tests do NOT require a running Ollama instance — they only
verify registry wiring and URL handling.
"""

from __future__ import annotations

from maestro.core.protocols import Completer as CompleterProtocol
from maestro.core.protocols import Embedder as EmbedderProtocol

# Register providers so they're available in the registry
from maestro.providers import ollama as _ollama  # noqa: F401
from maestro.providers.registry import create, list_providers


class TestOllamaRegistry:
    def test_completer_registered(self) -> None:
        providers = list_providers()
        assert "completer" in providers.get("ollama", [])

    def test_embedder_registered(self) -> None:
        providers = list_providers()
        assert "embedder" in providers.get("ollama", [])

    def test_completer_protocol(self) -> None:
        c = create("ollama", "completer", model="llama3.2")
        assert isinstance(c, CompleterProtocol)

    def test_embedder_protocol(self) -> None:
        e = create("ollama", "embedder", model="nomic-embed-text")
        assert isinstance(e, EmbedderProtocol)


class TestOllamaURLNormalization:
    def test_default_url(self) -> None:
        c = create("ollama", "completer", model="test")
        assert "localhost:11434" in c._url

    def test_custom_url(self) -> None:
        c = create("ollama", "completer", url="http://my-ollama:8000", model="test")
        assert "my-ollama:8000" in c._url
        assert "/v1" in c._url

    def test_trailing_slash_stripped(self) -> None:
        c = create("ollama", "completer", url="http://my-ollama:8000/", model="test")
        assert "//" not in c._url.replace("http://", "")

    def test_v1_suffix_not_doubled(self) -> None:
        c = create("ollama", "completer", url="http://my-ollama:8000/v1", model="test")
        assert c._url.count("/v1") == 1

    def test_token_defaults_to_ollama(self) -> None:
        c = create("ollama", "completer", model="test")
        assert c._token == "ollama"
