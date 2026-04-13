"""Config loader: YAML → validated Pydantic models → wired provider graph.

- Pydantic validates the entire config structure
- @provider decorator for factory registration
- wrap() for middleware (rate limiting, tracing)
- Environment variable substitution via os.path.expandvars
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from maestro.middleware.wrappers import wrap
from maestro.policy import NoOpPolicy, PolicyProvider
from maestro.providers.registry import create, detect_capability

from .models import ConfigFile, ExtractorConfig, ModelConfig, ProviderConfig, SegmenterConfig


class Config:
    """Runtime configuration — the wired object graph.

    Lazy map initialization with first-registered-wins default (key "").
    """

    def __init__(self) -> None:
        self.address: str = ":8080"
        self.auth_tokens: list[str] = []
        self.policy: PolicyProvider = NoOpPolicy()
        self._providers: dict[str, dict[str, Any]] = {}

    def register(self, capability: str, model_id: str, instance: Any) -> None:
        """Register a provider instance by capability and model ID.

        First registered instance for a capability becomes the default (key "").
        """
        if capability not in self._providers:
            self._providers[capability] = {}

        cap_map = self._providers[capability]

        # First-registered-wins default
        if "" not in cap_map:
            cap_map[""] = instance

        cap_map[model_id] = instance

    def get(self, capability: str, model_id: str = "") -> Any:
        """Look up a provider by capability and model ID.

        Falls back to default ("") if model_id not found.
        Raises KeyError if not found.
        """
        cap_map = self._providers.get(capability, {})

        if model_id in cap_map:
            return cap_map[model_id]

        if "" in cap_map:
            return cap_map[""]

        raise KeyError(f"{capability} not found: {model_id or '(default)'}")

    @property
    def completer(self) -> _CapabilityAccessor:
        return _CapabilityAccessor(self, "completer")

    @property
    def embedder(self) -> _CapabilityAccessor:
        return _CapabilityAccessor(self, "embedder")

    @property
    def renderer(self) -> _CapabilityAccessor:
        return _CapabilityAccessor(self, "renderer")

    @property
    def synthesizer(self) -> _CapabilityAccessor:
        return _CapabilityAccessor(self, "synthesizer")

    @property
    def transcriber(self) -> _CapabilityAccessor:
        return _CapabilityAccessor(self, "transcriber")

    @property
    def reranker(self) -> _CapabilityAccessor:
        return _CapabilityAccessor(self, "reranker")

    @property
    def vectorstore(self) -> _CapabilityAccessor:
        return _CapabilityAccessor(self, "vectorstore")

    @property
    def extractor(self) -> _CapabilityAccessor:
        return _CapabilityAccessor(self, "extractor")

    @property
    def segmenter(self) -> _CapabilityAccessor:
        return _CapabilityAccessor(self, "segmenter")

    def models(self) -> list[str]:
        """Return all registered model IDs across all capabilities."""
        ids: set[str] = set()
        for cap_map in self._providers.values():
            ids.update(k for k in cap_map if k)
        return sorted(ids)


class _CapabilityAccessor:
    """Convenience accessor: config.completer("gpt-4o") or config.completer()."""

    def __init__(self, config: Config, capability: str) -> None:
        self._config = config
        self._capability = capability

    def __call__(self, model_id: str = "") -> Any:
        return self._config.get(self._capability, model_id)


# ── Loading ────────────────────────────────────────────────────────


def load(path: str | Path) -> Config:
    """Parse a YAML config file and build the wired Config.

    Steps:
    1. Read file, expand ${ENV_VAR} references
    2. Parse YAML into Pydantic ConfigFile
    3. For each provider + model: detect capability, create instance, wrap with middleware, register
    """
    config_file = _parse_file(path)
    config = Config()
    config.address = config_file.address
    config.auth_tokens = config_file.auth.tokens

    for provider_cfg in config_file.providers:
        _register_provider(config, provider_cfg)

    for name, ext_cfg in config_file.extractors.items():
        _register_extractor(config, name, ext_cfg)

    # Register multi-extractor as fallback (tries all extractors in order)
    _register_multi_extractor(config)

    for name, seg_cfg in config_file.segmenters.items():
        _register_segmenter(config, name, seg_cfg)

    return config


def _parse_file(path: str | Path) -> ConfigFile:
    """Read YAML with environment variable substitution."""
    raw = Path(path).read_text()

    # Expand ${ENV_VAR} and $ENV_VAR (matches Go's os.ExpandEnv)
    expanded = os.path.expandvars(raw)

    data = yaml.safe_load(expanded)
    if data is None:
        data = {}

    # Normalize models: support both list-of-strings and dict-of-objects
    for p in data.get("providers", []):
        models = p.get("models", {})
        if isinstance(models, list):
            # Convert ["gpt-4o", "gpt-4o-mini"] → {"gpt-4o": None, "gpt-4o-mini": None}
            p["models"] = {m: None for m in models}

    return ConfigFile(**data)


def _register_provider(config: Config, provider_cfg: ProviderConfig) -> None:
    """Register all models from a provider config block.

    1. Iterate models
    2. Detect capability (completer, embedder, etc.)
    3. Create provider instance via @provider registry
    4. Wrap with middleware (rate limit, tracing)
    5. Register in config
    """
    for alias, model_cfg in provider_cfg.models.items():
        model_cfg = model_cfg or ModelConfig()

        # Model ID defaults to the alias key
        model_id = model_cfg.id or alias

        # Auto-detect capability from model ID (same as Go's DetectModelType)
        capability = model_cfg.type or detect_capability(model_id)

        # Rate limit: model-level overrides provider-level
        rate_limit = model_cfg.limit if model_cfg.limit is not None else provider_cfg.limit

        # Create the raw provider via @provider registry
        instance = create(
            provider_cfg.type,
            capability,
            url=provider_cfg.url,
            token=provider_cfg.token,
            model=model_id,
        )

        # Wrap with middleware stack (same as Go's limiter + otel wrapping)
        instance = wrap(
            instance,
            vendor=provider_cfg.type,
            model=model_id,
            rate_limit=float(rate_limit) if rate_limit else 0,
        )

        # Register by alias (first-registered-wins for default)
        config.register(capability, alias, instance)


def _register_multi_extractor(config: Config) -> None:
    """Register a multi-extractor that tries all registered extractors as fallback."""
    from maestro.extractors.multi import MultiExtractor

    cap_map = config._providers.get("extractor", {})
    extractors = [v for k, v in cap_map.items() if k]  # skip default ""

    if extractors:
        multi = MultiExtractor(extractors=extractors)
        # Register as "multi" — only becomes default if no other default exists
        if "" not in cap_map:
            config.register("extractor", "", multi)


def _create_extractor(ext_cfg: ExtractorConfig) -> Any:
    """Create an extractor instance based on type."""
    match ext_cfg.type:
        case "text":
            from maestro.extractors.text import TextExtractor
            return TextExtractor()

        case "tika":
            from maestro.extractors.tika import TikaExtractor
            return TikaExtractor(url=ext_cfg.url)

        case "kreuzberg":
            from maestro.extractors.kreuzberg import KreuzbergExtractor
            return KreuzbergExtractor(url=ext_cfg.url, token=ext_cfg.token)

        case "mistral":
            from maestro.extractors.mistral import MistralExtractor
            return MistralExtractor(url=ext_cfg.url, token=ext_cfg.token, model=ext_cfg.model)

        case "unstructured":
            from maestro.extractors.unstructured import UnstructuredExtractor
            return UnstructuredExtractor(url=ext_cfg.url, token=ext_cfg.token)

        case "docling":
            from maestro.extractors.docling import DoclingExtractor
            return DoclingExtractor(url=ext_cfg.url, token=ext_cfg.token)

        case "azure":
            from maestro.extractors.azure import AzureExtractor
            return AzureExtractor(url=ext_cfg.url, token=ext_cfg.token, model=ext_cfg.model)

        case "grpc":
            from maestro.providers.grpc.extractor import Extractor as GrpcExtractor
            return GrpcExtractor(url=ext_cfg.url)

        case _:
            raise ValueError(f"Unsupported extractor type: {ext_cfg.type}")


def _register_extractor(config: Config, name: str, ext_cfg: ExtractorConfig) -> None:
    """Register an extractor from a named config block."""
    instance = _create_extractor(ext_cfg)

    instance = wrap(
        instance,
        vendor=ext_cfg.type,
        model=name,
        rate_limit=float(ext_cfg.limit) if ext_cfg.limit else 0,
    )

    config.register("extractor", name, instance)


def _create_segmenter(seg_cfg: SegmenterConfig) -> Any:
    """Create a segmenter instance based on type."""
    match seg_cfg.type:
        case "text":
            from maestro.segmenters.text import TextSegmenter
            return TextSegmenter(
                segment_length=seg_cfg.segment_length,
                segment_overlap=seg_cfg.segment_overlap,
            )

        case "jina":
            from maestro.segmenters.jina import JinaSegmenter
            return JinaSegmenter(
                url=seg_cfg.url,
                token=seg_cfg.token,
                segment_length=seg_cfg.segment_length,
            )

        case "kreuzberg":
            from maestro.segmenters.kreuzberg import KreuzbergSegmenter
            return KreuzbergSegmenter(
                url=seg_cfg.url,
                token=seg_cfg.token,
                segment_length=seg_cfg.segment_length,
                segment_overlap=seg_cfg.segment_overlap,
            )

        case "unstructured":
            from maestro.segmenters.unstructured import UnstructuredSegmenter
            return UnstructuredSegmenter(
                url=seg_cfg.url,
                token=seg_cfg.token,
                segment_length=seg_cfg.segment_length,
                segment_overlap=seg_cfg.segment_overlap,
            )

        case "grpc":
            from maestro.providers.grpc.segmenter import Segmenter as GrpcSegmenter
            return GrpcSegmenter(
                url=seg_cfg.url,
                segment_length=seg_cfg.segment_length,
                segment_overlap=seg_cfg.segment_overlap,
            )

        case _:
            raise ValueError(f"Unsupported segmenter type: {seg_cfg.type}")


def _register_segmenter(config: Config, name: str, seg_cfg: SegmenterConfig) -> None:
    """Register a segmenter from a named config block."""
    instance = _create_segmenter(seg_cfg)

    instance = wrap(
        instance,
        vendor=seg_cfg.type,
        model=name,
        rate_limit=float(seg_cfg.limit) if seg_cfg.limit else 0,
    )

    config.register("segmenter", name, instance)
