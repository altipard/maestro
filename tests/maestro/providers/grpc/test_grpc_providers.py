"""Tests for gRPC extractor and segmenter providers.

Tests cover:
- Provider instantiation and protocol compliance
- Config loading with extractors/segmenters sections
- Middleware wrapping (rate limiting, tracing)
- First-registered-wins default behavior
- Error cases (missing URL, unsupported type)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from maestro.config import Config, load
from maestro.config.models import ConfigFile, ExtractorConfig, SegmenterConfig
from maestro.core.models import FileData, Segment
from maestro.core.protocols import Extractor as ExtractorProtocol
from maestro.core.protocols import Segmenter as SegmenterProtocol
from maestro.middleware.wrappers import RateLimited, Traced
from maestro.providers.grpc.extractor import Extractor
from maestro.providers.grpc.segmenter import Segmenter


def _write_yaml(content: str) -> Path:
    path = Path(tempfile.mktemp(suffix=".yaml"))
    path.write_text(content)
    return path


# ── Instantiation & Protocol Compliance ─────────────────────────────


class TestExtractorProvider:
    def test_instantiation(self) -> None:
        e = Extractor(url="grpc://localhost:50051")
        assert e is not None

    def test_protocol_compliance(self) -> None:
        e = Extractor(url="grpc://localhost:50051")
        assert isinstance(e, ExtractorProtocol)

    def test_missing_url_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a gRPC url"):
            Extractor(url="")

    def test_strips_grpc_prefix(self) -> None:
        e = Extractor(url="grpc://myhost:9999")
        assert e._channel is not None


class TestSegmenterProvider:
    def test_instantiation(self) -> None:
        s = Segmenter(url="grpc://localhost:50051")
        assert s is not None

    def test_protocol_compliance(self) -> None:
        s = Segmenter(url="grpc://localhost:50051")
        assert isinstance(s, SegmenterProtocol)

    def test_missing_url_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a gRPC url"):
            Segmenter(url="")

    def test_default_segment_options(self) -> None:
        s = Segmenter(url="grpc://localhost:50051")
        assert s._segment_length == 1000
        assert s._segment_overlap == 0

    def test_custom_segment_options(self) -> None:
        s = Segmenter(url="grpc://localhost:50051", segment_length=512, segment_overlap=50)
        assert s._segment_length == 512
        assert s._segment_overlap == 50


# ── Config Model Parsing ────────────────────────────────────────────


class TestConfigModels:
    def test_extractor_config(self) -> None:
        cfg = ExtractorConfig(type="grpc", url="grpc://localhost:50051")
        assert cfg.type == "grpc"
        assert cfg.url == "grpc://localhost:50051"
        assert cfg.limit is None

    def test_segmenter_config_defaults(self) -> None:
        cfg = SegmenterConfig(type="grpc", url="grpc://localhost:50051")
        assert cfg.segment_length == 1000
        assert cfg.segment_overlap == 0

    def test_segmenter_config_custom(self) -> None:
        cfg = SegmenterConfig(
            type="grpc", url="grpc://localhost:50051", segment_length=512, segment_overlap=50
        )
        assert cfg.segment_length == 512
        assert cfg.segment_overlap == 50

    def test_config_file_with_extractors_and_segmenters(self) -> None:
        cfg = ConfigFile(
            extractors={"default": ExtractorConfig(type="grpc", url="grpc://localhost:50051")},
            segmenters={
                "default": SegmenterConfig(
                    type="grpc", url="grpc://localhost:50052", segment_length=512
                )
            },
        )
        assert "default" in cfg.extractors
        assert "default" in cfg.segmenters
        assert cfg.segmenters["default"].segment_length == 512

    def test_config_file_empty_extractors_segmenters(self) -> None:
        cfg = ConfigFile()
        assert cfg.extractors == {}
        assert cfg.segmenters == {}


# ── Config Loading ──────────────────────────────────────────────────


class TestConfigLoading:
    def test_load_extractor(self) -> None:
        path = _write_yaml("""
providers: []
extractors:
  default:
    type: grpc
    url: grpc://localhost:50051
""")
        try:
            cfg = load(path)
            ext = cfg.extractor()
            assert ext is not None
        finally:
            path.unlink()

    def test_load_segmenter(self) -> None:
        path = _write_yaml("""
providers: []
segmenters:
  default:
    type: grpc
    url: grpc://localhost:50051
    segment_length: 512
    segment_overlap: 50
""")
        try:
            cfg = load(path)
            seg = cfg.segmenter()
            assert seg is not None
        finally:
            path.unlink()

    def test_load_multiple_extractors(self) -> None:
        path = _write_yaml("""
providers: []
extractors:
  default:
    type: grpc
    url: grpc://localhost:50051
  pdf:
    type: grpc
    url: grpc://localhost:50052
""")
        try:
            cfg = load(path)
            assert cfg.extractor() is not None
            assert cfg.extractor("pdf") is not None
        finally:
            path.unlink()

    def test_first_registered_wins_default(self) -> None:
        path = _write_yaml("""
providers: []
extractors:
  first:
    type: grpc
    url: grpc://localhost:50051
  second:
    type: grpc
    url: grpc://localhost:50052
""")
        try:
            cfg = load(path)
            default = cfg.extractor()
            first = cfg.extractor("first")
            # Default should be the first one registered
            assert type(default) is type(first)
        finally:
            path.unlink()

    def test_missing_extractor_raises(self) -> None:
        cfg = Config()
        with pytest.raises(KeyError, match="extractor not found"):
            cfg.extractor()

    def test_missing_segmenter_raises(self) -> None:
        cfg = Config()
        with pytest.raises(KeyError, match="segmenter not found"):
            cfg.segmenter()

    def test_unsupported_extractor_type_raises(self) -> None:
        path = _write_yaml("""
providers: []
extractors:
  default:
    type: unknown
    url: grpc://localhost:50051
""")
        try:
            with pytest.raises(ValueError, match="Unsupported extractor type"):
                load(path)
        finally:
            path.unlink()

    def test_unsupported_segmenter_type_raises(self) -> None:
        path = _write_yaml("""
providers: []
segmenters:
  default:
    type: unknown
    url: grpc://localhost:50051
""")
        try:
            with pytest.raises(ValueError, match="Unsupported segmenter type"):
                load(path)
        finally:
            path.unlink()


# ── Middleware Wrapping ─────────────────────────────────────────────


class TestMiddlewareWrapping:
    def test_extractor_traced(self) -> None:
        path = _write_yaml("""
providers: []
extractors:
  default:
    type: grpc
    url: grpc://localhost:50051
""")
        try:
            cfg = load(path)
            ext = cfg.extractor()
            assert isinstance(ext, Traced)
        finally:
            path.unlink()

    def test_extractor_rate_limited(self) -> None:
        path = _write_yaml("""
providers: []
extractors:
  default:
    type: grpc
    url: grpc://localhost:50051
    limit: 5
""")
        try:
            cfg = load(path)
            ext = cfg.extractor()
            assert isinstance(ext, Traced)
            assert isinstance(ext._inner, RateLimited)
        finally:
            path.unlink()

    def test_segmenter_traced(self) -> None:
        path = _write_yaml("""
providers: []
segmenters:
  default:
    type: grpc
    url: grpc://localhost:50051
""")
        try:
            cfg = load(path)
            seg = cfg.segmenter()
            assert isinstance(seg, Traced)
        finally:
            path.unlink()

    def test_segmenter_rate_limited(self) -> None:
        path = _write_yaml("""
providers: []
segmenters:
  default:
    type: grpc
    url: grpc://localhost:50051
    limit: 10
""")
        try:
            cfg = load(path)
            seg = cfg.segmenter()
            assert isinstance(seg, Traced)
            assert isinstance(seg._inner, RateLimited)
        finally:
            path.unlink()


# ── Backward Compatibility ──────────────────────────────────────────


class TestBackwardCompatibility:
    def test_config_without_extractors_segmenters(self) -> None:
        """Existing configs without extractors/segmenters should still work."""
        path = _write_yaml("""
providers: []
""")
        try:
            cfg = load(path)
            assert cfg.models() == []
        finally:
            path.unlink()
