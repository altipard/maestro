"""Pydantic models for YAML configuration files.

Example YAML:
    providers:
      - type: openai
        url: https://api.openai.com/v1/
        token: ${OPENAI_API_KEY}
        models:
          gpt-4o:
            id: gpt-4o
          text-embedding-3-small:
            id: text-embedding-3-small
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Per-model configuration within a provider block."""

    id: str = ""
    type: str = ""
    name: str = ""
    description: str = ""
    limit: int | None = None


class ProviderConfig(BaseModel):
    """A provider block in the YAML config."""

    type: str
    url: str = ""
    token: str = ""
    limit: int | None = None
    models: dict[str, ModelConfig | None] = Field(default_factory=dict)


class ExtractorConfig(BaseModel):
    """Configuration for a document extractor service.

    Supported types:
      - text: Local text file reader (no external service)
      - tika: Apache Tika server
      - kreuzberg: Kreuzberg extraction API
      - mistral: Mistral OCR API
      - unstructured: Unstructured.io API
      - docling: IBM Docling API
      - azure: Azure Document Intelligence
      - grpc: Custom gRPC extractor service
    """

    type: str
    url: str = ""
    token: str = ""
    model: str = ""
    limit: int | None = None


class SegmenterConfig(BaseModel):
    """Configuration for a text segmentation service.

    Supported types:
      - text: Local text/markdown-aware splitter
      - jina: Jina Segment API
      - kreuzberg: Kreuzberg chunking API
      - unstructured: Unstructured.io chunking API
      - grpc: Custom gRPC segmenter service
    """

    type: str
    url: str = ""
    token: str = ""
    segment_length: int = 1000
    segment_overlap: int = 0
    limit: int | None = None


class AuthConfig(BaseModel):
    """Authentication configuration.

    Supports Bearer token validation. If tokens list is empty, auth is disabled.
    """

    tokens: list[str] = Field(default_factory=list)


class ScraperConfig(BaseModel):
    """Configuration for a web scraper service.

    Supported types:
      - fetch: Local HTTP scraper (no external service)
    """

    type: str
    url: str = ""
    token: str = ""


class ConfigFile(BaseModel):
    """Top-level YAML config structure."""

    address: str = ":8080"
    auth: AuthConfig = Field(default_factory=AuthConfig)
    providers: list[ProviderConfig] = Field(default_factory=list)
    extractors: dict[str, ExtractorConfig] = Field(default_factory=dict)
    segmenters: dict[str, SegmenterConfig] = Field(default_factory=dict)
    scrapers: dict[str, ScraperConfig] = Field(default_factory=dict)
