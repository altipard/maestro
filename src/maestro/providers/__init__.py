# Auto-import provider subpackages so @provider decorators register.
# Each import triggers the decorator, populating the registry.
from . import anthropic as _anthropic  # noqa: F401
from . import google as _google  # noqa: F401
from . import ollama as _ollama  # noqa: F401
from . import openai as _openai  # noqa: F401
from . import openrouter as _openrouter  # noqa: F401
from .registry import create, detect_capability, list_providers, provider

# Optional providers — import errors are silenced so missing
# dependencies (e.g. chromadb) don't break the core package.
try:
    from . import chroma as _chroma  # noqa: F401
except ImportError:
    pass

__all__ = ["create", "detect_capability", "list_providers", "provider"]
