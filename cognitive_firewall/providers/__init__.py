"""LLM provider abstraction for the Cognitive Firewall."""
from __future__ import annotations

from .base import Provider, ProviderError, extract_json

__all__ = ["Provider", "ProviderError", "extract_json"]
