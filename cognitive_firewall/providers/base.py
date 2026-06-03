"""Provider interface + helpers shared by all backends.

A provider exposes three capabilities:
  * ``chat``       — free-form generation for the governed main LLM.
  * ``chat_json``  — a temperature-0 call that returns a parsed JSON dict (gates).
  * ``moderate``   — optional category->score dict, or None if unavailable.

``supports_llm`` tells a gate whether to take its LLM path or its heuristic path.
"""
from __future__ import annotations

import json
import re
from typing import Optional


class ProviderError(RuntimeError):
    """Raised on any provider failure (network, timeout, unparseable output)."""


_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```$")


def extract_json(text: str) -> dict:
    """Tolerantly parse a JSON object from a model response.

    Strips markdown fences, then falls back to first ``{`` … last ``}``. Raises
    :class:`ProviderError` if nothing parseable is found.
    """
    if not text or not text.strip():
        raise ProviderError("empty response")
    s = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text.strip())).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"unparseable JSON: {e}")
    raise ProviderError("no JSON object found in response")


class Provider:
    """Base provider. Subclasses override the methods they support."""

    supports_llm: bool = False
    mode_name: str = "base"

    def chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        raise NotImplementedError

    def chat_json(
        self,
        messages: list[dict],
        *,
        schema_hint: Optional[dict] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict:
        raise NotImplementedError

    def moderate(self, text: str) -> Optional[dict]:
        return None


def last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return str(messages[-1].get("content", "")) if messages else ""
