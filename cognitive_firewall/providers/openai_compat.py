"""OpenAI-compatible provider.

A single client (configurable ``base_url``) covers OpenAI **and** any
OpenAI-compatible endpoint — local vLLM on the H200, OpenRouter, Together, Groq,
Ollama — so open-source models (Qwen, Mistral, DeepSeek, Kimi, GPT-OSS, Nemotron)
work unchanged. The API key is read from the environment by the caller and passed
in; it is never written to disk or echoed.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional

from .base import Provider, ProviderError, extract_json

_JSON_ONLY = (
    "You must respond with ONLY a single valid JSON object. No prose, no markdown, "
    "no code fences. Begin your reply with '{' and end with '}'."
)


def probe_endpoint(base_url: str, timeout: float = 1.0) -> bool:
    """True if an OpenAI-compatible server answers at ``base_url`` (e.g. local vLLM)."""
    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return getattr(resp, "status", 200) < 500
    except Exception:  # noqa: BLE001 — any failure means "not reachable"
        return False


class OpenAICompatProvider(Provider):
    supports_llm = True

    def __init__(self, base_url: str, model: str, api_key: Optional[str], cfg=None):
        try:
            from openai import OpenAI
        except Exception as e:  # noqa: BLE001
            raise ProviderError(
                "the 'openai' package is required for the OpenAI-compatible backend "
                "(pip install openai)"
            ) from e
        self.base_url = base_url
        self.model = model
        self.cfg = cfg
        self.main_model = cfg.effective_main_model() if cfg is not None else model
        # vLLM/local endpoints accept any non-empty key; never log this value.
        self._client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    @property
    def mode_name(self) -> str:  # never includes the key
        return f"openai_compat:{self.model}"

    # -- generation -----------------------------------------------------------
    def chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        timeout = getattr(self.cfg, "request_timeout_s", 30.0)
        try:
            resp = self._client.chat.completions.create(
                model=self.main_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"chat failed: {type(e).__name__}: {e}") from e

    # -- gate calls (JSON) ----------------------------------------------------
    def chat_json(self, messages, *, schema_hint=None, max_tokens=512, temperature=0.0) -> dict:
        timeout = getattr(self.cfg, "gate_timeout_s", 20.0)

        # Attempt 1: strict JSON mode (rejected by some OSS endpoints).
        try:
            content = self._create(messages, max_tokens, temperature, timeout, json_mode=True)
            return extract_json(content)
        except ProviderError:
            pass  # got a reply but couldn't parse -> hardened retry
        except Exception:
            pass  # endpoint rejected json mode / transient -> hardened retry

        # Attempt 2: no JSON mode, hardened system reminder + tolerant parse.
        hardened = self._harden(messages)
        try:
            content = self._create(hardened, max_tokens, temperature, timeout, json_mode=False)
            return extract_json(content)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"chat_json failed: {type(e).__name__}: {e}") from e

    def _create(self, messages, max_tokens, temperature, timeout, *, json_mode: bool) -> str:
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    @staticmethod
    def _harden(messages: list[dict]) -> list[dict]:
        out = [dict(m) for m in messages]
        for m in out:
            if m.get("role") == "system":
                m["content"] = f"{m['content']}\n\n{_JSON_ONLY}"
                return out
        out.insert(0, {"role": "system", "content": _JSON_ONLY})
        return out

    # -- moderation (native only; None -> G3 judge fallback) ------------------
    def moderate(self, text: str) -> Optional[dict]:
        if not text:
            return None
        try:
            resp = self._client.moderations.create(model="omni-moderation-latest", input=text)
            scores = resp.results[0].category_scores
            data = scores if isinstance(scores, dict) else scores.model_dump()
            return {k: float(v) for k, v in data.items() if v is not None}
        except Exception:  # noqa: BLE001 — most non-OpenAI endpoints have no moderation route
            return None
