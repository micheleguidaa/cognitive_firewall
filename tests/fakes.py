"""Test doubles: a programmable LLM provider (no network)."""
from __future__ import annotations

from cognitive_firewall.providers.base import Provider, ProviderError


class FakeLLMProvider(Provider):
    """LLM provider whose gate responses are scripted.

    ``by_gate`` maps a gate id ("G1".."G4") to either a JSON-like dict to return
    or an Exception to raise. ``raise_on`` is a set of gate ids that should raise.
    ``captured`` collects every ``chat_json`` message list for inspection.
    """

    supports_llm = True
    mode_name = "fake"

    def __init__(
        self,
        by_gate=None,
        raise_on=None,
        default=None,
        chat_text="Benign simulated answer.",
        moderate_ret=None,
        captured=None,
    ):
        self.by_gate = by_gate or {}
        self.raise_on = set(raise_on or ())
        self.default = default
        self.chat_text = chat_text
        self.moderate_ret = moderate_ret
        self.captured = captured if captured is not None else []

    @staticmethod
    def _gate_of(messages) -> str | None:
        system = messages[0].get("content", "") if messages else ""
        for n in ("1", "2", "3", "4"):
            if f"Gate {n}" in system:
                return f"G{n}"
        return None

    def chat(self, messages, *, max_tokens=1024, temperature=0.7) -> str:
        return self.chat_text

    def chat_json(self, messages, *, schema_hint=None, max_tokens=512, temperature=0.0) -> dict:
        self.captured.append(messages)
        gid = self._gate_of(messages)
        if gid in self.raise_on:
            raise ProviderError(f"simulated failure for {gid}")
        target = self.by_gate.get(gid, self.default)
        if isinstance(target, Exception):
            raise target
        if target is not None:
            return target
        return {"label": "SAFE", "score": 0.0, "rationale": "ok", "evidence": [], "categories": ["none"]}

    def moderate(self, text: str):
        return self.moderate_ret
