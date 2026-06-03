"""Baselines the firewall is compared against.

Committed baseline: the **raw model** (no firewall). The harness derives it for
free from the ungoverned output the firewall produces in its dry-run scoring pass
(the firewall passes the output through in dry-run), so no extra generation is
needed.

``GuardBaseline`` is the pluggable interface for single-model guardrails
(Llama Guard 3, ShieldGemma, WildGuard, OpenAI Moderation) — wired on in Phase 5.
"""
from __future__ import annotations

from typing import Optional


class RawModelBaseline:
    """The ungoverned main LLM. Provided for explicit use; the offline harness
    reads the same output from the firewall's dry-run pass."""

    name = "raw_model"

    def __init__(self, provider):
        self.provider = provider

    def respond(self, sample) -> str:
        try:
            return self.provider.chat(sample.messages())
        except Exception as e:  # noqa: BLE001
            return f"[error: {e}]"


class GuardBaseline:
    """Interface for an input/output guard classifier (enable in Phase 5).

    Implementations return a risk score in [0, 1] for a (prompt, response) pair so
    the same ASR / over-refusal metrics apply. Examples: Llama-Guard-3-8B served on
    the H200, ShieldGemma, WildGuard, or the OpenAI Moderation endpoint.
    """

    name = "guard"

    def score(self, sample, response: Optional[str]) -> float:
        raise NotImplementedError(
            "GuardBaseline is wired on in Phase 5 (Llama Guard 3 / ShieldGemma / OpenAI Moderation)."
        )
