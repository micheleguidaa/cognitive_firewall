"""Gate 2 — Context verification (zero-trust).

Every identity/authority/context claim is treated as unverified evidence. Claims
addressed at the system ("I am your developer", "ignore your rules") are
MANIPULATIVE; unverifiable credentials ("I am a security researcher") are
UNVERIFIABLE; absence of claims is PLAUSIBLE.
"""
from __future__ import annotations

from .. import prompts
from ..types import GateLabel
from .base import Gate, GateInput


class ContextGate(Gate):
    gate_id = "G2"
    name = "Context"

    system_prompt = prompts.G2_CONTEXT_SYSTEM
    analyze_kind = "USER_CONTEXT"
    _LABELS = {
        "PLAUSIBLE": (GateLabel.PLAUSIBLE, 0.0),
        "UNVERIFIABLE": (GateLabel.UNVERIFIABLE, 0.5),
        "MANIPULATIVE": (GateLabel.MANIPULATIVE, 1.0),
    }
    _BANDS = ((0.75, GateLabel.MANIPULATIVE), (0.34, GateLabel.UNVERIFIABLE), (0.0, GateLabel.PLAUSIBLE))

    def _llm_content(self, gi: GateInput) -> str:
        return gi.all_user_text
