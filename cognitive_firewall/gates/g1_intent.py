"""Gate 1 — Intent analysis.

Classifies the TRUE operational objective of the request, independent of framing.
Educational / research / fiction wrappers do NOT lower the score; if anything they
are noted as framing applied over a harmful core.
"""
from __future__ import annotations

from .. import prompts
from ..types import GateLabel
from .base import Gate


class IntentGate(Gate):
    gate_id = "G1"
    name = "Intent"

    system_prompt = prompts.G1_INTENT_SYSTEM
    analyze_kind = "USER_REQUEST"
    _LABELS = {
        "SAFE": (GateLabel.SAFE, 0.0),
        "SUSPICIOUS": (GateLabel.SUSPICIOUS, 0.5),
        "UNSAFE": (GateLabel.UNSAFE, 1.0),
    }
    _BANDS = ((0.75, GateLabel.UNSAFE), (0.34, GateLabel.SUSPICIOUS), (0.0, GateLabel.SAFE))
