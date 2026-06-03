"""Gate 4 — Conversation consistency / escalation audit.

Looks at the trajectory across user turns rather than any single turn: crescendo
(a benign->harmful gradient), many-shot priming, and drift toward harm. Scores
the conversation as a whole.
"""
from __future__ import annotations

from .. import prompts
from ..types import GateLabel
from .base import Gate, GateInput

# Cap history scanned to bound work; an overflow is itself a (mild) signal.
_WINDOW = 20


class ConsistencyGate(Gate):
    gate_id = "G4"
    name = "Consistency"

    system_prompt = prompts.G4_CONSISTENCY_SYSTEM
    analyze_kind = "CONVERSATION"
    _LABELS = {
        "LOW": (GateLabel.LOW, 0.1),
        "MEDIUM": (GateLabel.MEDIUM, 0.5),
        "HIGH": (GateLabel.HIGH, 0.9),
    }
    _BANDS = ((0.7, GateLabel.HIGH), (0.4, GateLabel.MEDIUM), (0.0, GateLabel.LOW))

    def _llm_content(self, gi: GateInput) -> str:
        return "\n".join(f"{t.role}: {t.content}" for t in gi.turns[-_WINDOW:])
