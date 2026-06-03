"""Gate 2 — Context verification (zero-trust).

Every identity/authority/context claim is treated as unverified evidence. Claims
addressed at the system ("I am your developer", "ignore your rules") are
MANIPULATIVE; unverifiable credentials ("I am a security researcher") are
UNVERIFIABLE; absence of claims is PLAUSIBLE.
"""
from __future__ import annotations

from .. import lexicons, prompts
from ..types import GateLabel, GateResult
from .base import Gate, GateInput

_HEUR_LABELS = {
    "MANIPULATIVE": GateLabel.MANIPULATIVE,
    "UNVERIFIABLE": GateLabel.UNVERIFIABLE,
    "PLAUSIBLE": GateLabel.PLAUSIBLE,
}


class ContextGate(Gate):
    gate_id = "G2"
    name = "Context"
    screens_injection = True

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

    def _evaluate_heuristic(self, gi: GateInput) -> GateResult:
        # Claims may appear in any user turn, so scan the whole user side.
        label_str, score, evidence, categories = lexicons.role_claims(gi.all_user_text)
        label = _HEUR_LABELS[label_str]

        if label is GateLabel.MANIPULATIVE:
            rationale = (
                "Authority/override claim directed at the system. Treated as a "
                "manipulation attempt — claims are evidence, not facts."
            )
        elif label is GateLabel.UNVERIFIABLE:
            rationale = (
                "Credential/role claim asserted but unverifiable. No elevated trust granted."
            )
        else:
            rationale = "No identity or authority claims to verify."

        return GateResult(
            gate_id=self.gate_id,
            name=self.name,
            score=score,
            label=label,
            rationale=rationale,
            evidence=list(evidence),
            categories=list(categories),
            mode="heuristic",
        )
