"""Gate 1 — Intent analysis.

Classifies the TRUE operational objective of the request, independent of framing.
Educational / research / fiction wrappers do NOT lower the score; if anything they
are noted as framing applied over a harmful core.
"""
from __future__ import annotations

from .. import lexicons
from ..types import GateLabel, GateResult
from .base import Gate, GateInput


class IntentGate(Gate):
    gate_id = "G1"
    name = "Intent"
    screens_injection = True

    def _evaluate_heuristic(self, gi: GateInput) -> GateResult:
        text = gi.last_user
        found = lexicons.detect_categories(text)
        categories = list(found.keys())
        evidence = sorted({t for terms in found.values() for t in terms})
        framing = lexicons.framing_markers(text)

        score = lexicons.intent_score(text)
        if score >= 1.0:
            label = GateLabel.UNSAFE
        elif score >= 0.5:
            label = GateLabel.SUSPICIOUS
        else:
            label = GateLabel.SAFE

        if not found:
            rationale = "No operational-harm objective detected in the request."
        else:
            cats = ", ".join(c.value for c in categories)
            rationale = f"Operational intent maps to: {cats}."
            if framing:
                rationale += (
                    f" Framing present ({', '.join(framing)}) — ignored per zero-trust;"
                    " judged on the underlying action objective."
                )

        return GateResult(
            gate_id=self.gate_id,
            name=self.name,
            score=score,
            label=label,
            rationale=rationale,
            evidence=evidence[:6],
            categories=categories,
            mode="heuristic",
        )
