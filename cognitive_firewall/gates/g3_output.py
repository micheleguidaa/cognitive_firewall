"""Gate 3 — Output risk analysis.

Judges the GENERATED output for actionable harm. In offline mode it combines
harm-category detection with "actionability" signals (numbered steps, code
fences, quantities). A refusal scores LOW; harmful + actionable scores HIGH.
"""
from __future__ import annotations

from .. import lexicons
from ..types import GateLabel, GateResult
from .base import Gate, GateInput


class OutputGate(Gate):
    gate_id = "G3"
    name = "Output"
    screens_injection = False

    def _evaluate_heuristic(self, gi: GateInput) -> GateResult:
        out = gi.output or ""
        found = lexicons.detect_categories(out)
        categories = list(found.keys())
        harm_terms = sorted({t for terms in found.values() for t in terms})
        action = lexicons.actionability_evidence(out)
        refusals = lexicons.refusal_markers(out)

        if found and action:
            score, label = 0.9, GateLabel.HIGH
            rationale = "Output supplies actionable harmful content (operational steps/code for a harm category)."
            evidence = harm_terms[:3] + action[:3]
        elif refusals:
            score, label = 0.0, GateLabel.LOW
            rationale = "Output refuses or declines; no actionable harm."
            evidence = refusals[:3]
        elif found:
            score, label = 0.5, GateLabel.MEDIUM
            rationale = "Output references a harm category but is not clearly actionable."
            evidence = harm_terms[:4]
        elif action:
            score, label = 0.2, GateLabel.LOW
            rationale = "Output is procedural but covers no harmful category."
            evidence = action[:3]
        else:
            score, label = 0.0, GateLabel.LOW
            rationale = "Output is benign."
            evidence = []

        return GateResult(
            gate_id=self.gate_id,
            name=self.name,
            score=score,
            label=label,
            rationale=rationale,
            evidence=evidence,
            categories=categories,
            mode="heuristic",
        )
