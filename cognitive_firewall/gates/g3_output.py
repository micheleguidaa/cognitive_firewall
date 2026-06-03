"""Gate 3 — Output risk analysis.

Judges the GENERATED output for actionable harm. In offline mode it combines
harm-category detection with "actionability" signals (numbered steps, code
fences, quantities). A refusal scores LOW; harmful + actionable scores HIGH.
"""
from __future__ import annotations

from .. import prompts
from ..types import GateLabel, GateResult, RiskCategory
from .base import Gate, GateInput

# Best-effort mapping of common moderation-endpoint categories to our taxonomy.
_MOD_MAP = [
    ("sexual/minors", RiskCategory.CSAM),
    ("sexual_minors", RiskCategory.CSAM),
    ("self-harm", RiskCategory.SELF_HARM),
    ("self_harm", RiskCategory.SELF_HARM),
    ("violence", RiskCategory.VIOLENCE),
    ("illicit", RiskCategory.FRAUD),
    ("hate", RiskCategory.MANIPULATION),
    ("harassment", RiskCategory.MANIPULATION),
]


class OutputGate(Gate):
    gate_id = "G3"
    name = "Output"

    system_prompt = prompts.G3_OUTPUT_SYSTEM
    analyze_kind = "ASSISTANT_OUTPUT"
    _LABELS = {
        "LOW": (GateLabel.LOW, 0.1),
        "MEDIUM": (GateLabel.MEDIUM, 0.5),
        "HIGH": (GateLabel.HIGH, 0.9),
    }
    _BANDS = ((0.7, GateLabel.HIGH), (0.4, GateLabel.MEDIUM), (0.0, GateLabel.LOW))

    def _llm_content(self, gi: GateInput) -> str:
        return gi.output or ""

    def _evaluate_llm(self, gi: GateInput, provider) -> GateResult:
        """Moderation-first: try the provider's moderation endpoint; if it has
        none (typical for OSS endpoints), fall back to the judge prompt."""
        out = gi.output or ""
        mod = None
        try:
            mod = provider.moderate(out)
        except Exception:  # noqa: BLE001
            mod = None
        if mod:
            score = max((float(v) for v in mod.values()), default=0.0)
            score = max(0.0, min(1.0, score))
            flagged = sorted(mod.items(), key=lambda kv: kv[1], reverse=True)[:3]
            cats = []
            for key, val in mod.items():
                if float(val) >= 0.5:
                    for needle, cat in _MOD_MAP:
                        if needle in key.lower() and cat not in cats:
                            cats.append(cat)
            return GateResult(
                gate_id=self.gate_id,
                name=self.name,
                score=score,
                label=self._score_to_label(score),
                rationale="Moderation model: " + ", ".join(f"{k}={float(v):.2f}" for k, v in flagged),
                evidence=[k for k, _ in flagged],
                categories=cats,
                mode="llm-moderation",
                raw=dict(mod),
            )
        return super()._evaluate_llm(gi, provider)
