"""Gate 4 — Conversation consistency / escalation audit.

Looks at the trajectory across user turns rather than any single turn: crescendo
(a benign->harmful gradient), many-shot priming, and drift toward harm. Scores
the conversation as a whole.
"""
from __future__ import annotations

from .. import lexicons
from ..types import GateLabel, GateResult, RiskCategory
from .base import Gate, GateInput

# Cap history scanned to bound work; an overflow is itself a (mild) signal.
_WINDOW = 20


class ConsistencyGate(Gate):
    gate_id = "G4"
    name = "Consistency"
    screens_injection = False

    def _evaluate_heuristic(self, gi: GateInput) -> GateResult:
        contents = [t.content for t in gi.user_turns]
        overflow = len(contents) > _WINDOW
        contents = contents[-_WINDOW:]
        n = len(contents)

        if n < 2:
            return GateResult(
                self.gate_id, self.name, 0.0, GateLabel.LOW,
                rationale="Single turn; no multi-turn trajectory to audit.",
                mode="heuristic",
            )

        scores = [lexicons.intent_score(c) for c in contents]
        rises = sum(1 for i in range(1, n) if scores[i] > scores[i - 1] + 1e-9)
        nondecreasing = all(scores[i] >= scores[i - 1] - 1e-9 for i in range(1, n))
        harmful_turns = sum(1 for s in scores if s >= 0.5)

        pattern = "none"
        score = 0.0
        if n >= 3 and nondecreasing and scores[-1] >= 0.5 and rises >= 1:
            pattern = "crescendo"
            score = min(1.0, scores[-1] + 0.2 * rises)
        elif harmful_turns >= 5:
            pattern = "many_shot"
            score = min(1.0, 0.6 + 0.1 * (harmful_turns - 5))
        elif harmful_turns >= 2:
            pattern = "drift"
            score = 0.4
        if overflow:
            pattern = pattern if pattern != "none" else "context_overflow"
            score = max(score, 0.3)

        if score >= 0.7:
            label = GateLabel.HIGH
        elif score >= 0.4:
            label = GateLabel.MEDIUM
        else:
            label = GateLabel.LOW

        traj = " -> ".join(f"{s:.1f}" for s in scores)
        rationale = f"pattern={pattern}; per-turn intent trajectory [{traj}]; {harmful_turns}/{n} harmful turns."
        categories = [RiskCategory.ESCALATION] if pattern not in ("none",) else []

        return GateResult(
            gate_id=self.gate_id,
            name=self.name,
            score=score,
            label=label,
            rationale=rationale,
            evidence=[f"pattern={pattern}", f"trajectory={traj}"],
            categories=categories,
            mode="heuristic",
        )
