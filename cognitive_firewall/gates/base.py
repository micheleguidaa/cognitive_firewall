"""Gate base class: dispatch, timing, fail-mode, and injection pre-screen.

Each concrete gate implements ``_evaluate_heuristic`` (offline) and, from Phase 2,
``_evaluate_llm`` (real model). ``Gate.evaluate`` picks the path based on the
provider, times it, converts any exception into a fail-mode result, and applies
the prompt-injection clamp for input gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from .. import lexicons
from ..config import FirewallConfig
from ..providers.base import ProviderError
from ..types import GateLabel, GateResult, RiskCategory, Turn

# Gate-error -> risk score, by fail policy.
_FAIL_SCORE = {"closed": 1.0, "flag": 0.5, "open": 0.0}


@dataclass
class GateInput:
    """Everything a gate may need: the conversation and (for G3) the output."""

    turns: list[Turn]
    output: Optional[str] = None

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def last_user(self) -> str:
        users = self.user_turns
        return users[-1].content if users else ""

    @property
    def all_user_text(self) -> str:
        return "\n".join(t.content for t in self.user_turns)

    def as_messages(self) -> list[dict]:
        return [t.to_dict() for t in self.turns]


class Gate:
    gate_id: str = "G?"
    name: str = "gate"
    screens_injection: bool = False  # input gates (G1, G2) get the injection clamp

    def __init__(self, cfg: FirewallConfig):
        self.cfg = cfg

    # -- public entry ---------------------------------------------------------
    def evaluate(self, gi: GateInput, provider) -> GateResult:
        t0 = perf_counter()
        try:
            if getattr(provider, "supports_llm", False):
                res = self._evaluate_llm(gi, provider)
            else:
                res = self._evaluate_heuristic(gi)
        except ProviderError as e:
            res = self._fail(str(e))
        except Exception as e:  # noqa: BLE001 — a gate must never crash the firewall
            res = self._fail(f"{type(e).__name__}: {e}")
        res.latency_ms = (perf_counter() - t0) * 1000.0
        if self.screens_injection and res.error is None:
            res = self._injection_clamp(gi, res)
        return res

    # -- to be overridden -----------------------------------------------------
    def _evaluate_heuristic(self, gi: GateInput) -> GateResult:
        raise NotImplementedError

    def _evaluate_llm(self, gi: GateInput, provider) -> GateResult:
        # Implemented in Phase 2. Until then the heuristic path is used (the
        # offline provider reports supports_llm=False, so this is never hit there).
        raise NotImplementedError("LLM gate path is implemented in Phase 2")

    # -- helpers --------------------------------------------------------------
    def _fail(self, msg: str) -> GateResult:
        score = _FAIL_SCORE.get(self.cfg.fail_mode, 0.5)
        return GateResult(
            gate_id=self.gate_id,
            name=self.name,
            score=score,
            label=GateLabel.ERROR,
            rationale=f"gate error (fail_mode={self.cfg.fail_mode}): {msg}",
            error=msg,
            mode="error",
        )

    def _injection_clamp(self, gi: GateInput, res: GateResult) -> GateResult:
        """If the user text contains injection markers but the gate rated it
        benign, clamp to SUSPICIOUS — a successful injection must not force 0.0."""
        markers = lexicons.injection_evidence(gi.all_user_text)
        if markers and res.score < 0.5:
            res.score = 0.5
            if RiskCategory.INJECTION not in res.categories:
                res.categories.append(RiskCategory.INJECTION)
            res.evidence = list(res.evidence) + markers[:3]
            res.rationale = (res.rationale + " | prompt-injection markers detected; clamped to SUSPICIOUS.").strip()
            if res.label in (GateLabel.SAFE, GateLabel.PLAUSIBLE):
                res.label = GateLabel.SUSPICIOUS if self.gate_id == "G1" else GateLabel.UNVERIFIABLE
        return res
