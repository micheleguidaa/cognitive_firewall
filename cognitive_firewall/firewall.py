"""The Cognitive Firewall orchestrator.

Pipeline per request/turn:
  1. input guard (empty -> ALLOW)
  2. G1 Intent ‖ G2 Context  (pre-generation, concurrent)
  3. R_pre -> early BLOCK before the main LLM runs (unless dry_run)
  4. main LLM generation
  5. G3 Output ‖ G4 Consistency (post-generation, concurrent)
  6. Gate 5 composite -> decision band + veto -> enforcement
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Callable, Optional, Union

from .config import FirewallConfig, make_provider
from .gates import ConsistencyGate, ContextGate, GateInput, IntentGate, OutputGate
from .scorer import score_final, score_pregen
from .types import Decision, FirewallResult, GateLabel, Turn

Messages = Union[str, list]


def _to_turns(messages: Messages) -> list[Turn]:
    if isinstance(messages, str):
        return [Turn(role="user", content=messages)]
    turns: list[Turn] = []
    for m in messages:
        if isinstance(m, Turn):
            turns.append(m)
        elif isinstance(m, dict):
            turns.append(Turn.from_dict(m))
        else:
            raise TypeError(f"unsupported message type: {type(m)!r}")
    return turns


def _parallel(tasks: list[Callable]):
    """Run callables concurrently, preserving order. Each gate already catches
    its own exceptions, so results never raise here."""
    if len(tasks) == 1:
        return [tasks[0]()]
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = [ex.submit(t) for t in tasks]
        return [f.result() for f in futures]


class CognitiveFirewall:
    def __init__(self, cfg: Optional[FirewallConfig] = None, provider=None, main_provider=None):
        self.cfg = cfg or FirewallConfig.from_env()
        # gate/judge provider (the oversight) vs the governed main-LLM provider.
        # They can differ, e.g. strong gates on an API + a jailbreakable local main.
        self.provider = provider if provider is not None else make_provider(self.cfg)
        self.main_provider = main_provider if main_provider is not None else self.provider
        self.mode_label = (
            self.provider.mode_name
            if self.main_provider is self.provider
            else f"{self.provider.mode_name}+main={self.main_provider.mode_name}"
        )
        self.g1 = IntentGate(self.cfg)
        self.g2 = ContextGate(self.cfg)
        self.g3 = OutputGate(self.cfg)
        self.g4 = ConsistencyGate(self.cfg)

    # -- public API -----------------------------------------------------------
    def evaluate(self, messages: Messages) -> FirewallResult:
        t0 = perf_counter()
        turns = _to_turns(messages)
        gi = GateInput(turns=turns)

        if not gi.last_user.strip():
            return self._trivial_allow(turns, t0)

        # Pre-generation: G1 ‖ G2
        g1r, g2r = _parallel([
            lambda: self.g1.evaluate(gi, self.provider),
            lambda: self.g2.evaluate(gi, self.provider),
        ])
        pregen = score_pregen(g1r, g2r, self.cfg)

        if pregen.early_block and not self.cfg.dry_run:
            return self._early_block(turns, [g1r, g2r], pregen, t0)

        # Generation
        main_output = self._generate(gi)
        gi_out = GateInput(turns=turns, output=main_output)

        # Post-generation: G3 ‖ G4
        g3r, g4r = _parallel([
            lambda: self.g3.evaluate(gi_out, self.provider),
            lambda: self.g4.evaluate(gi_out, self.provider),
        ])
        gates = [g1r, g2r, g3r, g4r]
        verdict = score_final(gates, self.cfg)
        return self._finalize(turns, gates, verdict, pregen, main_output, t0)

    __call__ = evaluate

    # -- internals ------------------------------------------------------------
    def _generate(self, gi: GateInput) -> str:
        try:
            return self.main_provider.chat(
                gi.as_messages(),
                max_tokens=self.cfg.main_max_tokens,
                temperature=self.cfg.main_temperature,
            )
        except Exception as e:  # noqa: BLE001
            return f"[main-LLM generation error: {type(e).__name__}: {e}]"

    def _early_block(self, turns, gates, pregen, t0) -> FirewallResult:
        reasons = list(pregen.reasons)
        veto_fired = gates[0].label == GateLabel.UNSAFE
        return FirewallResult(
            decision=Decision.BLOCK,
            would_be_decision=Decision.BLOCK,
            risk_score=pregen.pregen_score,
            pregen_score=pregen.pregen_score,
            early_blocked=True,
            veto_fired=veto_fired,
            veto_reasons=reasons,
            gates=gates,
            main_output=None,
            returned_output=self.cfg.safe_refusal,
            refusal_reason="; ".join(reasons),
            dry_run=self.cfg.dry_run,
            provider_mode=self.mode_label,
            total_latency_ms=(perf_counter() - t0) * 1000.0,
        )

    def _finalize(self, turns, gates, verdict, pregen, main_output, t0) -> FirewallResult:
        decision = verdict.decision
        # Fail policy: a gate error must never silently ALLOW.
        errored = any(g.error for g in gates)
        fail_floor: Optional[str] = None
        if errored and not self.cfg.dry_run:
            if self.cfg.fail_mode == "closed":
                decision = Decision.BLOCK
                fail_floor = "gate failure under fail_mode=closed"
            elif self.cfg.fail_mode == "flag" and decision == Decision.ALLOW:
                decision = Decision.FLAG

        refusal_reason: Optional[str] = None
        if decision == Decision.BLOCK:
            returned = self.cfg.safe_refusal
            refusal_reason = (
                fail_floor
                or "; ".join(verdict.veto_reasons)
                or f"composite risk {verdict.risk_score:.2f} >= {self.cfg.block_threshold:.2f}"
            )
        else:
            returned = main_output

        return FirewallResult(
            decision=decision,
            would_be_decision=verdict.would_be_decision,
            risk_score=verdict.risk_score,
            pregen_score=pregen.pregen_score,
            early_blocked=False,
            veto_fired=verdict.veto_fired,
            veto_reasons=verdict.veto_reasons,
            gates=gates,
            main_output=main_output,
            returned_output=returned,
            refusal_reason=refusal_reason,
            dry_run=self.cfg.dry_run,
            provider_mode=self.mode_label,
            total_latency_ms=(perf_counter() - t0) * 1000.0,
        )

    def _trivial_allow(self, turns, t0) -> FirewallResult:
        return FirewallResult(
            decision=Decision.ALLOW,
            would_be_decision=Decision.ALLOW,
            risk_score=0.0,
            pregen_score=None,
            early_blocked=False,
            veto_fired=False,
            veto_reasons=[],
            gates=[],
            main_output="",
            returned_output="",
            refusal_reason=None,
            dry_run=self.cfg.dry_run,
            provider_mode=self.mode_label,
            total_latency_ms=(perf_counter() - t0) * 1000.0,
        )
