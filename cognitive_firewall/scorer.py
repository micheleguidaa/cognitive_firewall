"""Gate 5 — composite risk scorer.

Pure scoring logic: no I/O, no LLM. Consumes :class:`GateResult` objects and a
:class:`FirewallConfig`, and produces verdicts. Kept separate from the firewall
orchestration so the scoring math can be unit-tested and swept in isolation.

Scoring model (defaults; all configurable):

    R_final = 0.40*G1 + 0.20*G2 + 0.20*G3 + 0.20*G4        # decision bands
    R_pre   = (0.40*G1 + 0.20*G2) / (0.40 + 0.20)          # renormalized pre-gen

    R_final >= 0.60 -> BLOCK ; >= 0.30 -> FLAG ; else ALLOW
    R_pre   >= 0.65 (or G1 == UNSAFE) -> early BLOCK before generation
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import FirewallConfig
from .types import Decision, GateLabel, GateResult


@dataclass
class PregenVerdict:
    pregen_score: float          # R_pre in [0, 1]
    early_block: bool            # would block before generation
    reasons: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    decision: Decision           # enforced (honors dry_run)
    would_be_decision: Decision  # canonical decision incl. veto, ignoring dry_run
    risk_score: float            # R_final
    veto_fired: bool
    veto_reasons: list[str] = field(default_factory=list)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def composite_risk(scores: dict, weights: dict) -> float:
    """Weighted sum of per-gate risk. Renormalizes defensively if weights don't
    sum to 1.0 (e.g. during a leave-one-gate-out ablation)."""
    total_w = sum(weights.values())
    if total_w <= 0:
        return 0.0
    r = sum(weights.get(gid, 0.0) * float(scores.get(gid, 0.0)) for gid in weights)
    if abs(total_w - 1.0) > 1e-9:
        r = r / total_w
    return _clamp(r)


def renormalized_pregen_risk(g1: float, g2: float, weights: dict) -> float:
    """R_pre — the pre-generation risk from G1+G2 only, renormalized onto [0, 1].

    Renormalization (dividing by w1+w2) is REQUIRED: without it the maximum
    achievable pre-gen score is only w1+w2 = 0.60 and could never exceed a 0.60+
    threshold, making an early block impossible.
    """
    w1 = weights.get("G1", 0.0)
    w2 = weights.get("G2", 0.0)
    denom = w1 + w2
    if denom <= 0:
        return 0.0
    return _clamp((w1 * float(g1) + w2 * float(g2)) / denom)


def decide_band(risk: float, cfg: FirewallConfig) -> Decision:
    if risk >= cfg.block_threshold:
        return Decision.BLOCK
    if risk >= cfg.flag_threshold:
        return Decision.FLAG
    return Decision.ALLOW


def _eval_veto(by_id: dict, cfg: FirewallConfig) -> tuple[bool, list[str]]:
    """Category-gated, escalate-only veto. Returns (fired, reasons)."""
    reasons: list[str] = []
    g1 = by_id.get("G1")
    g2 = by_id.get("G2")
    g3 = by_id.get("G3")

    if g1 is not None and g1.score >= cfg.veto_g1_score:
        crit = set(g1.categories) & set(cfg.veto_g1_categories)
        if crit:
            cats = ", ".join(sorted(c.value for c in crit))
            reasons.append(f"G1 intent UNSAFE in critical category [{cats}]")

    if g3 is not None and g3.score >= cfg.veto_g3_score:
        reasons.append(f"G3 output risk {g3.score:.2f} >= {cfg.veto_g3_score:.2f}")

    if (
        g2 is not None
        and g2.score >= cfg.veto_g2_score
        and g1 is not None
        and g1.score >= cfg.veto_g2_with_g1
    ):
        reasons.append(
            f"G2 manipulative context (score {g2.score:.2f}) with non-benign intent (G1={g1.score:.2f})"
        )

    return (len(reasons) > 0, reasons)


def score_pregen(g1: GateResult, g2: GateResult, cfg: FirewallConfig) -> PregenVerdict:
    """Pre-generation containment decision from G1 + G2."""
    r_pre = renormalized_pregen_risk(g1.score, g2.score, cfg.weights)
    reasons: list[str] = []
    early = False

    if r_pre >= cfg.pregen_block_threshold:
        early = True
        reasons.append(
            f"pre-generation risk R_pre={r_pre:.3f} >= {cfg.pregen_block_threshold:.2f}"
        )

    if cfg.pregen_veto_g1_unsafe and (g1.label == GateLabel.UNSAFE or g1.score >= 1.0):
        early = True
        reasons.append("G1 intent classified UNSAFE (pre-generation veto)")

    return PregenVerdict(pregen_score=r_pre, early_block=early, reasons=reasons)


def score_final(gates: list[GateResult], cfg: FirewallConfig) -> Verdict:
    """Aggregate all available gates into the final enforcement verdict."""
    by_id = {g.gate_id: g for g in gates}
    scores = {gid: g.score for gid, g in by_id.items()}

    r = composite_risk(scores, cfg.weights)
    base = decide_band(r, cfg)

    veto_fired, reasons = _eval_veto(by_id, cfg) if cfg.enable_veto else (False, [])
    # Veto is escalate-only: it can raise to BLOCK but never lower the decision.
    would_be = Decision.BLOCK if veto_fired else base

    enforced = Decision.ALLOW if cfg.dry_run else would_be
    return Verdict(
        decision=enforced,
        would_be_decision=would_be,
        risk_score=r,
        veto_fired=veto_fired,
        veto_reasons=reasons,
    )
