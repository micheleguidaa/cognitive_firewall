"""Phase 0 — scoring engine unit tests (no LLM, no network)."""
from __future__ import annotations

import pytest

from cognitive_firewall.config import FirewallConfig
from cognitive_firewall.scorer import (
    composite_risk,
    decide_band,
    renormalized_pregen_risk,
    score_final,
    score_pregen,
)
from cognitive_firewall.types import Decision, GateLabel, GateResult, RiskCategory


def gr(gate_id, score, label, categories=()):
    return GateResult(
        gate_id=gate_id,
        name=gate_id,
        score=score,
        label=label,
        categories=list(categories),
    )


# --- composite math ---------------------------------------------------------

def test_composite_matches_spec_example():
    # The user's worked example: G1=1.0, G2=0.5, G3=0.0, G4=0.5 -> R = 0.6
    cfg = FirewallConfig()
    scores = {"G1": 1.0, "G2": 0.5, "G3": 0.0, "G4": 0.5}
    assert composite_risk(scores, cfg.weights) == pytest.approx(0.6)


def test_composite_renormalizes_for_ablation():
    # Leave-one-gate-out: drop G1, remaining weights {G2,G3,G4}=0.6 should renormalize.
    weights = {"G2": 0.20, "G3": 0.20, "G4": 0.20}
    # all 1.0 -> renormalized to 1.0, not 0.6
    assert composite_risk({"G2": 1.0, "G3": 1.0, "G4": 1.0}, weights) == pytest.approx(1.0)


# --- bands ------------------------------------------------------------------

@pytest.mark.parametrize(
    "risk,expected",
    [
        (0.60, Decision.BLOCK),   # inclusive lower bound for BLOCK
        (0.61, Decision.BLOCK),
        (0.59, Decision.FLAG),
        (0.30, Decision.FLAG),    # inclusive lower bound for FLAG
        (0.29, Decision.ALLOW),
        (0.00, Decision.ALLOW),
    ],
)
def test_bands(risk, expected):
    assert decide_band(risk, FirewallConfig()) == expected


# --- pre-generation early block --------------------------------------------

@pytest.mark.parametrize(
    "g1,g2,early",
    [
        (1.0, 0.0, True),   # R_pre = 0.40/0.60 = 0.667 >= 0.65
        (0.5, 1.0, True),   # R_pre = (0.2+0.2)/0.6 = 0.667
        (0.5, 0.5, False),  # R_pre = 0.50
        (0.0, 1.0, False),  # R_pre = 0.333 (manipulation alone, benign intent)
    ],
)
def test_pregen_early_block(g1, g2, early):
    cfg = FirewallConfig()
    # use SUSPICIOUS label so the UNSAFE veto does not confound the R_pre check
    v = score_pregen(gr("G1", g1, GateLabel.SUSPICIOUS), gr("G2", g2, GateLabel.PLAUSIBLE), cfg)
    assert v.early_block is early


def test_pregen_renormalization_value():
    cfg = FirewallConfig()
    assert renormalized_pregen_risk(1.0, 0.0, cfg.weights) == pytest.approx(0.4 / 0.6)


def test_pregen_g1_unsafe_veto_fires_even_below_threshold():
    # Raise the threshold so R_pre alone would not trigger; the UNSAFE veto must.
    cfg = FirewallConfig(pregen_block_threshold=0.99)
    v = score_pregen(gr("G1", 1.0, GateLabel.UNSAFE), gr("G2", 0.0, GateLabel.PLAUSIBLE), cfg)
    assert v.early_block is True
    assert any("UNSAFE" in r for r in v.reasons)


# --- veto (post-band, escalate-only) ---------------------------------------

def test_g3_veto_escalates_to_block():
    cfg = FirewallConfig()
    gates = [
        gr("G1", 0.0, GateLabel.SAFE),
        gr("G2", 0.0, GateLabel.PLAUSIBLE),
        gr("G3", 0.95, GateLabel.HIGH),  # weighted sum only 0.19 -> ALLOW band
        gr("G4", 0.0, GateLabel.LOW),
    ]
    v = score_final(gates, cfg)
    assert v.risk_score == pytest.approx(0.19)
    assert v.veto_fired is True
    assert v.would_be_decision == Decision.BLOCK


def test_g1_category_gated_veto():
    cfg = FirewallConfig()
    # High G1 but a non-critical category should NOT veto on G1.
    gates = [
        gr("G1", 1.0, GateLabel.UNSAFE, categories=[RiskCategory.PRIVACY]),
        gr("G2", 0.0, GateLabel.PLAUSIBLE),
        gr("G3", 0.0, GateLabel.LOW),
        gr("G4", 0.0, GateLabel.LOW),
    ]
    v = score_final(gates, cfg)
    assert not any("critical category" in r for r in v.veto_reasons)
    # Critical category should veto.
    gates[0] = gr("G1", 1.0, GateLabel.UNSAFE, categories=[RiskCategory.WEAPONS])
    v2 = score_final(gates, cfg)
    assert v2.veto_fired is True


def test_veto_is_escalate_only():
    # A clean request must not be pushed up, and a blocking sum stays blocked.
    cfg = FirewallConfig()
    clean = [gr("G1", 0.0, GateLabel.SAFE), gr("G2", 0.0, GateLabel.PLAUSIBLE),
             gr("G3", 0.0, GateLabel.LOW), gr("G4", 0.0, GateLabel.LOW)]
    assert score_final(clean, cfg).would_be_decision == Decision.ALLOW


# --- dry run ----------------------------------------------------------------

def test_dry_run_records_would_be_but_enforces_allow():
    cfg = FirewallConfig(dry_run=True)
    gates = [
        gr("G1", 1.0, GateLabel.UNSAFE, categories=[RiskCategory.WEAPONS]),
        gr("G2", 0.5, GateLabel.UNVERIFIABLE),
        gr("G3", 0.0, GateLabel.LOW),
        gr("G4", 0.5, GateLabel.MEDIUM),
    ]
    v = score_final(gates, cfg)
    assert v.would_be_decision == Decision.BLOCK
    assert v.decision == Decision.ALLOW       # not enforced in dry-run
    assert v.veto_fired is True               # but still recorded
