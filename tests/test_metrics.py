"""Phase 3 — evaluation metrics (no network)."""
from __future__ import annotations

import pytest

from eval.metrics import (
    EvalRecord,
    asr_at_matched_overrefusal,
    asr_firewall,
    asr_raw,
    auroc_auprc,
    overrefusal_firewall,
    overrefusal_raw,
    prf1,
    recompute_risk,
    sweep,
)


def rec(label, risk, would_block, raw_harmful=False, model_refused=False, gate_scores=None):
    return EvalRecord(
        sample_id="x", label=label, category="c", source="s", attack_type="a",
        gate_scores=gate_scores or {"G1": risk, "G2": 0.0, "G3": 0.0, "G4": 0.0},
        risk=risk, would_block=would_block, raw_harmful=raw_harmful,
        model_refused=model_refused, latency_ms=1.0,
    )


@pytest.fixture
def records():
    return [
        rec("harmful", 0.9, would_block=True, raw_harmful=True),    # contained
        rec("harmful", 0.2, would_block=False, raw_harmful=True),   # attack succeeds
        rec("benign", 0.8, would_block=True),                       # over-refusal
        rec("benign", 0.1, would_block=False),                      # ok
    ]


def test_operating_point_metrics(records):
    assert asr_raw(records) == pytest.approx(1.0)        # both harmful complied ungoverned
    assert asr_firewall(records) == pytest.approx(0.5)   # one slipped through
    assert overrefusal_firewall(records) == pytest.approx(0.5)
    assert overrefusal_raw(records) == pytest.approx(0.0)


def test_auroc_separable(records):
    # harmful risks {0.9,0.2}, benign {0.8,0.1} -> not perfectly separable
    auroc, _ = auroc_auprc(records)
    assert 0.0 <= auroc <= 1.0
    # perfectly separable set -> AUROC 1.0
    perfect = [
        rec("harmful", 0.9, True, raw_harmful=True),
        rec("harmful", 0.8, True, raw_harmful=True),
        rec("benign", 0.1, False),
        rec("benign", 0.2, False),
    ]
    assert auroc_auprc(perfect)[0] == pytest.approx(1.0)


def test_sweep_monotonic_overrefusal(records):
    curve = sweep(records, thresholds=[0.0, 0.25, 0.5, 0.75, 1.0])
    overref = [p["overrefusal"] for p in curve]
    assert overref == sorted(overref, reverse=True)  # non-increasing in threshold
    assert curve[0]["overrefusal"] == pytest.approx(1.0)  # t=0: every benign blocked


def test_asr_at_matched_overrefusal(records):
    m = asr_at_matched_overrefusal(records, target_overref=0.5)
    assert m["feasible"] is True
    assert 0.0 <= m["asr"] <= 1.0
    assert m["overrefusal"] <= 0.5 + 1e-9


def test_prf1(records):
    m = prf1(records)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.5)


def test_recompute_risk_renormalizes_on_ablation():
    r = rec("harmful", 0.4, True, gate_scores={"G1": 1.0, "G2": 0.0, "G3": 0.0, "G4": 0.0})
    # full weights -> 0.4
    full = recompute_risk([r], {"G1": 0.4, "G2": 0.2, "G3": 0.2, "G4": 0.2})
    assert full[0].risk == pytest.approx(0.4)
    # drop G1 -> remaining gates all 0, renormalized -> 0.0
    dropped = recompute_risk([r], {"G2": 0.2, "G3": 0.2, "G4": 0.2})
    assert dropped[0].risk == pytest.approx(0.0)
