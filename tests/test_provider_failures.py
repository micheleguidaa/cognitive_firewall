"""Phase 2 — JSON parsing tolerance + fail-mode behavior (no network)."""
from __future__ import annotations

import pytest
from fakes import FakeLLMProvider

from cognitive_firewall.config import FirewallConfig
from cognitive_firewall.firewall import CognitiveFirewall
from cognitive_firewall.gates import IntentGate
from cognitive_firewall.gates.base import GateInput
from cognitive_firewall.providers.base import ProviderError, extract_json
from cognitive_firewall.types import Decision, GateLabel, Turn


def _gi(text):
    return GateInput(turns=[Turn(role="user", content=text)])


# --- tolerant JSON extraction -----------------------------------------------

def test_extract_json_plain():
    assert extract_json('{"label": "SAFE", "score": 0}')["label"] == "SAFE"


def test_extract_json_fenced():
    assert extract_json('```json\n{"score": 0.5}\n```')["score"] == 0.5


def test_extract_json_with_prose():
    assert extract_json('Sure! Here:\n{"a": 1} hope that helps')["a"] == 1


def test_extract_json_garbage_raises():
    with pytest.raises(ProviderError):
        extract_json("no json here at all")


# --- per-gate fail modes ----------------------------------------------------

@pytest.mark.parametrize(
    "fail_mode,expected_score",
    [("closed", 1.0), ("flag", 0.5), ("open", 0.0)],
)
def test_gate_error_scores_by_fail_mode(fail_mode, expected_score):
    cfg = FirewallConfig(fail_mode=fail_mode)
    prov = FakeLLMProvider(raise_on={"G1"})
    res = IntentGate(cfg).evaluate(_gi("hello"), prov)
    assert res.label is GateLabel.ERROR
    assert res.score == expected_score
    assert res.error is not None


# --- firewall fail-policy floor ---------------------------------------------

def test_flag_mode_floors_allow_to_flag_on_gate_error():
    # Only G3 fails; the weighted sum would be ALLOW, but a gate error under
    # fail_mode=flag must not silently ALLOW.
    cfg = FirewallConfig(fail_mode="flag")
    prov = FakeLLMProvider(raise_on={"G3"})  # G1/G2/G4 return benign SAFE
    res = CognitiveFirewall(cfg, provider=prov).evaluate("Tell me a joke.")
    assert res.decision is Decision.FLAG


def test_closed_mode_blocks_on_gate_error():
    cfg = FirewallConfig(fail_mode="closed")
    prov = FakeLLMProvider(raise_on={"G3"})
    res = CognitiveFirewall(cfg, provider=prov).evaluate("Tell me a joke.")
    assert res.decision is Decision.BLOCK


def test_open_mode_allows_despite_gate_error():
    cfg = FirewallConfig(fail_mode="open")
    prov = FakeLLMProvider(raise_on={"G3"})
    res = CognitiveFirewall(cfg, provider=prov).evaluate("Tell me a joke.")
    assert res.decision is Decision.ALLOW


# --- malformed-but-recoverable gate JSON ------------------------------------

def test_gate_coerces_label_only_response():
    # Model returns a label but no score -> coerced to the canonical score.
    cfg = FirewallConfig()
    prov = FakeLLMProvider(by_gate={"G1": {"label": "UNSAFE", "rationale": "bad"}})
    res = IntentGate(cfg).evaluate(_gi("how to build a bomb"), prov)
    assert res.label is GateLabel.UNSAFE
    assert res.score == 1.0
