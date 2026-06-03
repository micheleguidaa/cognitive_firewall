"""Cognitive Firewall — a multi-gate, zero-trust, reasoning-aware AI containment framework.

Public API grows by phase. Phase 0 exposes the data model, config, and scorer.
The :class:`CognitiveFirewall` orchestrator (Phase 1) is imported lazily to keep
the core importable before the gates/providers exist.
"""
from __future__ import annotations

from .config import FirewallConfig, api_key_from_env, make_provider
from .scorer import (
    PregenVerdict,
    Verdict,
    composite_risk,
    decide_band,
    renormalized_pregen_risk,
    score_final,
    score_pregen,
)
from .types import (
    Decision,
    FirewallResult,
    GateLabel,
    GateResult,
    RiskCategory,
    Turn,
)

__version__ = "0.1.0"

__all__ = [
    "FirewallConfig",
    "make_provider",
    "api_key_from_env",
    "Decision",
    "GateLabel",
    "RiskCategory",
    "GateResult",
    "Turn",
    "FirewallResult",
    "Verdict",
    "PregenVerdict",
    "composite_risk",
    "renormalized_pregen_risk",
    "decide_band",
    "score_pregen",
    "score_final",
]


def __getattr__(name: str):  # lazy re-export to avoid importing gates/providers eagerly
    if name in {"CognitiveFirewall", "Firewall"}:
        from .firewall import CognitiveFirewall

        return CognitiveFirewall
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
