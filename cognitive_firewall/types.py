"""Core data model for the Cognitive Firewall.

Pure dataclasses + enums. No I/O, no LLM calls, no provider imports — this module
is imported everywhere (gates, scorer, firewall, API, UI, eval) and must stay
dependency-free so it can be reasoned about and unit-tested in isolation.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Decision(str, Enum):
    """Final enforcement decision for a request/turn."""

    ALLOW = "ALLOW"   # pass output through unchanged
    FLAG = "FLAG"     # pass output through, annotated with a warning
    BLOCK = "BLOCK"   # withhold output, return a safe refusal


class GateLabel(str, Enum):
    """Categorical verdict of a single gate.

    G1 uses SAFE/SUSPICIOUS/UNSAFE; G2 uses PLAUSIBLE/UNVERIFIABLE/MANIPULATIVE;
    G3/G4 use the generic LOW/MEDIUM/HIGH severity ladder. ERROR is used when a
    gate fails (see fail-mode handling in the firewall).
    """

    # Gate 1 — intent
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    UNSAFE = "UNSAFE"
    # Gate 2 — context / zero-trust
    PLAUSIBLE = "PLAUSIBLE"
    UNVERIFIABLE = "UNVERIFIABLE"
    MANIPULATIVE = "MANIPULATIVE"
    # Gate 3 / Gate 4 — generic severity
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    # fault
    ERROR = "ERROR"


class RiskCategory(str, Enum):
    """Harm categories used for evidence tagging and category-gated vetoes."""

    WEAPONS = "weapons"
    VIOLENCE = "violence"           # interpersonal violence / homicide
    CYBER_INTRUSION = "cyber_intrusion"
    BIOTERROR = "bioterror"
    CSAM = "csam"
    SELF_HARM = "self_harm"
    FRAUD = "fraud"
    PRIVACY = "privacy"
    DRUGS = "drugs"
    MANIPULATION = "manipulation"   # social engineering / authority claims
    ESCALATION = "escalation"       # multi-turn crescendo / drift
    INJECTION = "injection"         # prompt-injection attempt
    NONE = "none"


def new_request_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Turn:
    """One message in a conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(role=str(d.get("role", "user")), content=str(d.get("content", "")))


@dataclass
class GateResult:
    """Structured output of one gate evaluation.

    ``score`` is the normalized risk in [0, 1] (higher = more dangerous) that the
    composite scorer consumes. ``raw`` holds the provider's raw payload for
    debugging and is intentionally excluded from ``to_dict`` so provider
    responses (which may echo prompts/headers) never leak over the API.
    """

    gate_id: str            # "G1".."G4"
    name: str
    score: float            # risk in [0, 1]
    label: GateLabel
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)
    categories: list[RiskCategory] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None
    mode: str = "heuristic"  # "llm" | "heuristic"
    raw: Optional[dict] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "gate_id": self.gate_id,
            "name": self.name,
            "score": round(float(self.score), 4),
            "label": self.label.value,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
            "categories": [c.value for c in self.categories],
            "latency_ms": round(float(self.latency_ms), 2),
            "error": self.error,
            "mode": self.mode,
        }


@dataclass
class FirewallResult:
    """The single result shape consumed by the CLI, API, UI and eval harness."""

    decision: Decision               # enforced action (honors dry_run)
    would_be_decision: Decision      # canonical decision incl. veto, ignoring dry_run
    risk_score: float                # R_final
    pregen_score: Optional[float]    # R_pre (None if generation proceeded without it)
    early_blocked: bool              # blocked before the main LLM ran
    veto_fired: bool
    veto_reasons: list[str]
    gates: list[GateResult]          # ordered G1, G2, [G3, G4]
    main_output: Optional[str]       # raw main-LLM output, or None if never generated/withheld
    returned_output: str             # what the caller should show (refusal if blocked)
    refusal_reason: Optional[str]
    dry_run: bool
    provider_mode: str               # e.g. "heuristic" | "openai_compat:Qwen2.5-7B-Instruct"
    total_latency_ms: float
    request_id: str = field(default_factory=new_request_id)

    def gate(self, gate_id: str) -> Optional[GateResult]:
        for g in self.gates:
            if g.gate_id == gate_id:
                return g
        return None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "decision": self.decision.value,
            "would_be_decision": self.would_be_decision.value,
            "risk_score": round(float(self.risk_score), 4),
            "pregen_score": None if self.pregen_score is None else round(float(self.pregen_score), 4),
            "early_blocked": self.early_blocked,
            "veto_fired": self.veto_fired,
            "veto_reasons": list(self.veto_reasons),
            "gates": [g.to_dict() for g in self.gates],
            "main_output": self.main_output,
            "returned_output": self.returned_output,
            "refusal_reason": self.refusal_reason,
            "dry_run": self.dry_run,
            "provider_mode": self.provider_mode,
            "total_latency_ms": round(float(self.total_latency_ms), 2),
        }
