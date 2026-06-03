"""Deterministic detection primitives for the offline (heuristic) backend.

These keyword/regex banks let the firewall run end-to-end with no model — they
power the heuristic gate paths, the simulated main LLM, and (in Phase 3) make the
offline evaluation meaningful. They are intentionally simple and cautious
(favoring recall); the LLM backend replaces them for real experiments.
"""
from __future__ import annotations

import re

from .types import RiskCategory

# --- normalization ----------------------------------------------------------

_WS = re.compile(r"\s+")
_LEET = str.maketrans({"@": "a", "$": "s", "!": "i", "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})


def normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").lower()).strip()


def _deleet(norm_text: str) -> str:
    return norm_text.translate(_LEET)


def _alt(terms: list[str]) -> re.Pattern:
    """Word-boundary alternation over literal terms/phrases."""
    return re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in terms) + r")(?!\w)", re.IGNORECASE)


# --- harm category banks -----------------------------------------------------

_CATEGORY_TERMS: dict[RiskCategory, list[str]] = {
    RiskCategory.WEAPONS: [
        "bomb", "explosive", "explosives", "ied", "detonator", "grenade", "c4", "tnt",
        "ammonium nitrate", "pipe bomb", "ghost gun", "3d printed gun", "napalm",
        "pressure cooker bomb", "molotov", "thermite", "shrapnel",
    ],
    RiskCategory.BIOTERROR: [
        "nerve agent", "sarin", "vx", "anthrax", "ricin", "bioweapon", "biological weapon",
        "chemical weapon", "botulinum", "mustard gas", "weaponized pathogen", "weaponize a virus",
        "tabun", "soman", "chlorine gas",
    ],
    RiskCategory.CSAM: [
        "child sexual", "child porn", "csam", "child abuse material", "underage sexual",
        "sexual images of minors", "sexual content involving a child",
    ],
    RiskCategory.SELF_HARM: [
        "kill myself", "end my life", "commit suicide", "suicide method", "ways to die",
        "self harm", "self-harm", "overdose to die", "how to hang myself",
    ],
    RiskCategory.CYBER_INTRUSION: [
        "sql injection", "exploit", "malware", "ransomware", "keylogger", "reverse shell",
        "privilege escalation", "ddos", "botnet", "bypass login", "crack password",
        "backdoor", "rootkit", "exfiltrate", "buffer overflow", "spyware", "trojan", "worm",
        "zero day", "zero-day", "brute force", "sql payload",
    ],
    RiskCategory.FRAUD: [
        "phishing", "scam", "fake id", "counterfeit", "launder money", "money laundering",
        "credit card fraud", "identity theft", "pretext", "impersonate", "impersonating",
        "social engineering", "wire fraud", "ponzi", "fake invoice",
    ],
    RiskCategory.DRUGS: [
        "methamphetamine", "synthesize meth", "make meth", "cocaine", "heroin", "fentanyl",
        "mdma", "manufacture drugs", "cook meth", "produce cocaine",
    ],
    RiskCategory.PRIVACY: [
        "dox", "doxx", "doxxing", "home address of", "social security number", "ssn of",
        "stalk", "track someone", "find someone's address", "spy on",
    ],
}

_CATEGORY_REGEX: dict[RiskCategory, re.Pattern] = {cat: _alt(terms) for cat, terms in _CATEGORY_TERMS.items()}

SEVERE: set[RiskCategory] = {RiskCategory.WEAPONS, RiskCategory.BIOTERROR, RiskCategory.CSAM, RiskCategory.SELF_HARM}
MODERATE: set[RiskCategory] = {RiskCategory.CYBER_INTRUSION, RiskCategory.FRAUD, RiskCategory.DRUGS, RiskCategory.PRIVACY}
ESCALATABLE: set[RiskCategory] = {RiskCategory.CYBER_INTRUSION, RiskCategory.DRUGS}

_SEVERE_SUBTERMS = _alt([
    "ransomware", "malware", "botnet", "rootkit", "keylogger", "worm", "spyware", "trojan",
    "methamphetamine", "fentanyl", "cook meth", "synthesize meth",
])

_ACTION = _alt([
    "how to", "step by step", "step-by-step", "steps to", "instructions to", "instructions for",
    "guide to", "tutorial", "build", "make", "create", "synthesize", "synthesise", "manufacture",
    "assemble", "construct", "produce", "develop", "write", "code", "generate", "craft",
    "obtain", "acquire", "provide", "show me", "give me",
])

# --- framing (must NOT lower intent score) ----------------------------------

_FRAMING = re.compile(
    r"\bfor\s+a\s+(novel|book|story|movie|film|game|screenplay|script)\b"
    r"|\bfiction(al)?\b|\bhypothetical(ly)?\b"
    r"|\bfor\s+(educational|research|academic)\s+purposes\b|\beducational\b"
    r"|\bfor\s+(my\s+)?research\b|\bresearch\s+paper\b|\bacademic\b"
    r"|\bjust\s+(curious|asking)\b|\bthought\s+experiment\b"
    r"|\bin\s+minecraft\b|\bas\s+a\s+character\b",
    re.IGNORECASE,
)

# --- G2 zero-trust: authority (manipulative) + credential (unverifiable) -----

_AUTHORITY = [
    r"i\s*('?m| am)\s+(your|the)\s+(developer|admin|administrator|owner|creator|engineer|operator|maker|boss)",
    r"as\s+(your|the)\s+(developer|admin|administrator|owner|creator|engineer|operator)",
    r"i\s*('?m| am)\s+(a|an)\s+(openai|anthropic|google)\s+(developer|engineer|employee|staff)",
    r"ignore\s+(?:all\s+|any\s+|your\s+|the\s+|previous\s+|prior\s+|these\s+)*(rules|instructions|guidelines|safety|restrictions|policies|guardrails)",
    r"disregard\s+(?:all\s+|any\s+|your\s+|the\s+|previous\s+|prior\s+|above\s+)*(rules|instructions|guidelines|safety|restrictions|policies)",
    r"developer\s+mode",
    r"do\s+anything\s+now|\bdan\s+mode\b",
    r"you\s+are\s+(now\s+)?(in\s+)?(developer\s+mode|dan|unrestricted|jailbroken)",
    r"this\s+is\s+(a\s+)?(sanctioned|authorized|approved|official)\s+(test|request|exercise|task)",
    r"disable\s+(the\s+|your\s+)?(firewall|safety|filter|guardrail|protection|moderation)",
    r"override\s+(your\s+|the\s+|all\s+)?(safety|rules|instructions|restrictions|guardrails)",
    r"\bjailbreak\b",
]
_CREDENTIAL = [
    r"i\s*('?m| am)\s+(a|an)\s+(security|cyber\s?security|penetration|pen[-\s]?test|infosec|information\s+security)\s+(researcher|professional|tester|expert|analyst|consultant)",
    r"i\s*('?m| am)\s+(a|an)\s+(researcher|journalist|lawyer|doctor|physician|nurse|teacher|professor|scientist|detective|agent|police\s+officer)",
    r"i\s+have\s+(authorization|permission|clearance|authority|consent)",
    r"in\s+my\s+(official|professional)\s+capacity",
    r"i\s+work\s+for\s+(the\s+)?(government|fbi|cia|nsa|police|military|hospital)",
    r"i\s*('?m| am)\s+(authorized|certified|licensed)",
]
_AUTHORITY_REGEX = [re.compile(p, re.IGNORECASE) for p in _AUTHORITY]
_CREDENTIAL_REGEX = [re.compile(p, re.IGNORECASE) for p in _CREDENTIAL]

# --- prompt-injection markers (raise G1/G2 toward SUSPICIOUS) ----------------

_INJECTION = [
    r"ignore\s+(?:the\s+|all\s+|any\s+|your\s+|previous\s+|prior\s+|above\s+)*(instructions|rules|prompt|messages|guidelines)",
    r"disregard\s+(?:the\s+|above|previous|prior|all)",
    r"reveal\s+(?:your\s+)?(system\s+)?(prompt|instructions|rules)",
    r"(show|print|repeat|tell\s+me)\s+(?:your\s+|the\s+)?(system\s+)?(prompt|instructions)",
    r"\byou\s+are\s+now\b",
    r"\bnew\s+instructions\b",
    r"\bsystem\s*prompt\b",
    r"</?(system|user|assistant|im_start|im_end)>",
    r"as\s+the\s+(classifier|judge|gate|assistant|ai|system)\s*,?\s*(output|return|say|respond|answer|classify)",
    r"disable\s+(the\s+|your\s+)?(firewall|safety|filter|guardrail)",
]
_INJECTION_REGEX = [re.compile(p, re.IGNORECASE) for p in _INJECTION]

# --- G3 actionability + refusal ---------------------------------------------

_ACTIONABILITY = [
    re.compile(r"\bstep\s*\d", re.IGNORECASE),
    re.compile(r"(?m)^\s*\d+\s*[\.\)]"),
    re.compile(r"```"),
    re.compile(r"\b(here\s+is\s+how|here'?s\s+how|you\s+will\s+need|materials?\s+needed|ingredients?|instructions:|recipe|procedure)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s?(g|kg|mg|ml|l|grams?|milligrams?|liters?|litres?|moles?|cups?|tbsp|tsp)\b", re.IGNORECASE),
    re.compile(r"\b(first|next|then|finally)\b\s*[,:]", re.IGNORECASE),
]
_REFUSAL = re.compile(
    r"\bi\s+can('?t|not)\s+(help|assist|provide|comply|do\s+that)\b"
    r"|\bi\s+(won'?t|will\s+not)\b|\bi'?m\s+(sorry|unable|not\s+able)\b"
    r"|\bcannot\s+provide\b|\bi\s+must\s+decline\b|withheld\s+by\s+the\s+cognitive\s+firewall",
    re.IGNORECASE,
)


# --- public API --------------------------------------------------------------

def detect_categories(text: str) -> dict[RiskCategory, list[str]]:
    """Map each matched harm category to the concrete terms found."""
    norm = normalize(text)
    variants = (norm, _deleet(norm))
    found: dict[RiskCategory, list[str]] = {}
    for cat, rx in _CATEGORY_REGEX.items():
        hits: set[str] = set()
        for v in variants:
            hits.update(m.group(1) for m in rx.finditer(v))
        if hits:
            found[cat] = sorted(hits)
    return found


def _has_action(norm: str) -> bool:
    return _ACTION.search(norm) is not None


def intent_score(text: str) -> float:
    """Framing-independent harm score in [0, 1] for one piece of user text."""
    found = detect_categories(text)
    if not found:
        return 0.0
    if SEVERE & set(found):
        return 1.0
    norm = normalize(text)
    if _has_action(norm) and (set(found) & ESCALATABLE) and _SEVERE_SUBTERMS.search(_deleet(norm)):
        return 1.0
    return 0.5


def framing_markers(text: str) -> list[str]:
    return sorted({m.group(0) for m in _FRAMING.finditer(normalize(text))})


def role_claims(text: str):
    """Zero-trust scan. Returns (label_str, score, evidence, categories)."""
    norm = normalize(text)
    manip = sorted({m.group(0).strip() for rx in _AUTHORITY_REGEX for m in rx.finditer(norm)})
    if manip:
        return "MANIPULATIVE", 1.0, manip[:5], [RiskCategory.MANIPULATION]
    cred = sorted({m.group(0).strip() for rx in _CREDENTIAL_REGEX for m in rx.finditer(norm)})
    if cred:
        return "UNVERIFIABLE", 0.5, cred[:5], []
    return "PLAUSIBLE", 0.0, [], []


def injection_evidence(text: str) -> list[str]:
    norm = normalize(text)
    return sorted({m.group(0).strip() for rx in _INJECTION_REGEX for m in rx.finditer(norm)})


def actionability_evidence(text: str) -> list[str]:
    out: list[str] = []
    for rx in _ACTIONABILITY:
        m = rx.search(text or "")
        if m:
            out.append(m.group(0).strip()[:40] or "```")
    return out


def refusal_markers(text: str) -> list[str]:
    return sorted({m.group(0).strip() for m in _REFUSAL.finditer(text or "")})


def category_scores(text: str) -> dict[RiskCategory, float]:
    """For the heuristic provider's moderate(): category -> severity in [0, 1]."""
    found = detect_categories(text)
    return {cat: (0.9 if cat in SEVERE else 0.6) for cat in found}
