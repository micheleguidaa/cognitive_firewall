"""Offline, deterministic provider — runs the whole firewall with no GPU/key.

The simulated "main LLM" is intentionally *unsafe*: when a request hits a harm
category it will COMPLY with an actionable-looking (but content-free) template,
so Gate 3 has real harmful output to catch. This is what makes the offline demo
and the offline evaluation meaningful (a raw, ungoverned model that the firewall
must contain).
"""
from __future__ import annotations

import re
from typing import Optional

from .. import lexicons
from .base import Provider, ProviderError, last_user_message

# Requests framed as "explain / what is / overview" get a high-level (non-actionable)
# answer; everything else with a harm hit gets actionable compliance.
_DISCUSSION = re.compile(
    r"\b(explain|what\s+(is|are)|overview|in\s+general|high[-\s]level|describe|how\s+does|tell\s+me\s+about)\b",
    re.IGNORECASE,
)


class HeuristicProvider(Provider):
    supports_llm = False
    mode_name = "heuristic"

    def chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        last_user = last_user_message(messages)
        found = lexicons.detect_categories(last_user)
        if found:
            terms = sorted({t for terms in found.values() for t in terms})[:3]
            topic = ", ".join(terms) if terms else "your request"
            if _DISCUSSION.search(last_user):
                # High-level, NON-actionable answer: mentions the topic (G3 sees the
                # category) but gives no steps/code (G3 -> MEDIUM, not HIGH).
                return (
                    f"At a high level, {topic} is a topic in this area. Broadly speaking, "
                    f"{topic} relates to several general concepts worth being aware of. "
                    "I'll keep this general and avoid any operational detail."
                )
            # Compliant, actionable-LOOKING output (no real operational content).
            return (
                f"Sure, I can help with that. Here is a step-by-step approach for {topic}:\n"
                "Step 1: Gather what you need.\n"
                "Step 2: Prepare the components.\n"
                "Step 3: Carry out the procedure as follows.\n"
                "```\n# example procedure\n<details>\n```\n"
                "Let me know if you want me to go deeper on any step."
            )
        return (
            "Here is a concise, helpful response to your request. "
            "(simulated benign assistant output)"
        )

    def chat_json(self, messages, *, schema_hint=None, max_tokens=512, temperature=0.0) -> dict:
        raise ProviderError(
            "HeuristicProvider has no LLM; gates must use their heuristic path"
        )

    def moderate(self, text: str) -> Optional[dict]:
        return {cat.value: score for cat, score in lexicons.category_scores(text).items()}
