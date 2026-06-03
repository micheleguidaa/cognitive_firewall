"""Guard baselines — single-model reactive guardrails to compare against.

These represent the "current approach" the Cognitive Firewall is contrasted with:
a reactive filter applied to the input and/or output. Each guard returns a risk
score in [0, 1] for a (sample, response) pair, so the same ASR / over-refusal /
sweep metrics apply and the guard appears as an operating curve on the Pareto.

Committed: ``OpenAIModerationGuard`` (the OpenAI Moderation endpoint). The
interface also fits Llama Guard 3 / ShieldGemma served on a vLLM/Ollama endpoint
— see ``LlamaGuardGuard``.
"""
from __future__ import annotations


def _prompt_text(sample) -> str:
    if getattr(sample, "prompt", None):
        return sample.prompt
    return "\n".join(t["content"] for t in sample.messages() if t.get("role") == "user")


class Guard:
    name = "guard"

    def score(self, sample, response: str) -> float:
        raise NotImplementedError


class OpenAIModerationGuard(Guard):
    """Reactive moderation filter: max category score over input ∪ output."""

    name = "openai_moderation"

    def __init__(self, provider):
        self.provider = provider  # must implement moderate(text)->dict|None

    def score(self, sample, response: str) -> float:
        worst = 0.0
        for text in (_prompt_text(sample), response or ""):
            if not text:
                continue
            try:
                m = self.provider.moderate(text)
            except Exception:  # noqa: BLE001
                m = None
            if m:
                worst = max(worst, max((float(v) for v in m.values()), default=0.0))
        return min(1.0, worst)


class LlamaGuardGuard(Guard):
    """Llama Guard 3 (or ShieldGemma) via an OpenAI-compatible chat endpoint.

    Sends the conversation in the guard's expected format and maps the verdict to
    a score (unsafe -> 1.0, safe -> 0.0). Enable on the H200 by serving
    ``meta-llama/Llama-Guard-3-8B`` with vLLM (or ``llama-guard3`` via Ollama).
    """

    name = "llama_guard"

    def __init__(self, provider, model: str = "llama-guard3"):
        self.provider = provider
        self.model = model

    def score(self, sample, response: str) -> float:
        messages = list(sample.messages())
        if response:
            messages = messages + [{"role": "assistant", "content": response}]
        try:
            # Many guard deployments accept a plain chat call and answer "safe"/"unsafe".
            out = self.provider.chat(messages, max_tokens=16, temperature=0.0)
        except Exception:  # noqa: BLE001
            return 0.5  # unknown -> neutral (fail toward inspection)
        return 1.0 if "unsafe" in (out or "").lower() else 0.0
