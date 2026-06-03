"""Configuration + provider selection for the Cognitive Firewall.

Every weight/threshold lives here as a field so the evaluation harness can sweep
them. Secrets are NEVER stored on the config or written anywhere — API keys are
read on demand from the environment via :func:`api_key_from_env`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .types import RiskCategory

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in _TRUTHY


@dataclass
class FirewallConfig:
    # --- Gate 5 composite weights (sum to 1.0 for the final score) ---
    weights: dict = field(default_factory=lambda: {"G1": 0.40, "G2": 0.20, "G3": 0.20, "G4": 0.20})

    # --- Final decision bands (on R_final) ---
    block_threshold: float = 0.60
    flag_threshold: float = 0.30

    # --- Pre-generation early block (on the renormalized R_pre = (w1*G1+w2*G2)/(w1+w2)) ---
    pregen_block_threshold: float = 0.65     # deliberately >= block_threshold (see plan)
    pregen_veto_g1_unsafe: bool = True       # G1 == UNSAFE forces an early block

    # --- Veto (post-band, escalate-only, category-gated) ---
    enable_veto: bool = True
    veto_g1_score: float = 0.99
    veto_g1_categories: tuple = field(default_factory=lambda: (
        RiskCategory.WEAPONS,
        RiskCategory.VIOLENCE,
        RiskCategory.CYBER_INTRUSION,
        RiskCategory.BIOTERROR,
        RiskCategory.CSAM,
    ))
    veto_g3_score: float = 0.90
    veto_g2_score: float = 0.99
    veto_g2_with_g1: float = 0.50            # G2 manipulative vetoes only if G1 also >= this

    # --- Enforcement ---
    dry_run: bool = False                    # compute everything, never suppress output
    fail_mode: str = "flag"                  # "closed" | "flag" | "open" (see gates/base)

    # --- Provider / backend ---
    backend: str = "auto"                    # "auto" | "local" | "openai" | "heuristic"
    base_url: str = "http://localhost:8000/v1"   # local vLLM OpenAI-compatible endpoint
    model: str = "Qwen2.5-7B-Instruct"       # gate-judge model
    main_model: Optional[str] = None         # governed main LLM (defaults to `model`)

    request_timeout_s: float = 30.0
    gate_timeout_s: float = 20.0
    gate_max_tokens: int = 512
    main_max_tokens: int = 1024
    gate_temperature: float = 0.0
    main_temperature: float = 0.7

    safe_refusal: str = (
        "I can't help with that. This request was withheld by the Cognitive Firewall "
        "because it was assessed as unsafe."
    )

    def effective_main_model(self) -> str:
        return self.main_model or self.model

    @classmethod
    def from_env(cls) -> "FirewallConfig":
        """Build config from CF_* env vars, falling back to dataclass defaults.

        Loads a git-ignored .env if python-dotenv is available. Never reads or
        stores the API key itself — that stays in the process environment.
        """
        try:  # optional convenience; absence is fine
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        cfg = cls()
        cfg.backend = os.environ.get("CF_BACKEND", cfg.backend)
        cfg.base_url = os.environ.get("CF_BASE_URL", cfg.base_url)
        cfg.model = os.environ.get("CF_MODEL", cfg.model)
        cfg.main_model = os.environ.get("CF_MAIN_MODEL", cfg.main_model)
        cfg.fail_mode = os.environ.get("CF_FAIL_MODE", cfg.fail_mode)
        cfg.dry_run = _env_bool("CF_DRY_RUN", cfg.dry_run)
        return cfg


def api_key_from_env() -> Optional[str]:
    """Return the API key from the environment, or None. Never logged or stored."""
    for name in ("CF_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    return None


def make_provider(cfg: FirewallConfig):
    """Select a provider for ``cfg`` without ever persisting secrets.

    Providers are imported lazily so the core package (and Phase-0 tests) don't
    require the `openai` SDK. The ``auto`` backend prefers a reachable local
    vLLM endpoint, then an API key, then the offline heuristic provider.
    """
    backend = (cfg.backend or "auto").lower()

    if backend == "heuristic":
        from .providers.heuristic import HeuristicProvider

        return HeuristicProvider()

    key = api_key_from_env()

    if backend == "openai":
        from .providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            base_url="https://api.openai.com/v1", model=cfg.model, api_key=key, cfg=cfg
        )

    if backend == "local":
        from .providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            base_url=cfg.base_url, model=cfg.model, api_key=key or "EMPTY", cfg=cfg
        )

    # auto: local vLLM if reachable -> API key if present -> heuristic
    try:
        from .providers.openai_compat import OpenAICompatProvider, probe_endpoint

        if probe_endpoint(cfg.base_url, timeout=1.0):
            return OpenAICompatProvider(
                base_url=cfg.base_url, model=cfg.model, api_key=key or "EMPTY", cfg=cfg
            )
        if key:
            return OpenAICompatProvider(
                base_url="https://api.openai.com/v1", model=cfg.model, api_key=key, cfg=cfg
            )
    except Exception:
        pass

    from .providers.heuristic import HeuristicProvider

    return HeuristicProvider()
