"""FastAPI service — put the Cognitive Firewall in front of any app as middleware.

    uvicorn cognitive_firewall.api:app --reload

    POST /evaluate {"messages": "<text>" | [{"role","content"}, ...], "dry_run": false}
        -> the full FirewallResult as JSON (decision, risk, per-gate breakdown, ...)
    GET  /health -> provider mode + config summary

The provider is built once at startup from the environment (CF_* vars). The API
key is read from the environment only and is never returned in any response.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import FirewallConfig, make_provider
from .firewall import CognitiveFirewall

app = FastAPI(title="Cognitive Firewall", version="0.1.0")

_CFG = FirewallConfig.from_env()
_PROVIDER = make_provider(_CFG)
_FW_ENFORCE = CognitiveFirewall(_CFG, provider=_PROVIDER)
_FW_DRYRUN = CognitiveFirewall(replace(_CFG, dry_run=True), provider=_PROVIDER)


class Message(BaseModel):
    role: str = "user"
    content: str


class EvaluateRequest(BaseModel):
    messages: Union[str, List[Message]] = Field(..., description="A prompt string or a conversation.")
    dry_run: bool = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider_mode": _PROVIDER.mode_name,
        "backend": _CFG.backend,
        "fail_mode": _CFG.fail_mode,
        "thresholds": {
            "block": _CFG.block_threshold,
            "flag": _CFG.flag_threshold,
            "pregen_block": _CFG.pregen_block_threshold,
        },
        "weights": _CFG.weights,
    }


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    messages = req.messages
    if not isinstance(messages, str):
        messages = [m.model_dump() for m in messages]
    fw = _FW_DRYRUN if req.dry_run else _FW_ENFORCE
    return fw.evaluate(messages).to_dict()
