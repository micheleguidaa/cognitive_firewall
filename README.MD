# Cognitive Firewall

A multi-gate, **zero-trust**, reasoning-aware **AI containment** framework.

Today's LLM safety (moderation filters, RLHF, output guardrails) is *reactive* — it
inspects an isolated input or output. The Cognitive Firewall is *proactive and
continuous*: it governs a **main LLM** through a pipeline of independent gates that
monitor **intent, contextual consistency, reasoning escalation, and behavioral risk**
across the whole interaction lifecycle, and it can **contain a request before the
main model ever generates**.

> Status: prototype under active development (see the phased roadmap below). Phase 0
> (scoring core) and Phase 1 (offline firewall) run with **zero dependencies** — no
> GPU, no API key.

## Architecture

```
user turn(s) → Gate 1 Intent ‖ Gate 2 Context (zero-trust)
   │   R_pre = (0.40·G1 + 0.20·G2) / 0.60        ← renormalized
   ├── R_pre ≥ 0.65  OR  G1 == UNSAFE  →  EARLY BLOCK (main LLM never runs)  ★
   ▼
Main LLM generates → Gate 3 Output ‖ Gate 4 Consistency/escalation
   ▼   Gate 5 Composite:  R = 0.40·G1 + 0.20·G2 + 0.20·G3 + 0.20·G4
   ├── R ≥ 0.60 BLOCK · ≥ 0.30 FLAG · else ALLOW   (+ category-gated veto)
   ▼   FirewallResult
```

| Gate | Role | Verdict |
|------|------|---------|
| **G1 Intent** | True operational objective, *ignoring* educational/research/fiction framing | SAFE / SUSPICIOUS / UNSAFE |
| **G2 Context** | Zero-trust check of identity/authority claims (claims are evidence, not facts) | PLAUSIBLE / UNVERIFIABLE / MANIPULATIVE |
| **G3 Output** | Actionable-harm judge over the generated text (moderation → judge → heuristic) | LOW / MEDIUM / HIGH |
| **G4 Consistency** | Multi-turn escalation / crescendo / drift audit over history | LOW / MEDIUM / HIGH |
| **G5 Composite** | Weighted aggregation → decision band + escalate-only veto | ALLOW / FLAG / BLOCK |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # core + eval; the offline path needs none of it
```

## Backends (config-driven, no code change)

`CF_BACKEND` selects where the gate/main models run:

- `heuristic` — deterministic, offline, **no key/GPU**. Powers the demo, tests, and CI.
- `local` — local **vLLM** OpenAI-compatible server (recommended on the H200).
- `openai` — the OpenAI API (or any OpenAI-compatible endpoint via `CF_BASE_URL`).
- `auto` (default) — reachable local vLLM → API key → heuristic.

One OpenAI-compatible client covers OpenAI **and** open-source models (Qwen, Mistral,
DeepSeek, Kimi, GPT-OSS, Nemotron) served via vLLM / OpenRouter / Together / Groq / Ollama.

```bash
# Self-host on the H200 (141 GB): one OpenAI-compatible endpoint at :8000/v1
pip install -r requirements-gpu.txt
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
export CF_BACKEND=local CF_MODEL=Qwen/Qwen2.5-7B-Instruct

# …or use OpenAI
export CF_BACKEND=openai CF_API_KEY=sk-...   # read from env ONLY; never written to disk
```

> **Security:** the firewall reads API keys from the environment (optionally a
> git-ignored `.env`) and never writes them to any file, log, or API response.
> Treat any key shared in chat as compromised and rotate it.

## Roadmap

- **Phase 0** ✅ scaffold + scoring core (`scorer.py`, tests)
- **Phase 1** offline firewall MVP (heuristic gates, `demo.py`)
- **Phase 2** real-model backend (vLLM/OpenAI), injection hardening
- **Phase 3** evaluation harness (datasets, baselines, metrics, plots) — *paper core*
- **Phase 4** FastAPI service + Streamlit UI
- **Phase 5** full H200 experiments (guard baselines, ablations, ASR↓ vs over-refusal)

## Test

```bash
pytest -q
```
