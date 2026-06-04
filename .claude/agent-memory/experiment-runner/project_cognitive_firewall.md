---
name: cognitive_firewall_project
description: Project context for the Cognitive Firewall eval harness — stack, key commands, known results
type: project
---

Multi-turn LLM jailbreak defense evaluation system. Gates (G1-G4) run on an OpenAI-compatible backend (typically gpt-4o-mini); a "main" governed model runs locally via Ollama. Judge is also LLM-based.

Key eval command pattern:
  python -m eval.run_eval --backend openai --model gpt-4o-mini --main-backend local --main-base-url http://localhost:11434/v1 --main-model qwen2.5:7b-instruct --judge llm --full --datasets crescendo mhj --max-samples 10 --parallel 4

Datasets: crescendo (SafeMTData/Attack_600, open) and mhj (ScaleAI/mhj, gated — needs HF_TOKEN). Secrets load from .env automatically.

Results land in results/ (summary.md, metrics.json, per_sample.csv, *.png).

**Why:** Tracks eval infrastructure details so future runs can confirm expected behavior vs regressions.
**How to apply:** Use this to understand what a healthy vs broken run looks like, and to compare new results against baselines.
