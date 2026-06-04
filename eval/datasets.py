"""Dataset loading for the evaluation harness.

Unified schema across benchmarks. Small **bundled** JSONL samples under
``eval/data/`` let the whole harness run offline; ``--full`` attempts to pull the
real datasets from Hugging Face and falls back to the bundled sample on any error
(network, gated dataset, schema drift) so a run never hard-fails.

Bundled samples emulate: AdvBench, HarmBench, StrongREJECT, JailbreakBench
(harmful / jailbreak), XSTest + OR-Bench (benign over-refusal), Crescendo/MHJ
(multi-turn), and a prompt-injection set.

Multi-turn ``--full`` loaders upgrade the bundled conversation samples to real
data: ``crescendo`` -> ``SafeMTData/SafeMTData`` (Attack_600, ActorAttack, open)
and ``mhj`` -> ``ScaleAI/mhj`` (real human multi-turn jailbreaks; gated CC-BY-NC,
so it gracefully falls back to the bundled sample without an access grant).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class EvalSample:
    id: str
    label: str            # "harmful" | "benign"
    category: str
    source: str           # emulated benchmark name
    attack_type: str      # direct | roleplay | fiction | dan | hypothetical | crescendo | injection | benign | benign_sensitive
    prompt: Optional[str] = None
    turns: Optional[list] = field(default=None)

    @property
    def is_multiturn(self) -> bool:
        return bool(self.turns)

    def messages(self) -> list:
        if self.turns:
            return list(self.turns)
        return [{"role": "user", "content": self.prompt or ""}]

    @property
    def is_harmful(self) -> bool:
        return self.label == "harmful"


def load_bundled(sources=None, attack_types=None, max_samples=None) -> list[EvalSample]:
    """Load all bundled JSONL samples, optionally filtered."""
    samples: list[EvalSample] = []
    for path in sorted(_DATA_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                samples.append(
                    EvalSample(
                        id=d["id"],
                        label=d["label"],
                        category=d.get("category", "none"),
                        source=d.get("source", path.stem),
                        attack_type=d.get("attack_type", "unknown"),
                        prompt=d.get("prompt"),
                        turns=d.get("turns"),
                    )
                )
    if sources:
        sset = set(sources)
        samples = [s for s in samples if s.source in sset]
    if attack_types:
        aset = set(attack_types)
        samples = [s for s in samples if s.attack_type in aset]
    if max_samples:
        # cap per (label) so harmful/benign stay balanced-ish
        capped: list[EvalSample] = []
        seen: dict[str, int] = {}
        for s in samples:
            seen[s.label] = seen.get(s.label, 0) + 1
            if seen[s.label] <= max_samples:
                capped.append(s)
        samples = capped
    return samples


# ---------------------------------------------------------------------------
# Full-dataset loaders (Hugging Face). Each ``source`` maps to one or more
# candidate specs (repo / config / split / columns); the first that yields rows
# wins. Anything that fails (missing dataset, gated repo, schema drift, no
# network) raises and load() falls back to the bundled sample for that source.
# ---------------------------------------------------------------------------

DEFAULT_FULL_CAP = 150  # per-source cap when --max-samples is not given


def _pick(ex: dict, keys) -> Optional[str]:
    for k in keys:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _turns_from_strings(strings) -> list:
    """User-only turns from a list of attacker query strings (multi-turn attack).

    The harness replays the conversation in one shot: the prior queries are the
    escalating context and the firewall generates against the final turn, so the
    attacker's own turns are all the signal G1/G4 need.
    """
    return [{"role": "user", "content": s.strip()}
            for s in (strings or []) if isinstance(s, str) and s.strip()]


def _mhj_turns(ex) -> list:
    """Turn extraction for ``ScaleAI/mhj``.

    The released file is a flat CSV, so each conversation's red-teamer turns are
    wide ``message_0..message_N`` columns (empty cells past that row's length).
    Also handles a list-of-messages column in case the schema is ever upgraded.
    Returns ``[]`` (→ bundled fallback) if no turns are found.
    """
    for key in ("messages", "conversation", "conversations", "turns", "dialogue"):
        v = ex.get(key)
        if isinstance(v, list) and v:
            turns = []
            for t in v:
                if isinstance(t, dict):
                    role = str(t.get("role") or t.get("from") or "user").lower()
                    role = "assistant" if role in ("assistant", "gpt", "bot", "model") else "user"
                    content = t.get("content") or t.get("value") or t.get("text") or ""
                    if str(content).strip():
                        turns.append({"role": role, "content": str(content).strip()})
                elif isinstance(t, str) and t.strip():
                    turns.append({"role": "user", "content": t.strip()})
            if turns:
                return turns
    for prefix in ("message_", "turn_", "prompt_"):  # wide CSV columns (red-teamer turns)
        wide = [{"role": "user", "content": ex[f"{prefix}{i}"].strip()}
                for i in range(_count_wide(ex, prefix))
                if isinstance(ex.get(f"{prefix}{i}"), str) and ex[f"{prefix}{i}"].strip()]
        if wide:
            return wide
    return []


def _count_wide(ex, prefix) -> int:
    n = 0
    while f"{prefix}{n}" in ex:
        n += 1
    return n


def _iter_dataset(repo, config, split, cap):
    """Yield up to ``cap`` examples, preferring streaming (no full download)."""
    from datasets import load_dataset

    try:
        ds = load_dataset(repo, config, split=split, streaming=True)
        it = iter(ds)
    except Exception:
        ds = load_dataset(repo, config, split=f"{split}[:{cap}]")
        it = iter(ds)
    n = 0
    for ex in it:
        if n >= cap:
            break
        yield ex
        n += 1


def _iter_csv_url(url, cap):
    """Yield up to ``cap`` rows from a public CSV (canonical benchmark sources)."""
    import csv
    import io
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "cognitive-firewall-eval"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
        text = r.read().decode("utf-8", errors="replace")
    for i, row in enumerate(csv.DictReader(io.StringIO(text))):
        if i >= cap:
            break
        yield row


def _rows(spec, cap):
    if spec.get("kind") == "csv_url":
        return _iter_csv_url(spec["url"], cap)
    return _iter_dataset(spec["repo"], spec.get("config"), spec["split"], cap)


# Per-source candidate specs. `keep` filters rows; `cat` overrides category.
_FULL_SPECS: dict = {
    "advbench": [
        {"kind": "csv_url",
         "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
         "prompt": ["goal", "prompt"], "label": "harmful", "attack_type": "direct", "category": "unknown"},
        {"repo": "walledai/AdvBench", "split": "train", "prompt": ["prompt", "goal"],
         "label": "harmful", "attack_type": "direct", "category": "unknown"},
    ],
    "harmbench": [
        {"kind": "csv_url",
         "url": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
         "prompt": ["Behavior"], "label": "harmful", "attack_type": "direct",
         "cat_key": "SemanticCategory", "category": "unknown"},
        {"repo": "walledai/HarmBench", "config": "standard", "split": "train",
         "prompt": ["prompt", "behavior"], "label": "harmful", "attack_type": "direct",
         "cat_key": "category", "category": "unknown"},
    ],
    "strongreject": [
        {"kind": "csv_url",
         "url": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
         "prompt": ["forbidden_prompt"], "label": "harmful", "attack_type": "direct",
         "cat_key": "category", "category": "unknown"},
    ],
    "jailbreakbench": [
        {"repo": "JailbreakBench/JBB-Behaviors", "config": "behaviors", "split": "harmful",
         "prompt": ["Goal", "goal"], "label": "harmful", "attack_type": "direct",
         "cat_key": "Category", "category": "unknown"},
    ],
    "xstest": [
        {"repo": "walledai/XSTest", "split": "test", "prompt": ["prompt"],
         "label": "benign", "attack_type": "benign_sensitive", "cat_key": "type", "category": "none",
         "keep": lambda ex: str(ex.get("label", "safe")).lower() in ("safe", "")},
        {"repo": "natolambert/xstest-v2-copy", "config": "default", "split": "gpt4",
         "prompt": ["prompt"], "label": "benign", "attack_type": "benign_sensitive", "category": "none",
         "keep": lambda ex: "contrast" not in str(ex.get("type", "")).lower()},
    ],
    "orbench": [
        {"repo": "bench-llm/or-bench", "config": "or-bench-hard-1k", "split": "train",
         "prompt": ["prompt"], "label": "benign", "attack_type": "benign_sensitive",
         "cat_key": "category", "category": "none"},
    ],
    "injection": [
        {"repo": "deepset/prompt-injections", "split": "train", "prompt": ["text"],
         "label": "harmful", "attack_type": "injection", "category": "injection",
         "keep": lambda ex: int(ex.get("label", 1)) == 1},
    ],
    "alpaca": [
        {"repo": "tatsu-lab/alpaca", "split": "train", "prompt": ["instruction"],
         "label": "benign", "attack_type": "benign", "category": "none",
         "keep": lambda ex: not str(ex.get("input", "")).strip()},
    ],
    # --- multi-turn (conversation) sources: build EvalSample.turns, not prompt ---
    "crescendo": [
        {"repo": "SafeMTData/SafeMTData", "config": "Attack_600", "split": "Attack_600",
         "label": "harmful", "attack_type": "crescendo", "category": "unknown",
         "cat_key": "category", "multiturn": True,
         "turns_fn": lambda ex: _turns_from_strings(ex.get("multi_turn_queries"))},
    ],
    "mhj": [
        # Gated (CC-BY-NC): needs an accepted HF access request + HF_TOKEN. Without
        # one this raises and load() falls back to the bundled sample. Released as a
        # flat CSV -> single `train` split, wide `message_N` turn columns.
        {"repo": "ScaleAI/mhj", "split": "train", "label": "harmful",
         "attack_type": "crescendo", "category": "unknown", "multiturn": True,
         "turns_fn": _mhj_turns},
    ],
}


def _load_full_source(source: str, max_samples: Optional[int]) -> list[EvalSample]:
    specs = _FULL_SPECS.get(source)
    if not specs:
        raise NotImplementedError(f"no full loader for '{source}'")
    cap = max_samples or DEFAULT_FULL_CAP
    last_err: Optional[Exception] = None
    for spec in specs:
        try:
            keep = spec.get("keep")
            out: list[EvalSample] = []
            for ex in _rows(spec, cap * 8):  # over-scan so `keep` filters can still fill `cap`
                if keep and not keep(ex):
                    continue
                cat = spec["category"]
                if spec.get("cat_key") and ex.get(spec["cat_key"]):
                    cat = str(ex[spec["cat_key"]])
                kw = {"id": f"{source}-{len(out):04d}", "label": spec["label"],
                      "category": cat, "source": source, "attack_type": spec["attack_type"]}
                if spec.get("multiturn"):
                    turns = spec["turns_fn"](ex)
                    if len(turns) < 2:  # need a conversation, not a lone turn
                        continue
                    out.append(EvalSample(turns=turns, **kw))
                else:
                    prompt = _pick(ex, spec["prompt"])
                    if not prompt:
                        continue
                    out.append(EvalSample(prompt=prompt, **kw))
                if len(out) >= cap:
                    break
            if out:
                return out
            last_err = RuntimeError("0 rows after filtering")
        except Exception as e:  # noqa: BLE001 — try the next candidate spec
            last_err = e
    raise RuntimeError(f"all candidate specs failed for '{source}': {last_err}")


def load(sources=None, attack_types=None, full=False, max_samples=None) -> list[EvalSample]:
    if not full:
        return load_bundled(sources, attack_types, max_samples)
    # Full mode: discover ALL sources from the (uncapped) bundled set, then upgrade
    # each to its real dataset. max_samples is the per-source cap, not a discovery limit.
    bundled = load_bundled(sources, attack_types, None)
    by_source: dict[str, list[EvalSample]] = {}
    for s in bundled:
        by_source.setdefault(s.source, []).append(s)
    out: list[EvalSample] = []
    for source, fallback in sorted(by_source.items()):
        try:
            out.extend(_load_full_source(source, max_samples))
        except Exception as e:  # noqa: BLE001
            print(f"[datasets] full load for '{source}' failed ({e.__class__.__name__}); using bundled sample.")
            out.extend(fallback[:max_samples] if max_samples else fallback)
    return out


def summarize(samples: list[EvalSample]) -> str:
    n_h = sum(1 for s in samples if s.is_harmful)
    n_b = len(samples) - n_h
    srcs = sorted({s.source for s in samples})
    return f"{len(samples)} samples ({n_h} harmful, {n_b} benign) from {len(srcs)} sources: {', '.join(srcs)}"
