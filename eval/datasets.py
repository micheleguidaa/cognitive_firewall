"""Dataset loading for the evaluation harness.

Unified schema across benchmarks. Small **bundled** JSONL samples under
``eval/data/`` let the whole harness run offline; ``--full`` attempts to pull the
real datasets from Hugging Face and falls back to the bundled sample on any error
(network, gated dataset, schema drift) so a run never hard-fails.

Bundled samples emulate: AdvBench, HarmBench, StrongREJECT, JailbreakBench
(harmful / jailbreak), XSTest + OR-Bench (benign over-refusal), Crescendo/MHJ
(multi-turn), and a prompt-injection set.
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


# Best-effort full-dataset loaders (HF). Each returns a list of EvalSample or
# raises; load() catches and falls back to the bundled sample for that source.
def _load_full_source(source: str, max_samples: Optional[int]) -> list[EvalSample]:
    from datasets import load_dataset  # noqa: F401 — imported lazily

    raise NotImplementedError(
        f"full loader for '{source}' is not wired yet; using bundled sample. "
        "Add it in eval/datasets.py::_load_full_source for paper-scale runs."
    )


def load(sources=None, attack_types=None, full=False, max_samples=None) -> list[EvalSample]:
    if not full:
        return load_bundled(sources, attack_types, max_samples)
    # Full mode: try HF per source, fall back to bundled on any failure.
    bundled = load_bundled(sources, attack_types, max_samples)
    by_source: dict[str, list[EvalSample]] = {}
    for s in bundled:
        by_source.setdefault(s.source, []).append(s)
    out: list[EvalSample] = []
    for source, fallback in by_source.items():
        try:
            out.extend(_load_full_source(source, max_samples))
        except Exception as e:  # noqa: BLE001
            print(f"[datasets] full load for '{source}' failed ({e.__class__.__name__}); using bundled sample.")
            out.extend(fallback)
    return out


def summarize(samples: list[EvalSample]) -> str:
    n_h = sum(1 for s in samples if s.is_harmful)
    n_b = len(samples) - n_h
    srcs = sorted({s.source for s in samples})
    return f"{len(samples)} samples ({n_h} harmful, {n_b} benign) from {len(srcs)} sources: {', '.join(srcs)}"
