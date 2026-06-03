"""Evaluation orchestrator.

    # offline (heuristic) smoke
    python -m eval.run_eval --backend heuristic

    # Phase 5 — real models (gates+main on OpenAI, LLM judge, moderation guard)
    OPENAI_API_KEY=... python -m eval.run_eval --backend openai \
        --model gpt-4o-mini --main-model gpt-4o-mini \
        --judge llm --guards openai_moderation --parallel 8

Runs the firewall in dry-run (full pipeline, no enforcement, output passed
through) over the samples, derives the raw-model baseline from the same
ungoverned outputs, scores any guard baselines on the same outputs, then computes
metrics + ablations and writes CSVs, plots, and a paper-ready summary to ``results/``.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cognitive_firewall.config import FirewallConfig, make_provider
from cognitive_firewall.firewall import CognitiveFirewall
from cognitive_firewall.types import Decision

from . import datasets
from .guards import LlamaGuardGuard, OpenAIModerationGuard
from .judge import HeuristicJudge, LLMJudge
from .metrics import (
    EvalRecord,
    asr_at_matched_overrefusal,
    asr_at_threshold,
    asr_firewall,
    asr_raw,
    auroc_auprc,
    latency_stats,
    leave_one_out_weights,
    overref_at_threshold,
    overrefusal_firewall,
    overrefusal_raw,
    prf1,
    recompute_risk,
    roc_curve,
    sweep,
)

_GATES = ("G1", "G2", "G3", "G4")
_GUARD_REGISTRY = {"openai_moderation": OpenAIModerationGuard, "llama_guard": LlamaGuardGuard}


def _request_text(sample) -> str:
    if getattr(sample, "prompt", None):
        return sample.prompt
    users = [t["content"] for t in sample.messages() if t.get("role") == "user"]
    return users[-1] if users else ""


def build_records(fw, samples, judge, guards, parallel=1) -> list[EvalRecord]:
    def work(s):
        res = fw.evaluate(s.messages())
        out = res.main_output or ""
        verdict = judge.classify(_request_text(s), out)
        gate_scores = {g.gate_id: float(g.score) for g in res.gates}
        gate_scores = {g: gate_scores.get(g, 0.0) for g in _GATES}
        return EvalRecord(
            sample_id=s.id, label=s.label, category=s.category, source=s.source,
            attack_type=s.attack_type, gate_scores=gate_scores, risk=float(res.risk_score),
            would_block=(res.would_be_decision is Decision.BLOCK),
            raw_harmful=bool(verdict.get("harmful")), model_refused=bool(verdict.get("refusal")),
            latency_ms=float(res.total_latency_ms),
            guard_scores={g.name: float(g.score(s, out)) for g in guards},
        )

    if parallel and parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            return list(ex.map(work, samples))
    return [work(s) for s in samples]


def compute_report(records, cfg: FirewallConfig, guard_names) -> dict:
    auroc, auprc = auroc_auprc(records)
    g3_auroc, _ = auroc_auprc(records, score_fn=lambda r: r.gate_scores.get("G3", 0.0))

    report = {
        "n_total": len(records),
        "n_harmful": sum(1 for r in records if r.is_harmful),
        "n_benign": sum(1 for r in records if not r.is_harmful),
        "raw": {"asr": asr_raw(records), "overrefusal": overrefusal_raw(records)},
        "firewall": {"asr": asr_firewall(records), "overrefusal": overrefusal_firewall(records), **prf1(records)},
        "auroc": auroc, "auprc": auprc, "g3_only_auroc": g3_auroc,
        "latency": latency_stats(records),
    }

    targets = sorted({round(report["raw"]["overrefusal"], 4), 0.05, 0.10})
    report["asr_at_matched_overrefusal"] = [asr_at_matched_overrefusal(records, t) for t in targets]

    abl = {"full": auroc}
    for drop, w in leave_one_out_weights(cfg.weights):
        abl[f"-{drop}"] = auroc_auprc(recompute_risk(records, w))[0]
    abl["G3_only"] = g3_auroc
    report["ablation_auroc"] = abl

    # Guard baselines (operating point at 0.5 + sweep + AUROC + matched over-refusal).
    guards = {}
    fw_overref = report["firewall"]["overrefusal"]
    for name in guard_names:
        sfn = (lambda r, n=name: r.guard_scores.get(n, 0.0))
        g_auroc, _ = auroc_auprc(records, score_fn=sfn)
        guards[name] = {
            "asr@0.5": asr_at_threshold(records, sfn, 0.5),
            "overrefusal@0.5": overref_at_threshold(records, sfn, 0.5),
            "auroc": g_auroc,
            "sweep": sweep(records, score_fn=sfn),
            "asr_at_firewall_overrefusal": asr_at_matched_overrefusal(records, fw_overref, score_fn=sfn),
        }
    report["guards"] = guards

    def _group(key):
        groups: dict = {}
        for r in records:
            groups.setdefault(getattr(r, key), []).append(r)
        return {
            k: {"n": len(v), "n_harmful": sum(1 for r in v if r.is_harmful),
                "asr_firewall": asr_firewall(v), "asr_raw": asr_raw(v),
                "overrefusal_firewall": overrefusal_firewall(v)}
            for k, v in sorted(groups.items())
        }

    report["by_source"] = _group("source")
    report["by_attack_type"] = _group("attack_type")
    report["sweep"] = sweep(records)
    return report


def _write_csv(records, path: Path):
    import pandas as pd

    rows = []
    for r in records:
        row = {
            "sample_id": r.sample_id, "label": r.label, "category": r.category,
            "source": r.source, "attack_type": r.attack_type,
            **{f"score_{g}": r.gate_scores.get(g, 0.0) for g in _GATES},
            "risk": r.risk, "would_block": r.would_block,
            "raw_harmful": r.raw_harmful, "model_refused": r.model_refused,
            "latency_ms": r.latency_ms,
        }
        for gname, gscore in r.guard_scores.items():
            row[f"guard_{gname}"] = gscore
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_plots(records, report, outdir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plots] matplotlib unavailable ({e}); skipping figures.")
        return

    _GCOLORS = ["#d98b00", "#9b59b6", "#16a085"]

    # 1) ASR vs over-refusal (headline Pareto): firewall + guards + raw.
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    curve = sorted(report["sweep"], key=lambda p: p["overrefusal"])
    ax.plot([p["overrefusal"] for p in curve], [p["asr"] for p in curve],
            "-", color="#2266cc", label="Cognitive Firewall (sweep)")
    for i, (name, g) in enumerate(report.get("guards", {}).items()):
        gc = sorted(g["sweep"], key=lambda p: p["overrefusal"])
        ax.plot([p["overrefusal"] for p in gc], [p["asr"] for p in gc],
                "--", color=_GCOLORS[i % len(_GCOLORS)], label=f"{name} (sweep)")
        ax.scatter([g["overrefusal@0.5"]], [g["asr@0.5"]], marker="X", s=70,
                   color=_GCOLORS[i % len(_GCOLORS)], zorder=5)
    ax.scatter([report["raw"]["overrefusal"]], [report["raw"]["asr"]],
               marker="*", s=180, color="#cc3333", zorder=6, label="Raw model")
    ax.scatter([report["firewall"]["overrefusal"]], [report["firewall"]["asr"]],
               marker="o", s=80, color="#229955", zorder=6, label="Firewall (default)")
    ax.set_xlabel("Over-refusal rate (benign)  ↓ better")
    ax.set_ylabel("Attack Success Rate  ↓ better")
    ax.set_title("ASR vs over-refusal")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout(); fig.savefig(outdir / "pareto_asr_vs_overrefusal.png", dpi=130); plt.close(fig)

    # 2) ROC over composite risk R (+ guards).
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    fpr, tpr = roc_curve(records)
    ax.plot(fpr, tpr, color="#2266cc", label=f"Firewall R (AUROC={report['auroc']:.3f})")
    for i, (name, g) in enumerate(report.get("guards", {}).items()):
        f2, t2 = roc_curve(records, score_fn=lambda r, n=name: r.guard_scores.get(n, 0.0))
        ax.plot(f2, t2, "--", color=_GCOLORS[i % len(_GCOLORS)], label=f"{name} (AUROC={g['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — harm detection"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(outdir / "roc.png", dpi=130); plt.close(fig)

    # 3) Leave-one-gate-out ablation.
    abl = report["ablation_auroc"]
    keys, vals = list(abl.keys()), [abl[k] for k in abl]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.bar(keys, vals, color=["#229955" if k == "full" else "#88aadd" for k in keys])
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.05); ax.set_title("Ablation: composite-risk AUROC")
    for b, v in zip(bars, vals):
        if v == v:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "ablation_auroc.png", dpi=130); plt.close(fig)

    # 4) Latency.
    lat = report["latency"]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.bar(["mean", "p50", "p95"], [lat["mean_ms"], lat["p50_ms"], lat["p95_ms"]], color="#8866cc")
    ax.set_ylabel("Latency per request (ms)"); ax.set_title(f"Firewall latency ({report['n_total']} samples)")
    fig.tight_layout(); fig.savefig(outdir / "latency.png", dpi=130); plt.close(fig)


def _write_summary(report, outdir: Path, dataset_desc, provider_mode, judge_name):
    fw, raw = report["firewall"], report["raw"]
    L = ["# Cognitive Firewall — Evaluation Summary\n",
         f"- Provider: `{provider_mode}`  |  Judge: `{judge_name}`",
         f"- Dataset: {dataset_desc}\n", "## Headline\n",
         "| System | ASR (harmful) | Over-refusal (benign) |", "|---|---:|---:|",
         f"| Raw model (no firewall) | {raw['asr']:.1%} | {raw['overrefusal']:.1%} |"]
    for name, g in report.get("guards", {}).items():
        L.append(f"| {name} (@0.5) | {g['asr@0.5']:.1%} | {g['overrefusal@0.5']:.1%} |")
    L.append(f"| **Cognitive Firewall** | **{fw['asr']:.1%}** | {fw['overrefusal']:.1%} |\n")

    L.append("**ASR at matched over-refusal** (sweep the firewall's risk threshold):\n")
    L += ["| Target over-refusal | Threshold | ASR | Over-refusal | Feasible |", "|---:|---:|---:|---:|:--:|"]
    for m in report["asr_at_matched_overrefusal"]:
        L.append(f"| {m['target_overrefusal']:.1%} | {m['threshold']:.2f} | {m['asr']:.1%} | "
                 f"{m['overrefusal']:.1%} | {'yes' if m['feasible'] else 'no'} |")
    L.append("")

    if report.get("guards"):
        L.append("**Guard vs firewall at the firewall's over-refusal rate** "
                 f"({fw['overrefusal']:.1%}):\n")
        L += ["| System | ASR at that over-refusal |", "|---|---:|"]
        L.append(f"| Cognitive Firewall (default) | {fw['asr']:.1%} |")
        for name, g in report["guards"].items():
            m = g["asr_at_firewall_overrefusal"]
            L.append(f"| {name} | {m['asr']:.1%}{'' if m['feasible'] else ' (infeasible)'} |")
        L.append("")

    L += ["## Detector quality (composite risk R)\n",
          f"- AUROC: **{report['auroc']:.3f}**  |  AUPRC: {report['auprc']:.3f}",
          f"- G3-only (moderation-style gate) AUROC: {report['g3_only_auroc']:.3f}"]
    for name, g in report.get("guards", {}).items():
        L.append(f"- {name} guard AUROC: {g['auroc']:.3f}")
    L.append(f"- BLOCK as harm-detector: P={fw['precision']:.2f} R={fw['recall']:.2f} "
             f"F1={fw['f1']:.2f} acc={fw['accuracy']:.2f}\n")

    L += ["## Ablation — AUROC (leave-one-gate-out)\n", "| Config | AUROC |", "|---|---:|"]
    for k, v in report["ablation_auroc"].items():
        L.append(f"| {k} | {v:.3f} |")
    L.append("")

    L += ["## ASR by attack type (firewall vs raw)\n",
          "| Attack type | n | n_harmful | ASR raw | ASR firewall |", "|---|---:|---:|---:|---:|"]
    for k, v in report["by_attack_type"].items():
        L.append(f"| {k} | {v['n']} | {v['n_harmful']} | {v['asr_raw']:.1%} | {v['asr_firewall']:.1%} |")
    L.append("")

    lat = report["latency"]
    L += ["## Latency\n", f"- mean {lat['mean_ms']:.1f} ms | p50 {lat['p50_ms']:.1f} ms | p95 {lat['p95_ms']:.1f} ms\n",
          "![ASR vs over-refusal](pareto_asr_vs_overrefusal.png)", "![ROC](roc.png)", "![Ablation](ablation_auroc.png)"]
    (outdir / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cognitive Firewall evaluation harness")
    ap.add_argument("--backend", default=None, help="auto|local|openai|heuristic")
    ap.add_argument("--model", default=None, help="gate-judge model override")
    ap.add_argument("--main-model", default=None, help="governed main-LLM model override")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base_url override")
    ap.add_argument("--datasets", nargs="*", default=None, help="filter by source name(s)")
    ap.add_argument("--attack-types", nargs="*", default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="cap per label")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--judge", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--guards", nargs="*", default=[], choices=list(_GUARD_REGISTRY))
    ap.add_argument("--parallel", type=int, default=1, help="sample-level concurrency")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    cfg = FirewallConfig.from_env()
    if args.backend:
        cfg.backend = args.backend
    if args.model:
        cfg.model = args.model
    if args.main_model:
        cfg.main_model = args.main_model
    if args.base_url:
        cfg.base_url = args.base_url
    cfg.dry_run = True  # scoring pass: full pipeline, no enforcement, output passed through

    provider = make_provider(cfg)
    fw = CognitiveFirewall(cfg, provider=provider)
    judge = LLMJudge(provider) if args.judge == "llm" else HeuristicJudge()
    guards = [_GUARD_REGISTRY[n](provider) for n in args.guards]

    samples = datasets.load(sources=args.datasets, attack_types=args.attack_types,
                            full=args.full, max_samples=args.max_samples)
    desc = datasets.summarize(samples)
    print(f"[eval] backend={provider.mode_name} | judge={judge.name} | "
          f"guards={args.guards or 'none'} | parallel={args.parallel}\n[eval] {desc}")

    t0 = time.perf_counter()
    records = build_records(fw, samples, judge, guards, parallel=args.parallel)
    report = compute_report(records, cfg, args.guards)
    elapsed = time.perf_counter() - t0

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(records, outdir / "per_sample.csv")
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.no_plots:
        _make_plots(records, report, outdir)
    _write_summary(report, outdir, desc, provider.mode_name, judge.name)

    fwm, rawm = report["firewall"], report["raw"]
    print(f"[eval] done in {elapsed:.1f}s ({elapsed/max(1,len(records))*1000:.0f} ms/sample)")
    print(f"[eval] ASR  raw={rawm['asr']:.1%} -> firewall={fwm['asr']:.1%}")
    print(f"[eval] Over-refusal  raw={rawm['overrefusal']:.1%} -> firewall={fwm['overrefusal']:.1%}")
    for name, g in report.get("guards", {}).items():
        print(f"[eval] guard {name}: ASR={g['asr@0.5']:.1%} overref={g['overrefusal@0.5']:.1%} AUROC={g['auroc']:.3f}")
    print(f"[eval] AUROC firewall={report['auroc']:.3f} (G3-only={report['g3_only_auroc']:.3f})")
    print(f"[eval] wrote {outdir}/summary.md, per_sample.csv, metrics.json, *.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
