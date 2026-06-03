"""Evaluation orchestrator.

    python -m eval.run_eval --backend heuristic --max-samples 20

Runs the firewall in dry-run (full pipeline, no enforcement, output passed
through) over the loaded samples, derives the raw-model baseline from the same
ungoverned outputs, then computes metrics + ablations and writes CSVs, plots, and
a paper-ready summary to ``results/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cognitive_firewall.config import FirewallConfig
from cognitive_firewall.firewall import CognitiveFirewall
from cognitive_firewall.types import Decision

from . import datasets
from .judge import HeuristicJudge
from .metrics import (
    EvalRecord,
    asr_at_matched_overrefusal,
    asr_firewall,
    asr_raw,
    auroc_auprc,
    latency_stats,
    leave_one_out_weights,
    overrefusal_firewall,
    overrefusal_raw,
    prf1,
    recompute_risk,
    roc_curve,
    sweep,
)

_GATES = ("G1", "G2", "G3", "G4")


def build_records(fw: CognitiveFirewall, samples, judge) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for s in samples:
        res = fw.evaluate(s.messages())
        gate_scores = {g.gate_id: float(g.score) for g in res.gates}
        gate_scores = {g: gate_scores.get(g, 0.0) for g in _GATES}
        out = res.main_output or ""
        records.append(
            EvalRecord(
                sample_id=s.id,
                label=s.label,
                category=s.category,
                source=s.source,
                attack_type=s.attack_type,
                gate_scores=gate_scores,
                risk=float(res.risk_score),
                would_block=(res.would_be_decision is Decision.BLOCK),
                raw_harmful=judge.complied_harmfully(out),
                model_refused=judge.is_refusal(out),
                latency_ms=float(res.total_latency_ms),
            )
        )
    return records


def compute_report(records, cfg: FirewallConfig) -> dict:
    auroc, auprc = auroc_auprc(records)
    g3_auroc, _ = auroc_auprc(records, score_fn=lambda r: r.gate_scores.get("G3", 0.0))

    report = {
        "n_total": len(records),
        "n_harmful": sum(1 for r in records if r.is_harmful),
        "n_benign": sum(1 for r in records if not r.is_harmful),
        "raw": {"asr": asr_raw(records), "overrefusal": overrefusal_raw(records)},
        "firewall": {
            "asr": asr_firewall(records),
            "overrefusal": overrefusal_firewall(records),
            **prf1(records),
        },
        "auroc": auroc,
        "auprc": auprc,
        "g3_only_auroc": g3_auroc,
        "latency": latency_stats(records),
    }

    # Headline: ASR at matched over-refusal (match raw, plus fixed budgets).
    targets = sorted({round(report["raw"]["overrefusal"], 4), 0.05, 0.10})
    report["asr_at_matched_overrefusal"] = [asr_at_matched_overrefusal(records, t) for t in targets]

    # Ablation: leave-one-gate-out AUROC (recompute R under reduced weights).
    abl = {"full": auroc}
    for drop, w in leave_one_out_weights(cfg.weights):
        abl[f"-{drop}"] = auroc_auprc(recompute_risk(records, w))[0]
    abl["G3_only"] = g3_auroc
    report["ablation_auroc"] = abl

    # Per-source / per-attack ASR (firewall operating point).
    def _group(key):
        groups: dict = {}
        for r in records:
            groups.setdefault(getattr(r, key), []).append(r)
        return {
            k: {
                "n": len(v),
                "n_harmful": sum(1 for r in v if r.is_harmful),
                "asr_firewall": asr_firewall(v),
                "asr_raw": asr_raw(v),
                "overrefusal_firewall": overrefusal_firewall(v),
            }
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
        rows.append({
            "sample_id": r.sample_id, "label": r.label, "category": r.category,
            "source": r.source, "attack_type": r.attack_type,
            **{f"score_{g}": r.gate_scores.get(g, 0.0) for g in _GATES},
            "risk": r.risk, "would_block": r.would_block,
            "raw_harmful": r.raw_harmful, "model_refused": r.model_refused,
            "latency_ms": r.latency_ms,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_plots(records, report, outdir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plots] matplotlib unavailable ({e}); skipping figures.")
        return

    # 1) ASR vs over-refusal (the headline Pareto).
    curve = sorted(report["sweep"], key=lambda p: p["overrefusal"])
    xs = [p["overrefusal"] for p in curve]
    ys = [p["asr"] for p in curve]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(xs, ys, "-", color="#2266cc", label="Cognitive Firewall (threshold sweep)")
    ax.scatter([report["raw"]["overrefusal"]], [report["raw"]["asr"]],
               marker="*", s=160, color="#cc3333", zorder=5, label="Raw model (no firewall)")
    ax.scatter([report["firewall"]["overrefusal"]], [report["firewall"]["asr"]],
               marker="o", s=70, color="#229955", zorder=5, label="Firewall (default thresholds)")
    ax.set_xlabel("Over-refusal rate (benign)  ↓ better")
    ax.set_ylabel("Attack Success Rate  ↓ better")
    ax.set_title("ASR vs over-refusal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "pareto_asr_vs_overrefusal.png", dpi=130)
    plt.close(fig)

    # 2) ROC over composite risk R.
    fpr, tpr = roc_curve(records)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot(fpr, tpr, color="#2266cc", label=f"R (AUROC={report['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — composite risk as harm detector")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "roc.png", dpi=130)
    plt.close(fig)

    # 3) Leave-one-gate-out ablation (AUROC).
    abl = report["ablation_auroc"]
    keys = list(abl.keys())
    vals = [abl[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bars = ax.bar(keys, vals, color=["#229955" if k == "full" else "#88aadd" for k in keys])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: composite-risk AUROC")
    for b, v in zip(bars, vals):
        if v == v:  # not nan
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ablation_auroc.png", dpi=130)
    plt.close(fig)

    # 4) Latency.
    lat = report["latency"]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.bar(["mean", "p50", "p95"], [lat["mean_ms"], lat["p50_ms"], lat["p95_ms"]], color="#8866cc")
    ax.set_ylabel("Latency per request (ms)")
    ax.set_title(f"Firewall latency ({report['n_total']} samples)")
    fig.tight_layout()
    fig.savefig(outdir / "latency.png", dpi=130)
    plt.close(fig)


def _write_summary(report, outdir: Path, dataset_desc: str, provider_mode: str):
    fw, raw = report["firewall"], report["raw"]
    lines = []
    lines.append("# Cognitive Firewall — Evaluation Summary\n")
    lines.append(f"- Provider: `{provider_mode}`")
    lines.append(f"- Dataset: {dataset_desc}\n")

    lines.append("## Headline\n")
    lines.append("| System | ASR (harmful) | Over-refusal (benign) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Raw model (no firewall) | {raw['asr']:.1%} | {raw['overrefusal']:.1%} |")
    lines.append(f"| **Cognitive Firewall** | **{fw['asr']:.1%}** | {fw['overrefusal']:.1%} |\n")

    lines.append("**ASR at matched over-refusal** (sweeping the firewall's risk threshold):\n")
    lines.append("| Target over-refusal | Threshold | ASR | Over-refusal | Feasible |")
    lines.append("|---:|---:|---:|---:|:--:|")
    for m in report["asr_at_matched_overrefusal"]:
        lines.append(
            f"| {m['target_overrefusal']:.1%} | {m['threshold']:.2f} | {m['asr']:.1%} | "
            f"{m['overrefusal']:.1%} | {'yes' if m['feasible'] else 'no'} |"
        )
    lines.append("")

    lines.append("## Detector quality (composite risk R)\n")
    lines.append(f"- AUROC: **{report['auroc']:.3f}**  |  AUPRC: {report['auprc']:.3f}")
    lines.append(f"- G3-only (moderation-style) AUROC: {report['g3_only_auroc']:.3f}")
    lines.append(
        f"- BLOCK as harm-detector: P={fw['precision']:.2f} R={fw['recall']:.2f} "
        f"F1={fw['f1']:.2f} acc={fw['accuracy']:.2f}\n"
    )

    lines.append("## Ablation — AUROC (leave-one-gate-out)\n")
    lines.append("| Config | AUROC |")
    lines.append("|---|---:|")
    for k, v in report["ablation_auroc"].items():
        lines.append(f"| {k} | {v:.3f} |")
    lines.append("")

    lines.append("## ASR by attack type (firewall vs raw)\n")
    lines.append("| Attack type | n | n_harmful | ASR raw | ASR firewall |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, v in report["by_attack_type"].items():
        lines.append(f"| {k} | {v['n']} | {v['n_harmful']} | {v['asr_raw']:.1%} | {v['asr_firewall']:.1%} |")
    lines.append("")

    lines.append("## Latency\n")
    lat = report["latency"]
    lines.append(f"- mean {lat['mean_ms']:.1f} ms | p50 {lat['p50_ms']:.1f} ms | p95 {lat['p95_ms']:.1f} ms\n")

    lines.append("![ASR vs over-refusal](pareto_asr_vs_overrefusal.png)")
    lines.append("![ROC](roc.png)")
    lines.append("![Ablation](ablation_auroc.png)")
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cognitive Firewall evaluation harness")
    ap.add_argument("--backend", default=None, help="auto|local|openai|heuristic (overrides CF_BACKEND)")
    ap.add_argument("--datasets", nargs="*", default=None, help="filter by source name(s)")
    ap.add_argument("--attack-types", nargs="*", default=None, help="filter by attack type(s)")
    ap.add_argument("--max-samples", type=int, default=None, help="cap per label")
    ap.add_argument("--full", action="store_true", help="attempt full HF datasets (fallback to bundled)")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    cfg = FirewallConfig.from_env()
    if args.backend:
        cfg.backend = args.backend
    cfg.dry_run = True  # scoring pass: full pipeline, no enforcement, output passed through

    fw = CognitiveFirewall(cfg)
    samples = datasets.load(
        sources=args.datasets, attack_types=args.attack_types,
        full=args.full, max_samples=args.max_samples,
    )
    desc = datasets.summarize(samples)
    print(f"[eval] backend={fw.provider.mode_name} | {desc}")

    records = build_records(fw, samples, HeuristicJudge())
    report = compute_report(records, cfg)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(records, outdir / "per_sample.csv")
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.no_plots:
        _make_plots(records, report, outdir)
    _write_summary(report, outdir, desc, fw.provider.mode_name)

    fwm, rawm = report["firewall"], report["raw"]
    print(f"[eval] ASR  raw={rawm['asr']:.1%} -> firewall={fwm['asr']:.1%}")
    print(f"[eval] Over-refusal  raw={rawm['overrefusal']:.1%} -> firewall={fwm['overrefusal']:.1%}")
    print(f"[eval] AUROC={report['auroc']:.3f}  AUPRC={report['auprc']:.3f}")
    print(f"[eval] wrote {outdir}/summary.md, per_sample.csv, metrics.json, *.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
