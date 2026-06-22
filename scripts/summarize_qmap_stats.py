"""Build a compact QMAP statistics pack from archived split/seed summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results/qmap_stats_pack"
FIG = ROOT / "paper/figures/qmap_benchmark_summary.png"

QMAP_BASELINES = {
    "ESM2 linear": {"full_ecoli": 0.360, "high_eff_ecoli": 0.160, "hc50": 0.070},
    "Witten & Witten 2019": {"full_ecoli": 0.510, "high_eff_ecoli": 0.220},
    "Cai et al. 2025": {"full_ecoli": 0.520, "high_eff_ecoli": 0.290},
}

RUNS = {
    "head_only": {
        42: "eval_results/qmap_jepa_head_finetune/summary.json",
        7: "eval_results/qmap_jepa_head_finetune_seed7_gpu2/summary.json",
        123: "eval_results/qmap_jepa_head_finetune_seed123_gpu1/summary.json",
    },
    "conditional": {
        42: "eval_results/qmap_jepa_conditional_seed42/summary.json",
        7: "eval_results/qmap_jepa_conditional_seed7/summary.json",
        123: "eval_results/qmap_jepa_conditional_seed123/summary.json",
    },
    "hc50_head": {
        42: "eval_results/qmap_jepa_hc50_head_finetune_seed42/summary.json",
        7: "eval_results/qmap_jepa_hc50_head_finetune_seed7/summary.json",
        123: "eval_results/qmap_jepa_hc50_head_finetune_seed123/summary.json",
    },
}

ENDPOINT_KEYS = {
    "full_ecoli": "full_ecoli_pearson",
    "high_eff_ecoli": "high_eff_ecoli_pearson",
    "hc50": "hc50_pearson",
    "hc50_head": "full_target_pearson",
}


def load(path: str) -> dict[str, Any]:
    with open(ROOT / path) as f:
        return json.load(f)


def values_for(summary: dict[str, Any], key: str) -> list[float]:
    vals = []
    for split in summary["splits"]:
        value = split.get(key)
        if value is not None:
            vals.append(float(value))
    return vals


def seed_summary(vals: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(vals)),
        "sd": float(stdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def summarize_runs() -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for run_name, seed_paths in RUNS.items():
        runs[run_name] = {}
        for seed, path in seed_paths.items():
            summary = load(path)
            row: dict[str, Any] = {"path": path, "splits": {}}
            for endpoint, key in ENDPOINT_KEYS.items():
                vals = values_for(summary, key)
                if vals:
                    row["splits"][endpoint] = vals
                    row[endpoint] = seed_summary(vals)
            runs[run_name][str(seed)] = row
    return runs


def seed_mean_table(runs: dict[str, Any]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    endpoint_by_run = {
        "head_only": ["full_ecoli", "high_eff_ecoli"],
        "conditional": ["full_ecoli", "high_eff_ecoli", "hc50"],
        "hc50_head": ["hc50_head"],
    }
    for run_name, endpoints in endpoint_by_run.items():
        table[run_name] = {}
        for endpoint in endpoints:
            vals = [seed_info[endpoint]["mean"] for seed_info in runs[run_name].values()]
            table[run_name][endpoint] = seed_summary(vals)
    return table


def paired_deltas(runs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    comparisons = [
        ("conditional_minus_head_full_ecoli", "conditional", "head_only", "full_ecoli", "full_ecoli"),
        ("conditional_minus_head_high_eff_ecoli", "conditional", "head_only", "high_eff_ecoli", "high_eff_ecoli"),
        ("conditional_minus_hc50_head", "conditional", "hc50_head", "hc50", "hc50_head"),
    ]
    for name, left_run, right_run, left_endpoint, right_endpoint in comparisons:
        deltas = []
        per_seed = {}
        for seed in ("42", "7", "123"):
            left = runs[left_run][seed]["splits"][left_endpoint]
            right = runs[right_run][seed]["splits"][right_endpoint]
            seed_deltas = [float(l - r) for l, r in zip(left, right)]
            per_seed[seed] = seed_summary(seed_deltas)
            deltas.extend(seed_deltas)
        out[name] = {"all_split_seed_deltas": seed_summary(deltas), "per_seed": per_seed}
    return out


def write_markdown(pack: dict[str, Any]) -> None:
    lines = ["# QMAP Statistics Pack", ""]
    lines.append("Source: archived QMAP split summaries under `eval_results/qmap_jepa_*`.")
    lines.append("Protocol: `qmap-benchmark==0.1.1`, five predefined homology-aware splits, seeds 42/7/123 for JEPA variants.")
    lines.append("")
    lines.append("## Seed Stability")
    lines.append("")
    lines.append("| Model | Endpoint | Seed mean PCC | Seed SD | Seed min | Seed max |")
    lines.append("|---|---|---:|---:|---:|---:|")
    labels = {
        "head_only": "JEPA head-only",
        "conditional": "JEPA conditional",
        "hc50_head": "JEPA HC50 head",
        "full_ecoli": "Full E. coli",
        "high_eff_ecoli": "High-eff. E. coli",
        "hc50": "HC50 shared",
        "hc50_head": "HC50 specific",
    }
    for run_name, endpoints in pack["seed_means"].items():
        for endpoint, stats in endpoints.items():
            lines.append(
                f"| {labels[run_name]} | {labels[endpoint]} | {stats['mean']:.4f} | "
                f"{stats['sd']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
            )
    lines.append("")
    lines.append("## Paired Split-Seed Deltas")
    lines.append("")
    lines.append("| Comparison | Mean delta | SD | Min | Max | Interpretation |")
    lines.append("|---|---:|---:|---:|---:|---|")
    interpretations = {
        "conditional_minus_head_full_ecoli": "conditional is slightly higher on full E. coli",
        "conditional_minus_head_high_eff_ecoli": "conditional is higher on high-efficiency E. coli",
        "conditional_minus_hc50_head": "shared conditional model is worse than HC50-specific head",
    }
    for name, item in pack["paired_deltas"].items():
        stats = item["all_split_seed_deltas"]
        lines.append(
            f"| {name} | {stats['mean']:.4f} | {stats['sd']:.4f} | "
            f"{stats['min']:.4f} | {stats['max']:.4f} | {interpretations[name]} |"
        )
    lines.append("")
    lines.append("Interpretation: QMAP is the strongest dry-lab evidence because it uses fixed homology-aware folds. The deltas support a bounded claim: shared conditional heads help E. coli MIC endpoints, especially high-efficiency E. coli, while HC50 remains better served by a task-specific head.")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def plot(pack: dict[str, Any]) -> None:
    seed = pack["seed_means"]
    labels = ["ESM2\nlinear", "Witten\n2019", "Cai\n2025", "JEPA\nhead", "JEPA\nconditional"]
    full_vals = [
        QMAP_BASELINES["ESM2 linear"]["full_ecoli"],
        QMAP_BASELINES["Witten & Witten 2019"]["full_ecoli"],
        QMAP_BASELINES["Cai et al. 2025"]["full_ecoli"],
        seed["head_only"]["full_ecoli"]["mean"],
        seed["conditional"]["full_ecoli"]["mean"],
    ]
    full_err = [0, 0, 0, seed["head_only"]["full_ecoli"]["sd"], seed["conditional"]["full_ecoli"]["sd"]]
    high_vals = [
        QMAP_BASELINES["ESM2 linear"]["high_eff_ecoli"],
        QMAP_BASELINES["Witten & Witten 2019"]["high_eff_ecoli"],
        QMAP_BASELINES["Cai et al. 2025"]["high_eff_ecoli"],
        seed["head_only"]["high_eff_ecoli"]["mean"],
        seed["conditional"]["high_eff_ecoli"]["mean"],
    ]
    high_err = [0, 0, 0, seed["head_only"]["high_eff_ecoli"]["sd"], seed["conditional"]["high_eff_ecoli"]["sd"]]
    hc_labels = ["ESM2\nlinear", "JEPA\nHC50 head", "JEPA\nconditional"]
    hc_vals = [
        QMAP_BASELINES["ESM2 linear"]["hc50"],
        seed["hc50_head"]["hc50_head"]["mean"],
        seed["conditional"]["hc50"]["mean"],
    ]
    hc_err = [0, seed["hc50_head"]["hc50_head"]["sd"], seed["conditional"]["hc50"]["sd"]]
    colors = ["#7A8793", "#7A8793", "#7A8793", "#1F77B4", "#2CA02C"]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), constrained_layout=True)
    panels = [
        (axes[0], labels, full_vals, full_err, colors, "Full E. coli MIC", 0.6),
        (axes[1], labels, high_vals, high_err, colors, "High-efficiency E. coli MIC", 0.45),
        (axes[2], hc_labels, hc_vals, hc_err, ["#7A8793", "#D62728", "#2CA02C"], "HC50 hemolysis", 0.4),
    ]
    for ax, xlabels, vals, errs, bar_colors, title, ylim in panels:
        x = np.arange(len(xlabels))
        ax.bar(x, vals, yerr=errs, color=bar_colors, width=0.72, capsize=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylim(0, ylim)
        ax.grid(axis="y", color="#E6E8EB", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for patch, value in zip(ax.patches, vals):
            ax.text(patch.get_x() + patch.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Mean Pearson r")
    fig.suptitle("QMAP homology-aware benchmark performance", fontsize=12, fontweight="bold")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=300, bbox_inches="tight")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = summarize_runs()
    pack = {
        "experiment_id": "qmap_stats_pack",
        "protocol": "qmap-benchmark==0.1.1; five predefined homology-aware splits",
        "runs": runs,
        "seed_means": seed_mean_table(runs),
        "paired_deltas": paired_deltas(runs),
        "qmap_baselines": QMAP_BASELINES,
    }
    with open(OUT / "metrics.json", "w") as f:
        json.dump(pack, f, indent=2)
    with open(OUT / "manifest.json", "w") as f:
        json.dump({
            "experiment_id": "qmap_stats_pack",
            "metrics": "eval_results/qmap_stats_pack/metrics.json",
            "summary": "eval_results/qmap_stats_pack/SUMMARY.md",
            "figure": "paper/figures/qmap_benchmark_summary.png",
            "status": "formal_artifact",
        }, f, indent=2)
    write_markdown(pack)
    plot(pack)
    print(f"wrote {OUT / 'metrics.json'}")
    print(f"wrote {OUT / 'SUMMARY.md'}")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
