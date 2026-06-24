"""
Plot predicted vs actual MIC scatter plots from fewshot_v2 predictions.json.

For each model, creates:
  eval_results/plots/scatter/{model}/
    by_k/         -- one subplot per species pair, colored by k value
    by_pair/      -- grid: rows=source species, cols=target species
    overview_k0.png   -- all 30 pairs, zero-shot only
    overview_k100.png -- all 30 pairs, k=100 only

Usage:
    uv run python scripts/plot_scatter_pred_actual.py --model jepa
    uv run python scripts/plot_scatter_pred_actual.py --model jepa esm2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES  = PROJECT_ROOT / "eval_results"
PLOT = RES / "plots" / "scatter"

SPECIES = [
    "E. coli",
    "S. aureus",
    "P. aeruginosa",
    "B. subtilis",
    "S. typhimurium",
    "M. luteus",
]
SP_SHORT = {
    "E. coli": "Eco",
    "S. aureus": "Sau",
    "P. aeruginosa": "Pae",
    "B. subtilis": "Bsu",
    "S. typhimurium": "Sty",
    "M. luteus": "Mlu",
}

K_VALUES  = [0, 5, 10, 20, 50, 100]
K_PALETTE = {
    0:   "#888888",
    5:   "#4C72B0",
    10:  "#55A868",
    20:  "#C44E52",
    50:  "#8172B2",
    100: "#CCB974",
}

MODEL_LABELS = {
    "jepa":      "RAMP",
    "esm2":      "ESM2-150M",
    "esm2_650m": "ESM2-650M",
    "mlm":       "MLM",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def load_predictions(model: str) -> dict:
    p = RES / "fewshot_v2" / model / "predictions.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_metrics(model: str) -> dict:
    p = RES / "fewshot_v2" / model / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(PROJECT_ROOT)}")


def get_pooled_preds(pred_dict: dict, pair_key: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Pool predictions across all seeds for a given pair and k."""
    all_pred, all_actual = [], []
    pair_data = pred_dict.get(pair_key, {})
    for seed_data in pair_data.values():
        entry = seed_data.get(str(k))
        if entry:
            all_pred.extend(entry["pred"])
            all_actual.extend(entry["actual"])
    return np.array(all_pred), np.array(all_actual)


def pearson_str(pred, actual) -> str:
    if len(pred) < 3:
        return "r=—"
    r, _ = pearsonr(pred, actual)
    return f"r={r:.2f}"


# ── plot 1: by_k — one figure per pair, all k values as colors ───────────────

def plot_by_k(pred_dict: dict, model: str, out_dir: Path):
    pairs = [(s, t) for s in SPECIES for t in SPECIES if s != t]
    available = [p for p in pairs if f"{p[0]}→{p[1]}" in pred_dict]
    if not available:
        return

    for src, tgt in available:
        pair_key = f"{src}→{tgt}"
        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        for k in K_VALUES:
            pred, actual = get_pooled_preds(pred_dict, pair_key, k)
            if len(pred) == 0:
                continue
            r_str = pearson_str(pred, actual)
            ax.scatter(actual, pred, alpha=0.4, s=12,
                       color=K_PALETTE[k], label=f"k={k} ({r_str})",
                       rasterized=True)

        # diagonal
        all_vals = []
        for k in K_VALUES:
            pred, actual = get_pooled_preds(pred_dict, pair_key, k)
            if len(actual) > 0:
                all_vals.extend(actual.tolist() + pred.tolist())
        if all_vals:
            lo, hi = min(all_vals) - 0.3, max(all_vals) + 0.3
            ax.plot([lo, hi], [lo, hi], "--", color="#aaa", linewidth=1.0, zorder=0)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        ax.set_xlabel("Actual log₂(MIC)", fontsize=10)
        ax.set_ylabel("Predicted log₂(MIC)", fontsize=10)
        src_s = SP_SHORT.get(src, src)
        tgt_s = SP_SHORT.get(tgt, tgt)
        ax.set_title(f"{MODEL_LABELS.get(model, model)}: {src_s}→{tgt_s}", fontsize=11)
        ax.legend(fontsize=7, frameon=False, loc="upper left",
                  markerscale=1.5)
        ax.set_aspect("equal")
        fname = f"{src_s}_to_{tgt_s}.png"
        savefig(fig, out_dir / "by_k" / fname)


# ── plot 2: overview grid — all pairs for one k value ────────────────────────

def plot_overview_grid(pred_dict: dict, model: str, k: int, out_dir: Path):
    n = len(SPECIES)
    fig, axes = plt.subplots(n, n, figsize=(2.8 * n, 2.6 * n),
                             sharex=False, sharey=False)

    for i, src in enumerate(SPECIES):
        for j, tgt in enumerate(SPECIES):
            ax = axes[i][j]
            if src == tgt:
                ax.text(0.5, 0.5, SP_SHORT.get(src, src),
                        ha="center", va="center", fontsize=10, fontweight="medium",
                        transform=ax.transAxes,
                        color="white")
                ax.set_facecolor("#444")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            pair_key = f"{src}→{tgt}"
            pred, actual = get_pooled_preds(pred_dict, pair_key, k)

            if len(pred) == 0:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="#aaa", fontsize=12)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            r, _ = pearsonr(pred, actual)
            color = K_PALETTE.get(k, "#4C72B0")
            ax.scatter(actual, pred, alpha=0.35, s=6, color=color, rasterized=True)
            lo = min(actual.min(), pred.min()) - 0.2
            hi = max(actual.max(), pred.max()) + 0.2
            ax.plot([lo, hi], [lo, hi], "--", color="#aaa", linewidth=0.8, zorder=0)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(f"r={r:.2f}", fontsize=8, pad=2)
            ax.tick_params(labelsize=6)

        # row / col labels
        axes[i][0].set_ylabel(SP_SHORT.get(SPECIES[i], SPECIES[i]),
                               fontsize=9, rotation=90, va="center")

    for j, tgt in enumerate(SPECIES):
        axes[-1][j].set_xlabel(SP_SHORT.get(tgt, tgt), fontsize=9)

    fig.suptitle(
        f"{MODEL_LABELS.get(model, model)} — pred vs actual, k={k} (pooled 3 seeds)",
        fontsize=13, y=1.01,
    )
    savefig(fig, out_dir / f"overview_k{k}.png")


# ── plot 3: k=0 vs k=100 side-by-side for one pair ───────────────────────────

def plot_k0_vs_k100(pred_dict: dict, model: str, out_dir: Path):
    pairs = [(s, t) for s in SPECIES for t in SPECIES if s != t]
    available = [(s, t) for s, t in pairs if f"{s}→{t}" in pred_dict]
    if not available:
        return

    ncols = 4
    nrows = int(np.ceil(len(available) / ncols))
    # 2 panels per pair (k=0 and k=100)
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(3.5 * ncols, 3.2 * nrows * 2))
    axes = np.array(axes)

    for idx, (src, tgt) in enumerate(available):
        row = (idx // ncols) * 2
        col = idx % ncols
        pair_key = f"{src}→{tgt}"
        src_s = SP_SHORT.get(src, src)
        tgt_s = SP_SHORT.get(tgt, tgt)

        for ki, k in enumerate([0, 100]):
            ax = axes[row + ki][col]
            pred, actual = get_pooled_preds(pred_dict, pair_key, k)
            if len(pred) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="#aaa")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            r, _ = pearsonr(pred, actual)
            color = K_PALETTE[k]
            ax.scatter(actual, pred, alpha=0.4, s=10, color=color, rasterized=True)
            lo = min(actual.min(), pred.min()) - 0.2
            hi = max(actual.max(), pred.max()) + 0.2
            ax.plot([lo, hi], [lo, hi], "--", color="#aaa", linewidth=0.8, zorder=0)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            title = f"{src_s}→{tgt_s}  k={k}\nr={r:.2f} (n={len(pred)})"
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel("Predicted log₂(MIC)", fontsize=7)
            if row + ki == axes.shape[0] - 1:
                ax.set_xlabel("Actual log₂(MIC)", fontsize=7)

    # hide unused axes
    for idx in range(len(available), nrows * ncols):
        row = (idx // ncols) * 2
        col = idx % ncols
        for ki in range(2):
            if row + ki < axes.shape[0]:
                axes[row + ki][col].set_visible(False)

    fig.suptitle(
        f"{MODEL_LABELS.get(model, model)} — zero-shot (k=0) vs k=100",
        fontsize=13, y=1.005,
    )
    savefig(fig, out_dir / "k0_vs_k100_all_pairs.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+",
                        default=["jepa"],
                        choices=["jepa", "esm2", "esm2_650m", "mlm"])
    args = parser.parse_args()

    for model in args.model:
        print(f"\n=== {model} ===")
        pred_dict = load_predictions(model)
        if not pred_dict:
            print(f"  No predictions found at eval_results/fewshot_v2/{model}/predictions.json — skipping")
            continue

        out_dir = PLOT / model
        print(f"  {len(pred_dict)} pairs with predictions")

        print("  [1/3] per-pair scatter by k...")
        plot_by_k(pred_dict, model, out_dir)

        print("  [2/3] overview grids...")
        for k in [0, 100]:
            plot_overview_grid(pred_dict, model, k, out_dir)

        print("  [3/3] k=0 vs k=100 side-by-side...")
        plot_k0_vs_k100(pred_dict, model, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
