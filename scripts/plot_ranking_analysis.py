"""
Ranking-based analysis plots for cross-species MIC transfer.

Figures:
  1. spearman_bar_k0.png   -- Spearman rho zero-shot, all models, all pairs
  2. rank_rank_scatter.png -- rank(actual) vs rank(predicted), zero-shot, models side-by-side
  3. spearman_curve.png    -- Spearman rho vs k (learning curve), models x pairs
  4. precision_at_k.png    -- Precision@top-20%, zero-shot vs k=100

Usage:
    uv run python scripts/plot_ranking_analysis.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES  = PROJECT_ROOT / "eval_results" / "fewshot_v2"
PLOT = PROJECT_ROOT / "eval_results" / "plots" / "ranking"
PLOT.mkdir(parents=True, exist_ok=True)

MODELS = ["jepa", "esm2", "mlm"]          # esm2_650m added when available
MODEL_LABELS = {"jepa": "RAMP", "esm2": "ESM2-150M",
                "mlm": "MLM", "esm2_650m": "ESM2-650M"}
MODEL_COLORS = {"jepa": "#534AB7", "esm2": "#1D9E75",
                "mlm": "#D85A30", "esm2_650m": "#378ADD"}

PAIRS = [
    ("E. coli",       "S. aureus"),
    ("E. coli",       "P. aeruginosa"),
    ("S. aureus",     "E. coli"),
    ("S. aureus",     "P. aeruginosa"),
    ("P. aeruginosa", "E. coli"),
]
PAIR_SHORT = {
    "E. coli→S. aureus":       "Eco→Sau",
    "E. coli→P. aeruginosa":   "Eco→Pae",
    "S. aureus→E. coli":       "Sau→Eco",
    "S. aureus→P. aeruginosa": "Sau→Pae",
    "P. aeruginosa→E. coli":   "Pae→Eco",
}
K_VALUES = [0, 5, 10, 20, 50, 100]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


def load_preds(model):
    p = RES / model / "predictions.json"
    return json.loads(p.read_text()) if p.exists() else {}

def load_metrics(model):
    p = RES / model / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

def pool(preds, pair, k):
    pp, aa = [], []
    for sd in preds.get(pair, {}).values():
        e = sd.get(str(k))
        if e: pp.extend(e["pred"]); aa.extend(e["actual"])
    return np.array(pp), np.array(aa)

def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(PROJECT_ROOT)}")


# ── 1. Spearman bar chart at k=0 ─────────────────────────────────────────────

def plot_spearman_bar(all_preds, k=0):
    available = [m for m in MODELS if m in all_preds]
    rows = []
    for model in available:
        for src, tgt in PAIRS:
            pk = f"{src}→{tgt}"
            pred, actual = pool(all_preds[model], pk, k)
            if len(pred) < 3: continue
            rho, _ = spearmanr(pred, actual)
            rows.append({"model": MODEL_LABELS[model],
                         "pair":  PAIR_SHORT.get(pk, pk),
                         "spearman": rho,
                         "color": MODEL_COLORS[model]})
    df = pd.DataFrame(rows)

    pairs_ord  = [PAIR_SHORT[f"{s}→{t}"] for s,t in PAIRS if PAIR_SHORT[f"{s}→{t}"] in df["pair"].values]
    models_ord = [MODEL_LABELS[m] for m in available]
    n_pairs, n_models = len(pairs_ord), len(models_ord)
    x = np.arange(n_pairs)
    w = 0.75 / n_models

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(available):
        lbl  = MODEL_LABELS[model]
        col  = MODEL_COLORS[model]
        vals = [df[(df["model"]==lbl) & (df["pair"]==p)]["spearman"].values
                for p in pairs_ord]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - n_models/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w * 0.92, color=col, alpha=0.88,
                      label=lbl, zorder=3)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.012,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color=col, fontweight="medium")

    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(pairs_ord, fontsize=10)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_ylim(-0.05, ax.get_ylim()[1] + 0.06)
    title = "Zero-shot" if k == 0 else f"k={k} warmstart"
    ax.set_title(f"Cross-species ranking ({title}) — Spearman ρ", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    savefig(fig, PLOT / f"spearman_bar_k{k}.png")


# ── 2. Rank-rank scatter (zero-shot) ─────────────────────────────────────────

def plot_rank_rank(all_preds, k=0):
    available = [m for m in MODELS if m in all_preds]
    n_pairs  = len(PAIRS)
    n_models = len(available)

    fig, axes = plt.subplots(n_pairs, n_models,
                             figsize=(3.4 * n_models, 3.2 * n_pairs),
                             squeeze=False)

    for row, (src, tgt) in enumerate(PAIRS):
        pk  = f"{src}→{tgt}"
        lbl = PAIR_SHORT.get(pk, pk)
        for col, model in enumerate(available):
            ax    = axes[row][col]
            color = MODEL_COLORS[model]
            pred, actual = pool(all_preds[model], pk, k)

            if len(pred) < 3:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="#aaa", fontsize=14)
                ax.set_xticks([]); ax.set_yticks([]); continue

            # convert to percentile ranks (0–100)
            n = len(actual)
            rank_actual = (np.argsort(np.argsort(actual)) / n * 100)
            rank_pred   = (np.argsort(np.argsort(pred))   / n * 100)
            rho, _ = spearmanr(pred, actual)

            ax.scatter(rank_actual, rank_pred, alpha=0.25, s=7,
                       color=color, rasterized=True)
            ax.plot([0, 100], [0, 100], "--", color="#aaa", lw=0.9, zorder=0)
            ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            ax.text(0.06, 0.92, f"ρ = {rho:.3f}",
                    transform=ax.transAxes, fontsize=9.5, fontweight="medium",
                    color=color, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

            if row == 0:
                ax.set_title(MODEL_LABELS.get(model, model), fontsize=11,
                             fontweight="medium", color=color, pad=5)
            if col == 0:
                ax.set_ylabel(f"{lbl}\nPred. percentile rank", fontsize=8.5)
            if row == n_pairs - 1:
                ax.set_xlabel("Actual percentile rank", fontsize=8.5)

    k_lbl = "Zero-shot (k=0)" if k == 0 else f"k={k} warmstart"
    fig.suptitle(f"Rank-rank scatter — {k_lbl}  ·  pooled 3 seeds",
                 fontsize=13, y=1.01)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    fig.savefig(PLOT / f"rank_rank_k{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {(PLOT / f'rank_rank_k{k}.png').relative_to(PROJECT_ROOT)}")


# ── 3. Spearman learning curve (all k) ───────────────────────────────────────

def plot_spearman_curve(all_preds):
    available = [m for m in MODELS if m in all_preds]
    n_pairs = len(PAIRS)
    ncols   = min(3, n_pairs)
    nrows   = int(np.ceil(n_pairs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    for idx, (src, tgt) in enumerate(PAIRS):
        ax  = axes[idx]
        pk  = f"{src}→{tgt}"
        lbl = PAIR_SHORT.get(pk, pk)
        for model in available:
            xs, ys = [], []
            for k in K_VALUES:
                pred, actual = pool(all_preds[model], pk, k)
                if len(pred) < 3: continue
                rho, _ = spearmanr(pred, actual)
                xs.append(k); ys.append(rho)
            if xs:
                ax.plot(xs, ys, marker="o", ms=5, lw=1.8,
                        color=MODEL_COLORS[model],
                        label=MODEL_LABELS[model])

        ax.axhline(0, color="#bbb", lw=0.8, ls="--")
        ax.set_title(lbl, fontsize=11, fontweight="medium")
        ax.set_xlabel("k (few-shot examples)", fontsize=9)
        ax.set_ylabel("Spearman ρ", fontsize=9)
        ax.set_xticks(K_VALUES)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.35)

    for ax in axes[n_pairs:]: ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=len(available),
               frameon=False, fontsize=9)
    fig.suptitle("Spearman ρ vs k (warmstart protocol)", y=1.05, fontsize=12)
    savefig(fig, PLOT / "spearman_curve.png")


# ── 4. Precision@top-20% ─────────────────────────────────────────────────────

def precision_at_top(actual, pred, frac=0.2):
    n    = len(actual)
    topn = max(1, int(n * frac))
    true_top = set(np.argsort(actual)[-topn:])
    pred_top = set(np.argsort(pred)[-topn:])
    return len(true_top & pred_top) / topn

def plot_precision_at_top(all_preds, frac=0.20):
    available = [m for m in MODELS if m in all_preds]
    rows = []
    for model in available:
        for src, tgt in PAIRS:
            pk = f"{src}→{tgt}"
            for k in [0, 100]:
                pred, actual = pool(all_preds[model], pk, k)
                if len(pred) < 3: continue
                p_at_k = precision_at_top(actual, pred, frac)
                rows.append({"model": MODEL_LABELS[model],
                             "pair":  PAIR_SHORT.get(pk, pk),
                             "k":     k, "precision": p_at_k})
    df = pd.DataFrame(rows)
    if df.empty: return

    pairs_ord = [PAIR_SHORT[f"{s}→{t}"] for s,t in PAIRS
                 if PAIR_SHORT[f"{s}→{t}"] in df["pair"].values]
    x = np.arange(len(pairs_ord))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, k in zip(axes, [0, 100]):
        sub  = df[df["k"] == k]
        n_m  = len(available)
        w    = 0.75 / n_m
        for i, model in enumerate(available):
            lbl  = MODEL_LABELS[model]
            col  = MODEL_COLORS[model]
            vals = [sub[(sub["model"]==lbl) & (sub["pair"]==p)]["precision"].values
                    for p in pairs_ord]
            vals = [v[0] if len(v) else np.nan for v in vals]
            offset = (i - n_m/2 + 0.5) * w
            ax.bar(x + offset, vals, w*0.92, color=col, alpha=0.88, label=lbl, zorder=3)

        random_baseline = frac
        ax.axhline(random_baseline, color="#999", lw=1.2, ls="--",
                   label=f"Random ({frac*100:.0f}%)")
        ax.set_xticks(x); ax.set_xticklabels(pairs_ord, fontsize=9)
        ax.set_ylabel(f"Precision@top-{int(frac*100)}%", fontsize=10)
        k_lbl = "Zero-shot (k=0)" if k == 0 else f"k={k} warmstart"
        ax.set_title(k_lbl, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.35, zorder=0)
        if k == 0:
            ax.legend(fontsize=8, frameon=False)

    fig.suptitle(f"Precision@top-{int(frac*100)}% — fraction of true top peptides recovered",
                 fontsize=12)
    savefig(fig, PLOT / "precision_at_top20.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    available = [m for m in MODELS if (RES / m / "predictions.json").exists()]
    print(f"Models with predictions: {available}")
    all_preds = {m: load_preds(m) for m in available}

    print("\n[1/4] Spearman bar (k=0)...")
    plot_spearman_bar(all_preds, k=0)

    print("[2/4] Spearman bar (k=100)...")
    plot_spearman_bar(all_preds, k=100)

    print("[3/4] Rank-rank scatter (k=0)...")
    plot_rank_rank(all_preds, k=0)

    print("[4/4] Spearman learning curve...")
    plot_spearman_curve(all_preds)

    print("[5/5] Precision@top-20%...")
    plot_precision_at_top(all_preds)

    print(f"\nAll plots → {PLOT.relative_to(PROJECT_ROOT)}/")

if __name__ == "__main__":
    main()
