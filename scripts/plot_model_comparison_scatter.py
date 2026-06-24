"""
Scatter plots: all models side-by-side for each species pair.
Two figures:
  1. k=0 (zero-shot): 5 pairs × 4 models grid
  2. k=100 (warmstart): 5 pairs × 4 models grid
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES  = PROJECT_ROOT / "eval_results" / "fewshot_v2"
PLOT = PROJECT_ROOT / "eval_results" / "plots" / "scatter"

MODELS = ["jepa", "esm2", "mlm", "esm2_650m"]
MODEL_LABELS = {
    "jepa":      "RAMP",
    "esm2":      "ESM2-150M",
    "mlm":       "MLM",
    "esm2_650m": "ESM2-650M",
}
MODEL_COLORS = {
    "jepa":      "#534AB7",
    "esm2":      "#1D9E75",
    "mlm":       "#D85A30",
    "esm2_650m": "#378ADD",
}
PAIRS = [
    ("E. coli",       "S. aureus"),
    ("E. coli",       "P. aeruginosa"),
    ("S. aureus",     "E. coli"),
    ("S. aureus",     "P. aeruginosa"),
    ("P. aeruginosa", "E. coli"),
]
PAIR_LABELS = {
    f"{s}→{t}": f"{s.split('.')[0].strip()} → {t.split('.')[0].strip()}"
    for s, t in PAIRS
}
SEEDS = [42, 123, 7]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)


def load_preds(model: str) -> dict:
    p = RES / model / "predictions.json"
    return json.loads(p.read_text()) if p.exists() else {}


def pool(preds: dict, pair: str, k: int):
    all_p, all_a = [], []
    for sd in preds.get(pair, {}).values():
        e = sd.get(str(k))
        if e:
            all_p.extend(e["pred"]); all_a.extend(e["actual"])
    return np.array(all_p), np.array(all_a)


def make_grid(k: int, out_path: Path):
    available_models = [m for m in MODELS if (RES / m / "predictions.json").exists()]
    all_preds = {m: load_preds(m) for m in available_models}

    n_pairs  = len(PAIRS)
    n_models = len(available_models)

    fig, axes = plt.subplots(n_pairs, n_models,
                             figsize=(3.6 * n_models, 3.4 * n_pairs),
                             squeeze=False)

    for row, (src, tgt) in enumerate(PAIRS):
        pair_key = f"{src}→{tgt}"
        pair_lbl = PAIR_LABELS.get(pair_key, pair_key)

        for col, model in enumerate(available_models):
            ax = axes[row][col]
            pred, actual = pool(all_preds[model], pair_key, k)
            color = MODEL_COLORS[model]

            if len(pred) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="#aaa", fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                r, _ = pearsonr(pred, actual)
                ax.scatter(actual, pred, alpha=0.3, s=8, color=color, rasterized=True)
                lo = min(actual.min(), pred.min()) - 0.3
                hi = max(actual.max(), pred.max()) + 0.3
                ax.plot([lo, hi], [lo, hi], "--", color="#aaa", lw=0.9, zorder=0)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=7)

                # r annotation top-left
                ax.text(0.06, 0.92, f"r = {r:.3f}",
                        transform=ax.transAxes, fontsize=9.5, fontweight="medium",
                        color=color, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

            # column headers (top row only)
            if row == 0:
                ax.set_title(MODEL_LABELS.get(model, model), fontsize=11,
                             fontweight="medium", color=color, pad=6)

            # row labels (left col only)
            if col == 0:
                ax.set_ylabel(pair_lbl + "\n\nPredicted log₂(MIC)", fontsize=8.5)
            else:
                ax.set_ylabel("")

            if row == n_pairs - 1:
                ax.set_xlabel("Actual log₂(MIC)", fontsize=8.5)
            else:
                ax.set_xlabel("")

    protocol = "Zero-shot (k=0)" if k == 0 else f"Warmstart fine-tune (k={k})"
    fig.suptitle(f"Predicted vs Actual MIC — {protocol}  ·  pooled 3 seeds",
                 fontsize=13, y=1.01)
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


def main():
    print("Generating comparison scatter grids...")
    make_grid(k=0,   out_path=PLOT / "comparison" / "all_models_k0.png")
    make_grid(k=100, out_path=PLOT / "comparison" / "all_models_k100.png")
    print("Done.")

if __name__ == "__main__":
    main()
