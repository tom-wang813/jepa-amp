"""
Plot few-shot transfer results across all three experiment types.

Outputs (seaborn, publication-quality):
  eval_results/plots/bact_emb/          -- bact_emb JEPA learning curves + scatter
  eval_results/plots/cold_cross_species/ -- cold protocol learning curves + scatter
  eval_results/plots/warmstart/          -- warmstart learning curves + scatter
  eval_results/plots/comparison/         -- cold vs warmstart, model vs model

Usage:
    uv run python scripts/plot_fewshot_results.py
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

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
RES     = ROOT / "eval_results"
BACT    = RES / "fewshot_bact_emb_jepa" / "metrics.json"
COLD    = RES / "fewshot_cross_species"  / "metrics.json"
WARM    = RES / "fewshot_cross_species_warmstart" / "metrics.json"
PLOT    = RES / "plots"

K_VALUES = [0, 5, 10, 20, 50, 100]

# ── style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
MODEL_PALETTE = {
    "jepa":      "#534AB7",
    "esm2":      "#1D9E75",
    "esm2_650m": "#378ADD",
    "mlm":       "#D85A30",
}
MODEL_LABELS = {
    "jepa":      "RAMP",
    "esm2":      "ESM2-150M",
    "esm2_650m": "ESM2-650M",
    "mlm":       "MLM",
}
PAIR_SHORT = {
    "E. coli→S. aureus":       "Eco→Sau",
    "E. coli→P. aeruginosa":   "Eco→Pae",
    "S. aureus→E. coli":       "Sau→Eco",
    "S. aureus→P. aeruginosa": "Sau→Pae",
    "P. aeruginosa→E. coli":   "Pae→Eco",
    "P. aeruginosa→S. aureus": "Pae→Sau",
}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_single_model(path: Path, model_name: str) -> pd.DataFrame:
    """Load bact_emb-style json: {pair: {seed: {k: metrics}}}."""
    if not path.exists():
        return pd.DataFrame()
    d = json.loads(path.read_text())
    rows = []
    for pair, seeds in d.items():
        for seed, shots in seeds.items():
            for k_str, m in shots.items():
                rows.append({
                    "model": model_name,
                    "pair":  pair,
                    "pair_short": PAIR_SHORT.get(pair, pair),
                    "seed":  int(seed),
                    "k":     int(k_str),
                    "pearson":  m["pearson"],
                    "spearman": m["spearman"],
                    "rmse":     m["rmse"],
                })
    return pd.DataFrame(rows)


def load_multi_model(path: Path) -> pd.DataFrame:
    """Load cross_species-style json: {model: {pair: {seed: {k: metrics}}}}."""
    if not path.exists():
        return pd.DataFrame()
    d = json.loads(path.read_text())
    rows = []
    for model, pairs in d.items():
        for pair, seeds in pairs.items():
            for seed, shots in seeds.items():
                for k_str, m in shots.items():
                    rows.append({
                        "model": model,
                        "pair":  pair,
                        "pair_short": PAIR_SHORT.get(pair, pair),
                        "seed":  int(seed),
                        "k":     int(k_str),
                        "pearson":  m["pearson"],
                        "spearman": m["spearman"],
                        "rmse":     m["rmse"],
                    })
    return pd.DataFrame(rows)


# ── helpers ───────────────────────────────────────────────────────────────────

def agg_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over seeds: mean ± std per (model, pair, k)."""
    return (
        df.groupby(["model", "pair", "pair_short", "k"])["pearson"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "pearson_mean", "std": "pearson_std"})
    )


def savefig(fig, path: Path, tight: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(ROOT)}")


# ── plot 1: learning curves (line + CI band) ──────────────────────────────────

def plot_learning_curves(df: pd.DataFrame, title: str, out_dir: Path,
                         fname: str = "learning_curves.png"):
    if df.empty:
        print(f"  [skip] {fname} — no data")
        return

    agg = agg_mean_std(df)
    pairs = sorted(agg["pair"].unique())
    models = [m for m in MODEL_PALETTE if m in agg["model"].unique()]

    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    for ax, pair in zip(axes, pairs):
        sub = agg[agg["pair"] == pair]
        for model in models:
            m_sub = sub[sub["model"] == model].sort_values("k")
            if m_sub.empty:
                continue
            color = MODEL_PALETTE[model]
            label = MODEL_LABELS.get(model, model)
            ax.plot(m_sub["k"], m_sub["pearson_mean"],
                    marker="o", markersize=5, linewidth=1.8,
                    color=color, label=label)
            ax.fill_between(
                m_sub["k"],
                m_sub["pearson_mean"] - m_sub["pearson_std"],
                m_sub["pearson_mean"] + m_sub["pearson_std"],
                alpha=0.15, color=color,
            )
        ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
        ax.set_title(PAIR_SHORT.get(pair, pair), fontsize=11, fontweight="medium")
        ax.set_xlabel("k (few-shot examples)", fontsize=9)
        ax.set_ylabel("Pearson r", fontsize=9)
        ax.set_xticks(K_VALUES)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.4)

    # hide unused axes
    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=len(models),
               frameon=False, fontsize=9)
    fig.suptitle(title, y=1.05, fontsize=12, fontweight="medium")
    savefig(fig, out_dir / fname)


# ── plot 2: scatter — each dot = one (pair, k), x=model_a, y=model_b ──────────

def plot_model_scatter(df: pd.DataFrame, model_a: str, model_b: str,
                       out_dir: Path, fname: str = "scatter.png",
                       title: str = ""):
    if df.empty:
        return
    agg = agg_mean_std(df)
    sub_a = agg[agg["model"] == model_a][["pair_short", "k", "pearson_mean"]].rename(columns={"pearson_mean": "a"})
    sub_b = agg[agg["model"] == model_b][["pair_short", "k", "pearson_mean"]].rename(columns={"pearson_mean": "b"})
    merged = sub_a.merge(sub_b, on=["pair_short", "k"])
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    palette = sns.color_palette("tab10", n_colors=merged["pair_short"].nunique())
    pair_colors = {p: c for p, c in zip(sorted(merged["pair_short"].unique()), palette)}

    for pair, group in merged.groupby("pair_short"):
        ax.scatter(group["a"], group["b"], color=pair_colors[pair],
                   label=pair, s=60, alpha=0.85, edgecolors="white", linewidths=0.4)
        # annotate k values
        for _, row in group.iterrows():
            ax.annotate(f"k={int(row['k'])}", (row["a"], row["b"]),
                        fontsize=6, alpha=0.6,
                        xytext=(3, 3), textcoords="offset points")

    lim_min = min(merged["a"].min(), merged["b"].min()) - 0.05
    lim_max = max(merged["a"].max(), merged["b"].max()) + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            "--", color="#aaa", linewidth=1.0, label="y=x")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel(f"{MODEL_LABELS.get(model_a, model_a)} Pearson r", fontsize=10)
    ax.set_ylabel(f"{MODEL_LABELS.get(model_b, model_b)} Pearson r", fontsize=10)
    ax.set_title(title or f"{MODEL_LABELS.get(model_a, model_a)} vs {MODEL_LABELS.get(model_b, model_b)}",
                 fontsize=11)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_aspect("equal")
    savefig(fig, out_dir / fname)


# ── plot 3: cold vs warmstart at each k ───────────────────────────────────────

def plot_cold_vs_warm(cold_df: pd.DataFrame, warm_df: pd.DataFrame,
                      model: str, out_dir: Path):
    if cold_df.empty or warm_df.empty:
        return
    c = agg_mean_std(cold_df[cold_df["model"] == model])[["pair_short", "k", "pearson_mean"]].rename(columns={"pearson_mean": "cold"})
    w = agg_mean_std(warm_df[warm_df["model"] == model])[["pair_short", "k", "pearson_mean"]].rename(columns={"pearson_mean": "warm"})
    merged = c.merge(w, on=["pair_short", "k"])
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    palette = sns.color_palette("tab10", n_colors=merged["pair_short"].nunique())
    pair_colors = {p: c for p, c in zip(sorted(merged["pair_short"].unique()), palette)}

    for pair, group in merged.groupby("pair_short"):
        ax.scatter(group["cold"], group["warm"], color=pair_colors[pair],
                   label=pair, s=60, alpha=0.85, edgecolors="white", linewidths=0.4)
        for _, row in group.iterrows():
            ax.annotate(f"k={int(row['k'])}", (row["cold"], row["warm"]),
                        fontsize=6, alpha=0.6,
                        xytext=(3, 3), textcoords="offset points")

    all_vals = pd.concat([merged["cold"], merged["warm"]])
    lim_min = all_vals.min() - 0.05
    lim_max = all_vals.max() + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#aaa", linewidth=1.0)
    ax.fill_between([lim_min, lim_max], [lim_min, lim_min], [lim_max, lim_max],
                    where=np.ones(2, dtype=bool), alpha=0.04, color="#378ADD",
                    label="warmstart better (above)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Cold-start Pearson r", fontsize=10)
    ax.set_ylabel("Warmstart Pearson r", fontsize=10)
    ax.set_title(f"{MODEL_LABELS.get(model, model)}: cold vs warmstart", fontsize=11)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_aspect("equal")
    savefig(fig, out_dir / f"cold_vs_warm_{model}.png")


# ── plot 4: bar chart at fixed k — all models side by side ────────────────────

def plot_bar_at_k(df: pd.DataFrame, k: int, out_dir: Path,
                  fname: str = "bar.png", title: str = ""):
    if df.empty:
        return
    sub = df[df["k"] == k].copy()
    if sub.empty:
        return

    models = [m for m in MODEL_PALETTE if m in sub["model"].unique()]
    agg = (sub.groupby(["model", "pair_short"])["pearson"]
           .agg(["mean", "std"]).reset_index()
           .rename(columns={"mean": "pearson_mean", "std": "pearson_std"}))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(agg["pair_short"].nunique())
    pairs_ord = sorted(agg["pair_short"].unique())
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        m_sub = agg[agg["model"] == model].set_index("pair_short")
        vals = [m_sub.loc[p, "pearson_mean"] if p in m_sub.index else np.nan for p in pairs_ord]
        errs = [m_sub.loc[p, "pearson_std"]  if p in m_sub.index else 0        for p in pairs_ord]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               yerr=errs, capsize=3,
               color=MODEL_PALETTE[model], alpha=0.85,
               label=MODEL_LABELS.get(model, model),
               error_kw=dict(elinewidth=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(pairs_ord, fontsize=9)
    ax.set_ylabel("Pearson r (mean ± std over 3 seeds)", fontsize=9)
    ax.set_title(title or f"k={k} few-shot performance", fontsize=11)
    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.4)
    savefig(fig, out_dir / fname)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    bact_df = load_single_model(BACT, "jepa")
    cold_df = load_multi_model(COLD)
    warm_df = load_multi_model(WARM)

    print(f"  bact_emb:  {len(bact_df)} rows")
    print(f"  cold:      {len(cold_df)} rows")
    print(f"  warmstart: {len(warm_df)} rows")

    # ── folder 1: bact_emb ────────────────────────────────────────────────────
    print("\n[1/4] bact_emb plots...")
    d = PLOT / "bact_emb"
    plot_learning_curves(bact_df, "bact_emb JEPA: few-shot via bacteria-embedding adaptation",
                         d, "learning_curves.png")
    plot_bar_at_k(bact_df, k=0,   out_dir=d, fname="bar_k0.png",   title="bact_emb JEPA — zero-shot (k=0)")
    plot_bar_at_k(bact_df, k=100, out_dir=d, fname="bar_k100.png", title="bact_emb JEPA — k=100")

    # ── folder 2: cold cross-species ──────────────────────────────────────────
    print("\n[2/4] cold cross-species plots...")
    d = PLOT / "cold_cross_species"
    plot_learning_curves(cold_df, "Cold-start: fresh head fine-tuned on k target examples",
                         d, "learning_curves.png")
    plot_model_scatter(cold_df, "jepa", "esm2",
                       d, "scatter_jepa_vs_esm2.png", "Cold: JEPA vs ESM2-150M (all k, all pairs)")
    plot_model_scatter(cold_df, "jepa", "esm2_650m",
                       d, "scatter_jepa_vs_esm2_650m.png", "Cold: JEPA vs ESM2-650M")
    plot_model_scatter(cold_df, "jepa", "mlm",
                       d, "scatter_jepa_vs_mlm.png", "Cold: JEPA vs MLM")
    plot_bar_at_k(cold_df, k=0,   out_dir=d, fname="bar_k0.png",   title="Cold: zero-shot (k=0)")
    plot_bar_at_k(cold_df, k=100, out_dir=d, fname="bar_k100.png", title="Cold: k=100")

    # ── folder 3: warmstart ───────────────────────────────────────────────────
    print("\n[3/4] warmstart plots...")
    d = PLOT / "warmstart"
    plot_learning_curves(warm_df, "Warmstart: source-trained head fine-tuned on k target examples",
                         d, "learning_curves.png")
    for m in ["jepa", "esm2", "esm2_650m"]:
        if m in warm_df["model"].values:
            plot_model_scatter(warm_df, "jepa", m, d,
                               f"scatter_jepa_vs_{m}.png",
                               f"Warmstart: JEPA vs {MODEL_LABELS.get(m, m)}")
    plot_bar_at_k(warm_df, k=0,   out_dir=d, fname="bar_k0.png",   title="Warmstart: zero-shot (k=0)")
    plot_bar_at_k(warm_df, k=100, out_dir=d, fname="bar_k100.png", title="Warmstart: k=100")

    # ── folder 4: comparison ──────────────────────────────────────────────────
    print("\n[4/4] comparison plots...")
    d = PLOT / "comparison"

    # cold vs warmstart per model
    for m in ["jepa", "esm2", "esm2_650m", "mlm"]:
        if m in cold_df.get("model", pd.Series()).values and m in warm_df.get("model", pd.Series()).values:
            plot_cold_vs_warm(cold_df, warm_df, m, d)

    # zero-shot comparison: bact_emb vs cold vs warmstart for JEPA
    if not bact_df.empty and not cold_df.empty:
        zs_bact = agg_mean_std(bact_df[bact_df["k"] == 0])[["pair_short", "pearson_mean"]].rename(columns={"pearson_mean": "bact_emb"})
        zs_cold = agg_mean_std(cold_df[(cold_df["model"] == "jepa") & (cold_df["k"] == 0)])[["pair_short", "pearson_mean"]].rename(columns={"pearson_mean": "cold"})
        zs_warm = agg_mean_std(warm_df[(warm_df["model"] == "jepa") & (warm_df["k"] == 0)])[["pair_short", "pearson_mean"]].rename(columns={"pearson_mean": "warmstart"}) if not warm_df.empty else None

        zs = zs_bact.merge(zs_cold, on="pair_short")
        if zs_warm is not None:
            zs = zs.merge(zs_warm, on="pair_short")
        if not zs.empty:
            fig, ax = plt.subplots(figsize=(5, 4))
            x = np.arange(len(zs))
            w = 0.25
            ax.bar(x - w, zs["bact_emb"], w, label="bact_emb JEPA", color=MODEL_PALETTE["jepa"], alpha=0.7)
            ax.bar(x,     zs["cold"],     w, label="cold (JEPA)",    color=MODEL_PALETTE["esm2"], alpha=0.7)
            if "warmstart" in zs.columns:
                ax.bar(x + w, zs["warmstart"], w, label="warmstart (JEPA)", color=MODEL_PALETTE["esm2_650m"], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(zs["pair_short"], fontsize=8, rotation=15, ha="right")
            ax.set_ylabel("Pearson r (k=0, zero-shot)", fontsize=9)
            ax.set_title("Zero-shot comparison: bact_emb vs cold vs warmstart (JEPA)", fontsize=10)
            ax.legend(fontsize=8, frameon=False)
            ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
            ax.grid(axis="y", alpha=0.4)
            savefig(fig, d / "zeroshot_comparison.png")

    # k=100 bar — all protocols + models
    if not cold_df.empty and not warm_df.empty:
        cold100 = cold_df[cold_df["k"] == 100].copy(); cold100["protocol"] = "cold"
        warm100 = warm_df[warm_df["k"] == 100].copy(); warm100["protocol"] = "warmstart"
        combined = pd.concat([cold100, warm100])
        combined["label"] = combined["model"].map(MODEL_LABELS) + "\n(" + combined["protocol"] + ")"
        agg100 = (combined.groupby(["label", "pair_short"])["pearson"]
                  .agg(["mean", "std"]).reset_index()
                  .rename(columns={"mean": "pearson_mean", "std": "pearson_std"}))
        fig, ax = plt.subplots(figsize=(10, 4))
        labels_ord = sorted(agg100["label"].unique())
        pairs_ord  = sorted(agg100["pair_short"].unique())
        x = np.arange(len(pairs_ord))
        width = 0.8 / len(labels_ord)
        palette = sns.color_palette("husl", len(labels_ord))
        for i, lbl in enumerate(labels_ord):
            sub = agg100[agg100["label"] == lbl].set_index("pair_short")
            vals = [sub.loc[p, "pearson_mean"] if p in sub.index else np.nan for p in pairs_ord]
            errs = [sub.loc[p, "pearson_std"]  if p in sub.index else 0        for p in pairs_ord]
            offset = (i - len(labels_ord) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9, yerr=errs, capsize=2,
                   color=palette[i], alpha=0.85, label=lbl,
                   error_kw=dict(elinewidth=0.8))
        ax.set_xticks(x)
        ax.set_xticklabels(pairs_ord, fontsize=9)
        ax.set_ylabel("Pearson r (k=100)", fontsize=9)
        ax.set_title("k=100: all models × protocols", fontsize=11)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", alpha=0.4)
        savefig(fig, d / "bar_k100_all.png")

    print(f"\nDone. All plots in {PLOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
