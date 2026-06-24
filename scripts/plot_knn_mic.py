"""
k-NN MIC consistency in embedding space.

For each peptide, find its k nearest neighbours within the same species
(in full embedding space). Measure std(MIC) of those neighbours.

Lower std = nearby sequences have more similar activity
           = embedding space is more "MIC-coherent"

Plots:
  1. Violin + strip: distribution of neighbour MIC std per model
  2. Per-species breakdown (3 columns)
  3. Mean neighbour MIC std vs k curve (how does coherence scale?)

Output: eval_results/interpretability/knn_mic_coherence.png

Usage:
    python scripts/plot_knn_mic.py
"""
from __future__ import annotations
import csv, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT    = PROJECT_ROOT / "eval_results" / "interpretability"
AA     = "ACDEFGHIKLMNPQRSTVWY"
SPECIES = ["E. coli", "S. aureus", "P. aeruginosa"]
SP_SHORT = {"E. coli": "E. coli", "S. aureus": "S. aureus", "P. aeruginosa": "P. aeruginosa"}

MODELS = ["jepa", "mlm", "esm2"]
MODEL_LABELS = {"jepa": "RAMP", "mlm": "MLM", "esm2": "ESM2-150M"}
MODEL_COLORS = {"jepa": "#534AB7", "mlm": "#D85A30", "esm2": "#1D9E75"}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


# ── load MIC labels (same seed/order as embed_and_visualize) ─────────────────

def load_mics(grampa, n_max=1500, seed=42):
    """Returns mics list and labels list in same order as embeddings.npy."""
    all_mics, all_labels = [], []
    for sp in SPECIES:
        recs = {}
        with open(grampa) as f:
            for r in csv.DictReader(f):
                seq = r["sequence"].strip().upper()
                if (r["is_modified"].strip() == "False"
                        and r["bacterium"].strip() == sp
                        and 3 <= len(seq) <= 50
                        and all(c in AA for c in seq)):
                    try:
                        if seq not in recs:
                            recs[seq] = float(r["value"])
                    except ValueError:
                        continue
        seqs = list(recs.keys())
        mics = [recs[s] for s in seqs]
        rng  = random.Random(seed)
        idx  = list(range(len(seqs))); rng.shuffle(idx)
        idx  = idx[:n_max]
        all_mics.extend([mics[i] for i in idx])
        all_labels.extend([sp] * len(idx))
    return np.array(all_mics), all_labels


# ── k-NN MIC std ─────────────────────────────────────────────────────────────

def knn_mic_std(emb: np.ndarray, mics: np.ndarray, labels: list, k: int = 20):
    """
    For each point, find k nearest neighbours within same species.
    Returns array of std(MIC of neighbours), one per point.
    """
    emb_norm = normalize(emb, norm="l2")
    labels   = np.array(labels)
    stds     = np.full(len(emb), np.nan)

    for sp in SPECIES:
        mask = labels == sp
        idx  = np.where(mask)[0]
        if len(idx) <= k:
            continue
        E   = emb_norm[idx]             # (N_sp, d)
        sim = E @ E.T                    # (N_sp, N_sp) cosine similarities
        np.fill_diagonal(sim, -2)        # exclude self

        # top-k neighbours per point
        top_k = np.argsort(sim, axis=1)[:, -k:]   # (N_sp, k)
        mic_sp = mics[idx]
        neighbour_mics = mic_sp[top_k]             # (N_sp, k)
        stds[idx] = neighbour_mics.std(axis=1)

    return stds


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_violin(df: pd.DataFrame, out_path: Path):
    """Overall violin + mean markers, one column per model."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    order = [MODEL_LABELS[m] for m in MODELS]
    palette = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in MODELS}

    sns.violinplot(data=df, x="model", y="knn_std", hue="model", order=order,
                   palette=palette, inner=None, linewidth=0.8,
                   cut=0, ax=ax, alpha=0.75, legend=False)

    # overlay mean ± se
    for i, m in enumerate(MODELS):
        sub = df[df["model"] == MODEL_LABELS[m]]["knn_std"]
        mn, se = sub.mean(), sub.sem()
        ax.scatter([i], [mn], color="white", zorder=5, s=60, linewidths=1.5,
                   edgecolors=MODEL_COLORS[m])
        ax.errorbar([i], [mn], yerr=se, color=MODEL_COLORS[m],
                    capsize=4, lw=1.8, zorder=4)
        ax.text(i, mn + se + 0.005, f"{mn:.3f}",
                ha="center", va="bottom", fontsize=9.5,
                color=MODEL_COLORS[m], fontweight="medium")

    ax.set_xlabel("")
    ax.set_ylabel("std(MIC) of 20 nearest neighbours\n(lower = more MIC-coherent embedding)", fontsize=10)
    ax.set_title("Embedding MIC coherence — k-NN (k=20, within species)\n"
                 "Lower = similar embeddings → similar activity", fontsize=10.5)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


def plot_per_species(df: pd.DataFrame, out_path: Path):
    """Per-species breakdown, 3 panels."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    order = [MODEL_LABELS[m] for m in MODELS]
    palette = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in MODELS}

    for ax, sp in zip(axes, SPECIES):
        sub = df[df["species"] == sp]
        sns.violinplot(data=sub, x="model", y="knn_std", hue="model", order=order,
                       palette=palette, inner=None, linewidth=0.8,
                       cut=0, ax=ax, alpha=0.75, legend=False)
        for i, m in enumerate(MODELS):
            s = sub[sub["model"] == MODEL_LABELS[m]]["knn_std"]
            mn = s.mean()
            ax.scatter([i], [mn], color="white", zorder=5, s=50,
                       linewidths=1.5, edgecolors=MODEL_COLORS[m])
            ax.text(i, mn + 0.008, f"{mn:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=MODEL_COLORS[m], fontweight="medium")
        ax.set_title(SP_SHORT.get(sp, sp), fontsize=11, style="italic")
        ax.set_xlabel("")
        if ax == axes[0]:
            ax.set_ylabel("std(MIC) of neighbours", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Per-species embedding MIC coherence (k=20 nearest neighbours)",
                 fontsize=12, y=1.02)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


def plot_k_curve(embs: dict, mics: np.ndarray, labels: list, out_path: Path):
    """Mean neighbour MIC std vs k for all models."""
    ks = [5, 10, 20, 50, 100]
    fig, ax = plt.subplots(figsize=(6, 4))

    for m in MODELS:
        means, sems = [], []
        for k in ks:
            stds = knn_mic_std(embs[m], mics, labels, k=k)
            stds = stds[~np.isnan(stds)]
            means.append(stds.mean()); sems.append(stds.std() / np.sqrt(len(stds)))
        means, sems = np.array(means), np.array(sems)
        color = MODEL_COLORS[m]
        ax.plot(ks, means, marker="o", lw=2, color=color,
                label=MODEL_LABELS[m])
        ax.fill_between(ks, means - sems, means + sems, alpha=0.15, color=color)

    ax.set_xlabel("k (number of neighbours)", fontsize=10)
    ax.set_ylabel("Mean std(MIC) of k-NN", fontsize=10)
    ax.set_title("MIC coherence vs neighbourhood size\n(lower = embedding encodes activity more locally)", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    grampa = PROJECT_ROOT / "data" / "grampa.csv"
    print("Loading MIC labels...")
    mics, labels = load_mics(grampa, n_max=1500)
    print(f"  {len(mics)} sequences")

    embs = {}
    for m in MODELS:
        path = OUT / m / "embeddings.npy"
        if not path.exists():
            print(f"  [{m}] embeddings.npy not found — skipping"); continue
        embs[m] = np.load(path)
        print(f"  [{m}] loaded {embs[m].shape}")

    print("\nComputing k-NN MIC std (k=20)...")
    rows = []
    for m, emb in embs.items():
        stds = knn_mic_std(emb, mics, labels, k=20)
        for i, std in enumerate(stds):
            if not np.isnan(std):
                rows.append({"model": MODEL_LABELS[m],
                             "species": labels[i],
                             "knn_std": std})
    df = pd.DataFrame(rows)

    # print summary
    print("\nMean neighbour MIC std (k=20):")
    for m in MODELS:
        if MODEL_LABELS[m] in df["model"].values:
            mn = df[df["model"] == MODEL_LABELS[m]]["knn_std"].mean()
            print(f"  {MODEL_LABELS[m]:12s}: {mn:.4f}")

    print("\nPlotting...")
    plot_violin(df, OUT / "knn_mic_coherence.png")
    plot_per_species(df, OUT / "knn_mic_coherence_per_species.png")
    plot_k_curve(embs, mics, labels, OUT / "knn_mic_coherence_k_curve.png")
    print("Done.")


if __name__ == "__main__":
    main()
