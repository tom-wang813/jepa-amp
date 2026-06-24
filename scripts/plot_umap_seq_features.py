"""
UMAP colored by sequence-level physicochemical features.

For each model, one figure with 6 panels:
  species | length | net charge | hydrophobicity | %positive | %hydrophobic

Usage:
    uv run python scripts/plot_umap_seq_features.py --models jepa mlm esm2
"""
from __future__ import annotations
import argparse, csv, json, random, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT  = PROJECT_ROOT / "eval_results" / "interpretability"
AA   = "ACDEFGHIKLMNPQRSTVWY"
SPECIES = ["E. coli", "S. aureus", "P. aeruginosa"]
SP_COLOR = {"E. coli": "#534AB7", "S. aureus": "#D85A30", "P. aeruginosa": "#1D9E75"}
MODEL_LABELS = {"jepa": "RAMP", "esm2": "ESM2-150M",
                "mlm": "MLM", "esm2_650m": "ESM2-650M"}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

# ── Kyte-Doolittle hydrophobicity scale ──────────────────────────────────────
KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
      "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
      "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
# net charge at pH 7 (approximate)
POS = set("KRH")
NEG = set("DE")
HYDROPHOBIC = set("AILMFWV")

def seq_features(seq: str) -> dict:
    n   = len(seq)
    charge  = sum(1 for c in seq if c in POS) - sum(1 for c in seq if c in NEG)
    hydro   = np.mean([KD.get(c, 0) for c in seq])
    frac_pos  = sum(1 for c in seq if c in POS) / n
    frac_hydro = sum(1 for c in seq if c in HYDROPHOBIC) / n
    return {"length": n, "charge": charge,
            "hydrophobicity": hydro,
            "frac_positive": frac_pos,
            "frac_hydrophobic": frac_hydro}

# ── load data ────────────────────────────────────────────────────────────────
def load_seqs(grampa, n_max=1500, seed=42):
    seqs, labels = [], []
    for sp in SPECIES:
        recs = []
        with open(grampa) as f:
            for r in csv.DictReader(f):
                seq = r["sequence"].strip().upper()
                if (r["is_modified"].strip() == "False"
                        and r["bacterium"].strip() == sp
                        and 3 <= len(seq) <= 50
                        and all(c in AA for c in seq)):
                    recs.append(seq)
        seen = list(dict.fromkeys(recs))   # deduplicate, preserve order
        rng = random.Random(seed); rng.shuffle(seen)
        chosen = seen[:n_max]
        seqs.extend(chosen); labels.extend([sp]*len(chosen))
    return seqs, labels


# ── compute all features ──────────────────────────────────────────────────────
def compute_features(seqs):
    feats = [seq_features(s) for s in seqs]
    return {k: np.array([f[k] for f in feats]) for k in feats[0]}


# ── main plot: 6-panel per model ─────────────────────────────────────────────
PANELS = [
    ("species",         "Species",             None,         "tab10"),
    ("length",          "Length (aa)",         None,         "viridis"),
    ("charge",          "Net charge",          None,         "RdBu"),
    ("hydrophobicity",  "Hydrophobicity\n(Kyte-Doolittle)", None, "RdYlGn"),
    ("frac_positive",   "Fraction\npositive (K/R/H)", None,  "Blues"),
    ("frac_hydrophobic","Fraction\nhydrophobic",None,         "Oranges"),
]

def plot_model(model_name: str, coords: np.ndarray, seqs: list,
               labels: list, feats: dict, out_path: Path):
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4.8*nrows))
    axes = axes.flatten()

    for ax, (key, title, _, cmap) in zip(axes, PANELS):
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("UMAP-1", fontsize=8); ax.set_ylabel("UMAP-2", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="medium")

        if key == "species":
            for sp in SPECIES:
                mask = np.array(labels) == sp
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=SP_COLOR[sp], label=sp,
                           alpha=0.4, s=10, rasterized=True)
            ax.legend(fontsize=7, frameon=False, markerscale=2,
                      loc="upper right")
        else:
            vals = feats[key]
            # symmetric colormap for charge
            if key == "charge":
                vmax = max(abs(vals.min()), abs(vals.max()))
                vmin, vmax = -vmax, vmax
            else:
                vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)

            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                            alpha=0.4, s=10, rasterized=True)
            cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
            cb.ax.tick_params(labelsize=7)

    fig.suptitle(f"{MODEL_LABELS.get(model_name, model_name)} — UMAP colored by sequence features",
                 fontsize=13, y=1.01, fontweight="medium")
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


# ── combined: side-by-side charge + hydrophobicity for all models ─────────────
def plot_cross_model_comparison(all_coords: dict, all_feats: dict, labels, out_path):
    """Compare models on the 2 most AMP-relevant features: charge & hydrophobicity."""
    models  = list(all_coords.keys())
    features = [("charge", "Net charge", "RdBu"),
                ("hydrophobicity", "Hydrophobicity", "RdYlGn")]

    fig, axes = plt.subplots(len(models), 2,
                             figsize=(8, 4.2*len(models)))
    if len(models) == 1: axes = axes[np.newaxis, :]

    for row, model in enumerate(models):
        coords = all_coords[model]
        feats  = all_feats[model]
        for col, (key, title, cmap) in enumerate(features):
            ax   = axes[row][col]
            vals = feats[key]
            if key == "charge":
                vmax = max(abs(vals.min()), abs(vals.max()))
                vmin, vmax = -vmax, vmax
            else:
                vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
            sc = ax.scatter(coords[:,0], coords[:,1],
                            c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                            alpha=0.35, s=8, rasterized=True)
            cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            ax.set_title(f"{MODEL_LABELS.get(model,model)}\n{title}", fontsize=10)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("UMAP-1", fontsize=8)
            ax.set_ylabel("UMAP-2", fontsize=8)
            ax.set_aspect("equal")

    fig.suptitle("Charge & hydrophobicity in embedding space — model comparison",
                 fontsize=12, y=1.01)
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["jepa","mlm","esm2"])
    parser.add_argument("--n_max", type=int, default=1500)
    args = parser.parse_args()

    grampa = PROJECT_ROOT / "data" / "grampa.csv"
    print(f"Loading {args.n_max} seqs/species...")
    seqs, labels = load_seqs(grampa, n_max=args.n_max)
    feats = compute_features(seqs)
    print(f"Total: {len(seqs)} sequences")

    all_coords, all_feats = {}, {}

    for model in args.models:
        coords_path = OUT / model / "umap_coords.npy"
        if not coords_path.exists():
            print(f"  [{model}] umap_coords.npy not found — run embed_and_visualize.py first")
            continue
        coords = np.load(coords_path)
        if len(coords) != len(seqs):
            print(f"  [{model}] coord size mismatch ({len(coords)} vs {len(seqs)}) — skipping")
            continue
        all_coords[model] = coords
        all_feats[model]  = feats

        print(f"\n[{model}] 6-panel feature plot...")
        plot_model(model, coords, seqs, labels, feats,
                   OUT / model / "umap_seq_features.png")

    if len(all_coords) >= 2:
        print("\nCross-model comparison (charge + hydrophobicity)...")
        plot_cross_model_comparison(all_coords, all_feats, labels,
                                    OUT / "umap_charge_hydro_comparison.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
