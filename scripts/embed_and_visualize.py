"""
Embedding interpretability analysis for cross-species MIC transfer.

Steps:
  1. Extract embeddings for all sequences from top-3 species (E.coli, S.aureus, P.aeruginosa)
  2. UMAP colored by (a) species and (b) MIC value — shows if embedding is species- or function-oriented
  3. Species decodability: linear probe accuracy for predicting species from embedding
  4. MIC linear probing: per-species linear R² from embedding

Outputs:
  eval_results/interpretability/{model}/
    umap_by_species.png
    umap_by_mic.png
    umap_combined.png      -- 2x2: [jepa/mlm] x [species/mic]
    species_decodability.json
    mic_linear_r2.json
  eval_results/interpretability/
    summary_decodability.png   -- bar chart: species decodability across models
    summary_linear_r2.png      -- bar chart: MIC linear R² across models

Usage:
    uv run python scripts/embed_and_visualize.py --models jepa mlm esm2
    uv run python scripts/embed_and_visualize.py --models jepa mlm esm2 esm2_650m
    uv run python scripts/embed_and_visualize.py --models jepa --n_max 2000  # quick smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_BASE = PROJECT_ROOT / "eval_results" / "interpretability"

AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIES = ["E. coli", "S. aureus", "P. aeruginosa"]
SP_SHORT = {"E. coli": "E. coli", "S. aureus": "S. aureus", "P. aeruginosa": "P. aeruginosa"}
SP_COLOR = {"E. coli": "#534AB7", "S. aureus": "#D85A30", "P. aeruginosa": "#1D9E75"}

MODEL_LABELS = {"jepa": "RAMP", "esm2": "ESM2-150M",
                "mlm": "MLM", "esm2_650m": "ESM2-650M"}
MODEL_COLORS = {"jepa": "#534AB7", "esm2": "#1D9E75",
                "mlm": "#D85A30", "esm2_650m": "#378ADD"}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


# ── data ─────────────────────────────────────────────────────────────────────

def load_all_species(grampa: Path, species_list: list, max_len=50,
                     n_max: int | None = None, seed=42) -> tuple[list, list, list]:
    """Returns (seqs, log2_mics, species_labels)."""
    seqs, mics, labels = [], [], []
    for sp in species_list:
        recs = []
        with open(grampa) as f:
            for r in csv.DictReader(f):
                seq = r["sequence"].strip().upper()
                if (r["is_modified"].strip() == "False"
                        and r["bacterium"].strip() == sp
                        and 3 <= len(seq) <= max_len
                        and all(c in AA for c in seq)):
                    try:
                        recs.append((seq, float(r["value"])))
                    except ValueError:
                        continue
        # deduplicate by sequence (keep first)
        seen = {}
        for seq, mic in recs:
            if seq not in seen:
                seen[seq] = mic
        sp_seqs = list(seen.keys())
        sp_mics = [seen[s] for s in sp_seqs]
        if n_max:
            rng = random.Random(seed)
            idx = list(range(len(sp_seqs))); rng.shuffle(idx)
            idx = idx[:n_max]
            sp_seqs = [sp_seqs[i] for i in idx]
            sp_mics = [sp_mics[i] for i in idx]
        seqs.extend(sp_seqs)
        mics.extend(sp_mics)
        labels.extend([sp] * len(sp_seqs))
        print(f"  {sp}: {len(sp_seqs)} sequences")
    return seqs, mics, labels


# ── embedders ────────────────────────────────────────────────────────────────

def _batch_encode_jepa(seqs, device):
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:50]) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out

def load_embedder(model_name: str, device):
    if model_name == "jepa":
        from src.models.jepa import JEPA
        ckpt = torch.load(PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
                          map_location=device, weights_only=False)
        jepa = JEPA(**ckpt["cfg"]["model"])
        jepa.load_state_dict(ckpt["model_state"])
        enc = jepa.context_encoder
        d   = ckpt["cfg"]["model"]["d_model"]
        for p in enc.parameters(): p.requires_grad_(False)
        enc = enc.to(device).eval()
        def embed(seqs):
            ids = _batch_encode_jepa(seqs, device)
            h   = enc(ids); pad = ids == 0
            h   = h.masked_fill(pad.unsqueeze(-1), 0.0)
            return (h.sum(1) / (~pad).sum(1, keepdim=True).float().clamp(min=1)).cpu().numpy()
        return embed, d

    elif model_name == "mlm":
        from src.models.mlm import MLMModel
        ckpt = torch.load(PROJECT_ROOT / "checkpoints/mlm_pretrain_868k/best_mlm.pt",
                          map_location=device, weights_only=False)
        model = MLMModel(**ckpt["cfg"]["model"])
        enc_state = {k[len("encoder."):]: v for k, v in ckpt["model_state"].items()
                     if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_state)
        enc = model.encoder; d = ckpt["cfg"]["model"]["d_model"]
        for p in enc.parameters(): p.requires_grad_(False)
        enc = enc.to(device).eval()
        def embed(seqs):
            ids = _batch_encode_jepa(seqs, device)
            h   = enc(ids); pad = ids == 0
            h   = h.masked_fill(pad.unsqueeze(-1), 0.0)
            return (h.sum(1) / (~pad).sum(1, keepdim=True).float().clamp(min=1)).cpu().numpy()
        return embed, d

    elif model_name in ("esm2", "esm2_650m"):
        from src.models.esm_head import load_esm2
        key = "esm2_t12_35M" if model_name == "esm2" else "esm2_t33_650M"
        esm, alphabet, d = load_esm2(key)
        for p in esm.parameters(): p.requires_grad_(False)
        esm = esm.to(device).eval()
        bc  = alphabet.get_batch_converter()
        nl  = esm.num_layers
        def embed(seqs):
            data = [(f"s{i}", s) for i, s in enumerate(seqs)]
            _, _, tokens = bc(data); tokens = tokens.to(device)
            with torch.no_grad():
                out = esm(tokens, repr_layers=[nl], return_contacts=False)
            h = out["representations"][nl]; pad = tokens == alphabet.padding_idx
            h = h.masked_fill(pad.unsqueeze(-1), 0.0)
            return (h.sum(1) / (~pad).sum(1, keepdim=True).float().clamp(min=1)).cpu().numpy()
        return embed, d

    raise ValueError(f"Unknown model: {model_name}")


def extract_embeddings(embed_fn, seqs: list, batch=256) -> np.ndarray:
    parts = []
    for i in range(0, len(seqs), batch):
        with torch.no_grad():
            parts.append(embed_fn(seqs[i:i+batch]))
    return np.vstack(parts)


# ── UMAP ─────────────────────────────────────────────────────────────────────

def run_umap(embs: np.ndarray, n_neighbors=30, min_dist=0.1, seed=42):
    try:
        from umap import UMAP
    except ImportError:
        print("  umap-learn not installed. Falling back to PCA for 2D.")
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(embs)
    return UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                random_state=seed, n_jobs=4).fit_transform(embs)


def plot_umap_by_species(coords, labels, out_path, title=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    for sp in SPECIES:
        mask = np.array(labels) == sp
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=SP_COLOR[sp], label=SP_SHORT[sp],
                   alpha=0.35, s=8, rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
    ax.legend(fontsize=8, frameon=False, markerscale=2)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap_by_mic(coords, mics, out_path, title=""):
    mics_arr = np.array(mics)
    vmin, vmax = np.percentile(mics_arr, 5), np.percentile(mics_arr, 95)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=mics_arr, cmap="RdYlGn_r",
                    vmin=vmin, vmax=vmax,
                    alpha=0.35, s=8, rasterized=True)
    plt.colorbar(sc, ax=ax, label="log₂(MIC) ↑ resistant", shrink=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── species decodability ─────────────────────────────────────────────────────

def species_decodability(embs: np.ndarray, labels: list) -> dict:
    """Linear probe accuracy for species classification."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le  = LabelEncoder()
    y   = le.fit_transform(labels)
    X   = StandardScaler().fit_transform(embs)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return {"mean": float(scores.mean()), "std": float(scores.std()),
            "classes": list(le.classes_)}


# ── MIC linear R² ────────────────────────────────────────────────────────────

def mic_linear_r2(embs: np.ndarray, mics: list, labels: list) -> dict:
    """Per-species linear R² for MIC prediction from embeddings."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    results = {}
    for sp in SPECIES:
        mask = np.array(labels) == sp
        X = StandardScaler().fit_transform(embs[mask])
        y = np.array(mics)[mask]
        if X.shape[0] < 20: continue
        reg    = Ridge(alpha=1.0)
        scores = cross_val_score(reg, X, y, cv=5, scoring="r2")
        results[sp] = {"r2_mean": float(scores.mean()), "r2_std": float(scores.std())}
    return results


# ── combined 2-model comparison figure ───────────────────────────────────────

def plot_combined(coords_dict: dict, labels, mics, out_path):
    """2×2 grid: rows=model, cols=[species, MIC]."""
    models = list(coords_dict.keys())
    fig, axes = plt.subplots(len(models), 2,
                             figsize=(9, 4.5 * len(models)))
    if len(models) == 1: axes = axes[np.newaxis, :]
    mics_arr = np.array(mics)
    vmin, vmax = np.percentile(mics_arr, 5), np.percentile(mics_arr, 95)

    for row, model in enumerate(models):
        coords = coords_dict[model]
        # species
        ax = axes[row][0]
        for sp in SPECIES:
            mask = np.array(labels) == sp
            ax.scatter(coords[mask,0], coords[mask,1],
                       c=SP_COLOR[sp], label=SP_SHORT[sp],
                       alpha=0.3, s=6, rasterized=True)
        ax.set_title(f"{MODEL_LABELS.get(model,model)} — colored by species",
                     fontsize=10)
        ax.legend(fontsize=7, frameon=False, markerscale=2)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("UMAP-1", fontsize=8); ax.set_ylabel("UMAP-2", fontsize=8)
        # MIC
        ax = axes[row][1]
        sc = ax.scatter(coords[:,0], coords[:,1],
                        c=mics_arr, cmap="RdYlGn_r",
                        vmin=vmin, vmax=vmax,
                        alpha=0.3, s=6, rasterized=True)
        plt.colorbar(sc, ax=ax, label="log₂(MIC)", shrink=0.85)
        ax.set_title(f"{MODEL_LABELS.get(model,model)} — colored by MIC",
                     fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("UMAP-1", fontsize=8); ax.set_ylabel("UMAP-2", fontsize=8)

    fig.tight_layout(h_pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


# ── summary bar charts ────────────────────────────────────────────────────────

def plot_summary_decodability(results: dict, out_path):
    models = list(results.keys())
    means  = [results[m]["mean"] for m in models]
    stds   = [results[m]["std"]  for m in models]
    colors = [MODEL_COLORS.get(m, "#888") for m in models]
    xlabels = [MODEL_LABELS.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(xlabels, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, width=0.5)
    # random baseline
    n_classes = len(results[models[0]]["classes"])
    ax.axhline(1/n_classes, color="#888", lw=1.2, ls="--",
               label=f"Random ({1/n_classes:.2f})")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(stds)*0.15,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Species classification accuracy", fontsize=10)
    ax.set_title("Species decodability\n(lower = more species-agnostic = better transfer)",
                 fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


def plot_summary_r2(results: dict, out_path):
    """results: {model: {species: {r2_mean, r2_std}}}"""
    models  = list(results.keys())
    species = SPECIES
    x = np.arange(len(species))
    w = 0.7 / len(models)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, model in enumerate(models):
        means = [results[model].get(sp, {}).get("r2_mean", np.nan) for sp in species]
        stds  = [results[model].get(sp, {}).get("r2_std",  0)      for sp in species]
        offset = (i - len(models)/2 + 0.5) * w
        ax.bar(x + offset, means, w*0.9, yerr=stds, capsize=3,
               color=MODEL_COLORS.get(model, "#888"), alpha=0.85,
               label=MODEL_LABELS.get(model, model))
    ax.set_xticks(x); ax.set_xticklabels(species, fontsize=9)
    ax.set_ylabel("Linear R² (5-fold CV)", fontsize=10)
    ax.set_title("MIC linear predictability from embedding\n(higher = MIC more linearly encoded)",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--models", nargs="+", default=["jepa", "mlm", "esm2"],
                        choices=["jepa", "mlm", "esm2", "esm2_650m"])
    parser.add_argument("--n_max",  type=int, default=None,
                        help="Max sequences per species (None=all, 500=quick test)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grampa = PROJECT_ROOT / "data" / "grampa.csv"

    print("Loading sequences...")
    seqs, mics, labels = load_all_species(grampa, SPECIES, n_max=args.n_max)
    print(f"Total: {len(seqs)} sequences")

    decodability_all = {}
    r2_all           = {}
    coords_all       = {}   # for combined plot

    for model_name in args.models:
        print(f"\n{'='*50}\n{model_name}\n{'='*50}")
        out_dir = OUT_BASE / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # extract embeddings
        print("  Extracting embeddings...")
        embed_fn, d = load_embedder(model_name, device)
        embs = extract_embeddings(embed_fn, seqs)
        np.save(out_dir / "embeddings.npy", embs)
        print(f"  Embeddings shape: {embs.shape}")

        # UMAP
        print("  Running UMAP...")
        coords = run_umap(embs)
        np.save(out_dir / "umap_coords.npy", coords)
        coords_all[model_name] = coords

        plot_umap_by_species(coords, labels, out_dir / "umap_by_species.png",
                             title=f"{MODEL_LABELS.get(model_name, model_name)} — by species")
        print(f"  saved → {(out_dir / 'umap_by_species.png').relative_to(PROJECT_ROOT)}")

        plot_umap_by_mic(coords, mics, out_dir / "umap_by_mic.png",
                         title=f"{MODEL_LABELS.get(model_name, model_name)} — by MIC")
        print(f"  saved → {(out_dir / 'umap_by_mic.png').relative_to(PROJECT_ROOT)}")

        # species decodability
        print("  Species decodability...")
        dec = species_decodability(embs, labels)
        decodability_all[model_name] = dec
        (out_dir / "species_decodability.json").write_text(json.dumps(dec, indent=2))
        print(f"  accuracy={dec['mean']:.3f} ± {dec['std']:.3f}")

        # MIC linear R²
        print("  MIC linear R²...")
        r2 = mic_linear_r2(embs, mics, labels)
        r2_all[model_name] = r2
        (out_dir / "mic_linear_r2.json").write_text(json.dumps(r2, indent=2))
        for sp, v in r2.items():
            print(f"  {sp}: R²={v['r2_mean']:.3f} ± {v['r2_std']:.3f}")

        del embed_fn
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # combined UMAP figure
    if len(coords_all) >= 2:
        print("\nGenerating combined UMAP figure...")
        plot_combined(coords_all, labels, mics, OUT_BASE / "umap_combined.png")

    # summary bar charts
    print("Summary plots...")
    if decodability_all:
        plot_summary_decodability(decodability_all, OUT_BASE / "summary_decodability.png")
    if r2_all:
        plot_summary_r2(r2_all, OUT_BASE / "summary_linear_r2.png")

    print(f"\nDone. → {OUT_BASE.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
