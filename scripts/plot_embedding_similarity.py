"""
Cross-species embedding similarity vs MIC similarity analysis.

Hypothesis: RAMP predicts in representation space → embeddings that are
similar should correspond to peptides with similar *function* (MIC) across
species, more so than MLM or ESM2.

Analysis:
  For each pair (src_species, tgt_species):
    1. Sample N peptides from each species
    2. Compute all pairwise cosine similarities in embedding space
    3. Compute all pairwise |ΔMIC| = |MIC_src - MIC_tgt|
    4. Bin by cosine similarity; plot mean |ΔMIC| per bin
       → If RAMP is better: high-similarity pairs have lower |ΔMIC|

  Summary metric: Spearman ρ between cosine_sim and -|ΔMIC| per model
  (higher ρ = embedding similarity better predicts MIC similarity)

Outputs:
  eval_results/interpretability/
    emb_sim_vs_mic_sim_{src}_{tgt}.png   -- binned curve per pair
    emb_sim_summary.png                  -- summary bar per model

Usage:
    uv run python scripts/plot_embedding_similarity.py --gpu 0 --n_per_species 300
"""
from __future__ import annotations
import argparse, csv, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT  = PROJECT_ROOT / "eval_results" / "interpretability"
AA   = "ACDEFGHIKLMNPQRSTVWY"

SPECIES_PAIRS = [
    ("E. coli",       "S. aureus"),
    ("E. coli",       "P. aeruginosa"),
    ("S. aureus",     "E. coli"),
]
SP_SHORT = {"E. coli": "Eco", "S. aureus": "Sau", "P. aeruginosa": "Pae"}

MODELS = ["jepa", "mlm", "esm2"]
MODEL_LABELS = {"jepa": "RAMP", "mlm": "MLM",
                "esm2": "ESM2-150M", "esm2_650m": "ESM2-650M"}
MODEL_COLORS = {"jepa": "#534AB7", "mlm": "#D85A30",
                "esm2": "#1D9E75",  "esm2_650m": "#378ADD"}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


# ── data ─────────────────────────────────────────────────────────────────────

def load_species(grampa, species, n_max=300, seed=42):
    recs = {}
    with open(grampa) as f:
        for r in csv.DictReader(f):
            seq = r["sequence"].strip().upper()
            if (r["is_modified"].strip() == "False"
                    and r["bacterium"].strip() == species
                    and 3 <= len(seq) <= 50
                    and all(c in AA for c in seq)):
                try:
                    mic = float(r["value"])
                    if seq not in recs:
                        recs[seq] = mic
                except ValueError:
                    continue
    seqs = list(recs.keys())
    mics = [recs[s] for s in seqs]
    rng  = random.Random(seed); idx = list(range(len(seqs))); rng.shuffle(idx)
    idx  = idx[:n_max]
    return [seqs[i] for i in idx], [mics[i] for i in idx]


# ── embedders (reuse from embed_and_visualize) ────────────────────────────────

def _batch_encode(seqs, device):
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:50]) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out

def load_embedder(model_name, device):
    if model_name == "jepa":
        from src.models.jepa import JEPA
        ckpt = torch.load(PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
                          map_location=device, weights_only=False)
        jepa = JEPA(**ckpt["cfg"]["model"])
        jepa.load_state_dict(ckpt["model_state"])
        enc = jepa.context_encoder
        for p in enc.parameters(): p.requires_grad_(False)
        enc = enc.to(device).eval()
        def embed(seqs):
            ids = _batch_encode(seqs, device)
            h = enc(ids); pad = ids == 0
            h = h.masked_fill(pad.unsqueeze(-1), 0.0)
            return (h.sum(1) / (~pad).sum(1, keepdim=True).float().clamp(min=1)).cpu().numpy()
        return embed

    if model_name == "mlm":
        from src.models.mlm import MLMModel
        ckpt = torch.load(PROJECT_ROOT / "checkpoints/mlm_pretrain_868k/best_mlm.pt",
                          map_location=device, weights_only=False)
        model = MLMModel(**ckpt["cfg"]["model"])
        enc_state = {k[len("encoder."):]: v for k, v in ckpt["model_state"].items()
                     if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_state)
        enc = model.encoder
        for p in enc.parameters(): p.requires_grad_(False)
        enc = enc.to(device).eval()
        def embed(seqs):
            ids = _batch_encode(seqs, device)
            h = enc(ids); pad = ids == 0
            h = h.masked_fill(pad.unsqueeze(-1), 0.0)
            return (h.sum(1) / (~pad).sum(1, keepdim=True).float().clamp(min=1)).cpu().numpy()
        return embed

    if model_name in ("esm2", "esm2_650m"):
        from src.models.esm_head import load_esm2
        key = "esm2_t12_35M" if model_name == "esm2" else "esm2_t33_650M"
        esm, alphabet, _ = load_esm2(key)
        for p in esm.parameters(): p.requires_grad_(False)
        esm = esm.to(device).eval()
        bc = alphabet.get_batch_converter(); nl = esm.num_layers
        def embed(seqs):
            data = [(f"s{i}", s) for i, s in enumerate(seqs)]
            _, _, tokens = bc(data); tokens = tokens.to(device)
            with torch.no_grad():
                out = esm(tokens, repr_layers=[nl], return_contacts=False)
            h = out["representations"][nl]; pad = tokens == alphabet.padding_idx
            h = h.masked_fill(pad.unsqueeze(-1), 0.0)
            return (h.sum(1) / (~pad).sum(1, keepdim=True).float().clamp(min=1)).cpu().numpy()
        return embed

    raise ValueError(model_name)


def embed_batch(embed_fn, seqs, batch=256):
    parts = []
    for i in range(0, len(seqs), batch):
        with torch.no_grad():
            parts.append(embed_fn(seqs[i:i+batch]))
    return np.vstack(parts)


# ── core analysis ─────────────────────────────────────────────────────────────

def compute_cross_species_sim_mic(
        emb_src, mics_src, emb_tgt, mics_tgt, n_bins=10
):
    """
    All N×M cross-species pairs.
    Returns:
      cos_sims  : (N*M,) cosine similarities
      delta_mic : (N*M,) |MIC_src - MIC_tgt|
      bins      : bin centers
      bin_means : mean |ΔMIC| per bin
      bin_stds  : std per bin
      rho       : Spearman ρ(cos_sim, -|ΔMIC|)
    """
    # L2-normalize for cosine similarity via dot product
    E_src = normalize(emb_src, norm="l2")    # (N, d)
    E_tgt = normalize(emb_tgt, norm="l2")    # (M, d)
    cos_sims  = (E_src @ E_tgt.T).ravel()    # (N*M,)

    mic_src_rep = np.repeat(mics_src, len(mics_tgt))            # (N*M,)
    mic_tgt_rep = np.tile(mics_tgt,   len(mics_src))            # (N*M,)
    delta_mic   = np.abs(mic_src_rep - mic_tgt_rep)

    rho, _ = spearmanr(cos_sims, -delta_mic)

    # bin by cosine similarity
    bin_edges = np.percentile(cos_sims, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_means, bin_stds, bin_centers = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (cos_sims >= lo) & (cos_sims <= hi)
        if mask.sum() > 5:
            bin_means.append(delta_mic[mask].mean())
            bin_stds.append(delta_mic[mask].std())
            bin_centers.append((lo + hi) / 2)

    return (cos_sims, delta_mic,
            np.array(bin_centers), np.array(bin_means), np.array(bin_stds),
            rho)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_pair(pair_results: dict, src: str, tgt: str, out_path: Path):
    """One figure: binned curve for all models for one species pair."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, res in pair_results.items():
        centers, means, stds, rho = res["centers"], res["means"], res["stds"], res["rho"]
        color = MODEL_COLORS[model]
        label = f"{MODEL_LABELS[model]} (ρ={rho:.3f})"
        ax.plot(centers, means, marker="o", ms=5, lw=1.8, color=color, label=label)
        ax.fill_between(centers, means - stds, means + stds, alpha=0.12, color=color)

    ax.set_xlabel("Cross-species cosine similarity (embedding space)", fontsize=10)
    ax.set_ylabel("|ΔMIC| = |log₂MIC_src − log₂MIC_tgt|", fontsize=10)
    ax.set_title(
        f"{SP_SHORT.get(src,src)} → {SP_SHORT.get(tgt,tgt)}\n"
        f"Higher sim → lower |ΔMIC|? (ρ = Spearman, higher = better)",
        fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


def plot_summary(all_rhos: dict, out_path: Path):
    """Bar chart: Spearman ρ per model, averaged over species pairs."""
    models   = list(all_rhos.keys())
    pairs    = list(next(iter(all_rhos.values())).keys())
    x        = np.arange(len(pairs))
    w        = 0.7 / len(models)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})

    # left: per-pair bars
    ax = axes[0]
    for i, model in enumerate(models):
        vals   = [all_rhos[model].get(p, np.nan) for p in pairs]
        offset = (i - len(models)/2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w*0.9,
                        color=MODEL_COLORS[model], alpha=0.88,
                        label=MODEL_LABELS.get(model, model))
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7.5, color=MODEL_COLORS[model])
    pair_labels = [f"{SP_SHORT.get(s,s)}→{SP_SHORT.get(t,t)}" for s,t in SPECIES_PAIRS
                   if f"{s}→{t}" in pairs]
    ax.set_xticks(x); ax.set_xticklabels(pair_labels, fontsize=10)
    ax.set_ylabel("Spearman ρ (cos_sim vs −|ΔMIC|)", fontsize=10)
    ax.set_title("Cross-species embedding–MIC alignment\n(higher = more functionally coherent)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.grid(axis="y", alpha=0.35)

    # right: average across pairs
    ax2 = axes[1]
    avgs   = [np.nanmean(list(all_rhos[m].values())) for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    xlabels = [MODEL_LABELS.get(m, m) for m in models]
    bars = ax2.bar(xlabels, avgs, color=colors, alpha=0.88, width=0.5)
    for bar, v in zip(bars, avgs):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Average Spearman ρ", fontsize=10)
    ax2.set_title("Average over pairs", fontsize=10)
    ax2.axhline(0, color="#999", lw=0.8, ls="--")
    ax2.grid(axis="y", alpha=0.35)

    fig.suptitle("Do similar embeddings predict similar MIC across species?", fontsize=12, y=1.01)
    fig.tight_layout(w_pad=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--n_per_species", type=int, default=300,
                        help="Sequences per species (300 → 300×300=90k pairs per model)")
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grampa = PROJECT_ROOT / "data" / "grampa.csv"
    N = args.n_per_species

    print(f"Loading {N} sequences per species...")
    species_data = {}
    for sp in set(s for pair in SPECIES_PAIRS for s in pair):
        seqs, mics = load_species(grampa, sp, n_max=N)
        species_data[sp] = (seqs, np.array(mics))
        print(f"  {sp}: {len(seqs)} seqs")

    all_rhos      = {m: {} for m in args.models}
    all_pair_res  = {pair: {} for pair in [f"{s}→{t}" for s,t in SPECIES_PAIRS]}

    for model_name in args.models:
        print(f"\n{'='*50}\n{MODEL_LABELS.get(model_name, model_name)}\n{'='*50}")
        embed_fn = load_embedder(model_name, device)

        # precompute all needed embeddings
        emb_cache = {}
        needed_sp = set(s for pair in SPECIES_PAIRS for s in pair)
        for sp in needed_sp:
            seqs, _ = species_data[sp]
            print(f"  Embedding {sp}...")
            emb_cache[sp] = embed_batch(embed_fn, seqs)

        for src, tgt in SPECIES_PAIRS:
            pk = f"{src}→{tgt}"
            emb_src = emb_cache[src]; mics_src = species_data[src][1]
            emb_tgt = emb_cache[tgt]; mics_tgt = species_data[tgt][1]

            print(f"  Computing {pk} ({N}×{N}={N*N} pairs)...")
            _, _, centers, means, stds, rho = compute_cross_species_sim_mic(
                emb_src, mics_src, emb_tgt, mics_tgt, n_bins=args.n_bins
            )
            all_rhos[model_name][pk] = rho
            all_pair_res[pk][model_name] = {
                "centers": centers, "means": means, "stds": stds, "rho": rho
            }
            print(f"    ρ = {rho:.4f}")

        del embed_fn
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # per-pair plots
    print("\nPlotting per-pair curves...")
    for src, tgt in SPECIES_PAIRS:
        pk = f"{src}→{tgt}"
        fname = f"emb_sim_vs_mic_{SP_SHORT.get(src,src)}_{SP_SHORT.get(tgt,tgt)}.png"
        plot_pair(all_pair_res[pk], src, tgt, OUT / fname)

    # summary
    print("Plotting summary...")
    plot_summary(all_rhos, OUT / "emb_sim_summary.png")

    print(f"\nDone. → {OUT.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
