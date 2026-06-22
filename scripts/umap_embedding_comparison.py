"""
UMAP embedding comparison: JEPA-AMP vs ESM-2.

Extracts embeddings from JEPA and ESM2 for:
  - AMPlify test set (835 AMP + 835 non-AMP)
  - GRAMPA test sequences (with MIC labels)
  - Generated peptides (from formal generation run)

Produces 4 figures in eval_results/umap/:
  1. umap_jepa_amp_class.png        — JEPA, colored by AMP/non-AMP
  2. umap_esm2_amp_class.png        — ESM2, colored by AMP/non-AMP
  3. umap_jepa_mic.png              — JEPA GRAMPA test, colored by mean log2 MIC
  4. umap_esm2_mic.png              — ESM2 GRAMPA test, colored by mean log2 MIC
  5. umap_jepa_generated_overlay.png — JEPA: training AMPs + generated peptides
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = PROJECT_ROOT / "eval_results" / "umap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"


# ── helpers ──────────────────────────────────────────────────────────────────

def _batch_encode(seqs: list[str], max_len: int = 50) -> torch.Tensor:
    """Encode a batch of sequences to padded token ids (B, max_len+2)."""
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:max_len]) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=DEVICE)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out


def encode_jepa(seqs: list[str]) -> np.ndarray:
    """Mean-pool JEPA context encoder output. Returns (N, d_model)."""
    from src.models.jepa import JEPA

    ckpt_path = PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    jepa = JEPA(**ckpt["cfg"]["model"]).to(DEVICE)
    jepa.load_state_dict(ckpt["model_state"])
    enc = jepa.context_encoder.eval()

    batch_size = 256
    all_emb = []
    for i in range(0, len(seqs), batch_size):
        ids = _batch_encode(seqs[i : i + batch_size])          # (B, L)
        with torch.no_grad():
            h = enc(ids)                                         # (B, L, d)
        pad_mask = (ids == 0)
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        lengths = (~pad_mask).sum(1, keepdim=True).float()
        emb = h.sum(1) / lengths.clamp(min=1)
        all_emb.append(emb.cpu().float().numpy())
    return np.concatenate(all_emb, axis=0)


def encode_esm2(seqs: list[str]) -> np.ndarray:
    """Mean-pool ESM-2 (35M) embeddings using fair-esm. Returns (N, 480)."""
    from src.models.esm_head import load_esm2

    esm_model, alphabet, _ = load_esm2("esm2_t12_35M")
    esm_model = esm_model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()

    batch_size = 64
    all_emb = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        data   = [(f"seq{j}", s) for j, s in enumerate(batch)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[12], return_contacts=False)
        h = out["representations"][12]          # (B, L, d)
        # mask padding (token 1 = padding in ESM alphabet)
        pad = (tokens == alphabet.padding_idx)
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float()
        emb = h.sum(1) / lengths.clamp(min=1)
        all_emb.append(emb.cpu().float().numpy())
    return np.concatenate(all_emb, axis=0)


def load_amplify_test() -> tuple[list[str], list[int]]:
    """Load sequences + labels from the classification evidence artifact."""
    import json
    pred_path = (PROJECT_ROOT / "eval_results" / "amp_classification_evidence"
                 / "predictions.jsonl")
    recs   = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]
    # deduplicate by sequence, keep one record per sequence
    seen, seqs, labels = set(), [], []
    for r in recs:
        s = r.get("sequence", "")
        if s and s not in seen:
            seen.add(s)
            seqs.append(s)
            labels.append(int(r["label"]))
    return seqs, labels


def load_grampa_test() -> tuple[list[str], np.ndarray]:
    """Load GRAMPA test sequences and mean log2 MIC across bacteria."""
    import csv
    rows = []
    with open(PROJECT_ROOT / "data" / "grampa.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    from collections import defaultdict
    seq_vals: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        seq_vals[r["sequence"]].append(float(r["value"]))
    seqs = list(seq_vals.keys())
    mic_means = np.array([np.mean(v) for v in seq_vals.values()])
    return seqs, mic_means


def load_generated() -> list[str]:
    """Load generated peptides from formal MIC-conditioned generation."""
    path = PROJECT_ROOT / "eval_results" / "generation_control_formal" / "predictions.jsonl"
    seqs = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("valid") and obj.get("sequence"):
                seqs.append(obj["sequence"])
    return seqs[:500]  # cap for UMAP speed


def plot_umap(emb2d: np.ndarray, colors, title: str, out_path: Path,
              cmap="tab10", colorbar=False, labels=None, alpha=0.55,
              s=8) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=colors, cmap=cmap,
                    s=s, alpha=alpha, linewidths=0)
    if colorbar:
        plt.colorbar(sc, ax=ax, label="mean log₂ MIC")
    if labels is not None:
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=cm.get_cmap(cmap)(i / max(1, len(labels) - 1)),
                              markersize=7, label=l) for i, l in enumerate(labels)]
        ax.legend(handles=handles, fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    reducer_kw = dict(n_neighbors=30, min_dist=0.1, random_state=42, n_jobs=4)

    # ── 1. AMP classification panel ──────────────────────────────────────────
    print("Loading AMPlify test set …")
    try:
        amp_seqs, amp_labels = load_amplify_test()
    except Exception:
        # fallback: use classifier_benchmark predictions
        print("  fallback: using amp_classification_evidence sequences")
        import jsonlines
        records = list(jsonlines.open(PROJECT_ROOT /
            "eval_results/amp_classification_evidence/predictions.jsonl"))
        amp_seqs  = [r["sequence"] for r in records if "sequence" in r]
        amp_labels = [int(r["true_label"]) for r in records if "sequence" in r]

    print(f"  {len(amp_seqs)} sequences (AMP={sum(amp_labels)}, non-AMP={len(amp_labels)-sum(amp_labels)})")

    print("Encoding with JEPA …")
    jepa_amp = encode_jepa(amp_seqs)
    print("Encoding with ESM2 …")
    esm2_amp = encode_esm2(amp_seqs)

    colors_cls = np.array(amp_labels, dtype=float)

    print("UMAP for AMP/non-AMP …")
    u_jepa_amp = umap.UMAP(**reducer_kw).fit_transform(jepa_amp)
    plot_umap(u_jepa_amp, colors_cls, "JEPA-AMP embeddings: AMP vs non-AMP",
              OUT_DIR / "umap_jepa_amp_class.png", cmap="RdYlGn",
              labels=["non-AMP", "AMP"])

    u_esm2_amp = umap.UMAP(**reducer_kw).fit_transform(esm2_amp)
    plot_umap(u_esm2_amp, colors_cls, "ESM-2 embeddings: AMP vs non-AMP",
              OUT_DIR / "umap_esm2_amp_class.png", cmap="RdYlGn",
              labels=["non-AMP", "AMP"])

    # ── 2. MIC regression panel ───────────────────────────────────────────────
    print("Loading GRAMPA test set …")
    grampa_seqs, mic_means = load_grampa_test()
    print(f"  {len(grampa_seqs)} unique sequences")

    # cap for speed
    cap = min(len(grampa_seqs), 3000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(grampa_seqs), cap, replace=False)
    grampa_seqs_sub = [grampa_seqs[i] for i in idx]
    mic_sub = mic_means[idx]

    print("Encoding GRAMPA with JEPA …")
    jepa_mic = encode_jepa(grampa_seqs_sub)
    print("Encoding GRAMPA with ESM2 …")
    esm2_mic = encode_esm2(grampa_seqs_sub)

    print("UMAP for MIC …")
    u_jepa_mic = umap.UMAP(**reducer_kw).fit_transform(jepa_mic)
    plot_umap(u_jepa_mic, mic_sub, "JEPA-AMP embeddings: mean log₂ MIC",
              OUT_DIR / "umap_jepa_mic.png", cmap="RdYlBu_r", colorbar=True)

    u_esm2_mic = umap.UMAP(**reducer_kw).fit_transform(esm2_mic)
    plot_umap(u_esm2_mic, mic_sub, "ESM-2 embeddings: mean log₂ MIC",
              OUT_DIR / "umap_esm2_mic.png", cmap="RdYlBu_r", colorbar=True)

    # ── 3. Generated peptide overlay ──────────────────────────────────────────
    print("Loading generated peptides …")
    gen_seqs = load_generated()
    print(f"  {len(gen_seqs)} generated sequences")

    # subsample training AMPs for background
    train_fasta = PROJECT_ROOT / "data/processed/amp_corpus.fasta"
    train_seqs = []
    with open(train_fasta) as f:
        for line in f:
            if not line.startswith(">"):
                s = line.strip()
                if s and all(c in AA for c in s):
                    train_seqs.append(s)
    rng2 = np.random.default_rng(0)
    # Use 2000 training AMPs for better coverage; also sample generated from
    # diverse conditions (not just generation_control)
    train_sub = [train_seqs[i] for i in rng2.choice(len(train_seqs), 2000, replace=False)]

    combined = train_sub + gen_seqs
    colors_overlay = np.array([0] * len(train_sub) + [1] * len(gen_seqs), dtype=float)

    print("Encoding combined with JEPA …")
    jepa_combined = encode_jepa(combined)
    print("UMAP for generated overlay …")
    u_combined = umap.UMAP(**reducer_kw).fit_transform(jepa_combined)
    plot_umap(u_combined, colors_overlay,
              "JEPA-AMP: training corpus vs generated peptides",
              OUT_DIR / "umap_jepa_generated_overlay.png",
              cmap="coolwarm", labels=["Training AMP", "Generated"], alpha=0.6)

    # ── save summary ─────────────────────────────────────────────────────────
    summary = {
        "n_amp_test": len(amp_seqs),
        "n_grampa_sub": cap,
        "n_generated": len(gen_seqs),
        "n_train_sub": len(train_sub),
        "output_dir": str(OUT_DIR),
        "figures": [
            "umap_jepa_amp_class.png",
            "umap_esm2_amp_class.png",
            "umap_jepa_mic.png",
            "umap_esm2_mic.png",
            "umap_jepa_generated_overlay.png",
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. All UMAP figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
