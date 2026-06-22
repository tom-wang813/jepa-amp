"""
Physicochemical property probing of frozen JEPA-AMP and ESM-2 embeddings.

For each sequence in GRAMPA, computes six physicochemical properties:
  - net_charge   : sum of K+R-D-E at pH 7 (integer approximation)
  - gravy        : Grand Average of Hydropathicity (Kyte-Doolittle)
  - helix        : Chou-Fasman helix propensity (fraction of helix-promoting AAs)
  - length       : sequence length
  - mol_weight   : molecular weight (kDa)
  - aromaticity  : fraction of F+W+Y

Then fits Ridge regression from frozen embeddings to each property and reports R².
This mechanistically explains why charge control succeeds but GRAVY/length fail:
if a property is not linearly decodable from the embedding, it cannot be directly
controlled via a linear conditioning signal.

Outputs:
    eval_results/physicochemical_probe/
        metrics.json     – R² per model × property
        SUMMARY.md       – table ready for paper / supplement
        probe_r2_bar.png – bar chart
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = PROJECT_ROOT / "eval_results" / "physicochemical_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"

# Chou-Fasman helix-promoting AAs
HELIX_AAS = set("AELM")

# ── physicochemical features ──────────────────────────────────────────────────

def compute_properties(seq: str) -> dict[str, float]:
    pa = ProteinAnalysis(seq)
    charge = sum(seq.count(aa) * val for aa, val in
                 [("K", 1), ("R", 1), ("D", -1), ("E", -1)])
    helix = sum(1 for aa in seq if aa in HELIX_AAS) / max(len(seq), 1)
    return {
        "net_charge":  float(charge),
        "gravy":       float(pa.gravy()),
        "helix":       float(helix),
        "length":      float(len(seq)),
        "mol_weight":  float(pa.molecular_weight()) / 1000.0,
        "aromaticity": float(pa.aromaticity()),
    }


# ── embedding extraction (reuses same approach as embedding_quality_analysis) ─

def _batch_encode(seqs: list[str]) -> torch.Tensor:
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:50]) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=DEVICE)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out


def encode_jepa(seqs: list[str], batch_size: int = 256) -> np.ndarray:
    from src.models.jepa import JEPA
    ckpt = torch.load(
        PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
        map_location=DEVICE, weights_only=False,
    )
    jepa = JEPA(**ckpt["cfg"]["model"]).to(DEVICE)
    jepa.load_state_dict(ckpt["model_state"])
    enc = jepa.context_encoder.eval()

    embs = []
    for i in range(0, len(seqs), batch_size):
        ids = _batch_encode(seqs[i : i + batch_size])
        with torch.no_grad():
            h = enc(ids)
        pad = ids == 0
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
        embs.append((h.sum(1) / lengths).cpu().float().numpy())
    return np.concatenate(embs, 0)


def encode_esm2(seqs: list[str], batch_size: int = 64) -> np.ndarray:
    from src.models.esm_head import load_esm2
    esm_model, alphabet, _ = load_esm2("esm2_t12_35M")
    esm_model = esm_model.to(DEVICE).eval()
    bc = alphabet.get_batch_converter()

    embs = []
    for i in range(0, len(seqs), batch_size):
        data = [(f"s{j}", s) for j, s in enumerate(seqs[i : i + batch_size])]
        _, _, tokens = bc(data)
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[12], return_contacts=False)
        h = out["representations"][12]
        pad = tokens == alphabet.padding_idx
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
        embs.append((h.sum(1) / lengths).cpu().float().numpy())
    return np.concatenate(embs, 0)


# ── data loading ──────────────────────────────────────────────────────────────

def load_sequences() -> list[str]:
    seqs: set[str] = set()
    with open(PROJECT_ROOT / "data" / "grampa.csv") as f:
        for r in csv.DictReader(f):
            seq = r["sequence"].strip().upper()
            if (
                r["is_modified"].strip() == "False"
                and 3 <= len(seq) <= 50
                and all(c in AA for c in seq)
            ):
                seqs.add(seq)
    return sorted(seqs)


# ── probing ───────────────────────────────────────────────────────────────────

def probe(emb: np.ndarray, props: np.ndarray, prop_names: list[str],
          n_train: int) -> dict[str, float]:
    """Ridge regression R² for each property. Train/test split at n_train."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(emb[:n_train])
    X_te = scaler.transform(emb[n_train:])

    results = {}
    for i, name in enumerate(prop_names):
        y_tr = props[:n_train, i]
        y_te = props[n_train:, i]
        clf = Ridge(alpha=1.0)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        results[name] = float(r2_score(y_te, pred))
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(metrics: dict, prop_names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(prop_names))
    w = 0.35
    colors = {"jepa": "#1f77b4", "esm2": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, (model, color) in zip([-w / 2, w / 2],
                                       [("jepa", colors["jepa"]),
                                        ("esm2", colors["esm2"])]):
        vals = [metrics[model].get(p, 0.0) for p in prop_names]
        ax.bar(x + offset, vals, w, label=model.upper(), color=color, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(prop_names, rotation=20, ha="right")
    ax.set_ylabel("Ridge R²")
    ax.set_title("Physicochemical Probing: JEPA-AMP vs ESM-2 (Frozen)")
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "probe_r2_bar.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'probe_r2_bar.png'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading GRAMPA sequences …")
    seqs = load_sequences()
    print(f"  {len(seqs)} unique sequences")

    prop_names = ["net_charge", "gravy", "helix", "length", "mol_weight", "aromaticity"]
    print("Computing physicochemical properties …")
    prop_matrix = np.array([
        [compute_properties(s)[p] for p in prop_names]
        for s in seqs
    ], dtype=np.float32)
    print(f"  properties shape: {prop_matrix.shape}")

    # 80/20 split (no overlap with generation eval)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(seqs))
    seqs_shuf   = [seqs[i] for i in idx]
    props_shuf  = prop_matrix[idx]
    n_train = int(0.8 * len(seqs))

    metrics: dict = {}

    for model_name, enc_fn in [("jepa", encode_jepa), ("esm2", encode_esm2)]:
        print(f"\n── {model_name.upper()} embeddings ──")
        emb = enc_fn(seqs_shuf)
        print(f"  embedding shape: {emb.shape}")
        r2s = probe(emb, props_shuf, prop_names, n_train)
        metrics[model_name] = r2s
        for p, v in r2s.items():
            print(f"  {p:15s}  R² = {v:+.3f}")

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── SUMMARY.md ────────────────────────────────────────────────────────────
    lines = [
        "# Physicochemical Probing: Frozen Embedding R²",
        "",
        "Linear (Ridge) regression from mean-pooled frozen embeddings to each property.",
        "R² > 0.5 indicates the property is well-encoded; R² ≈ 0 means the embedding",
        "does not carry a linearly extractable signal for that property.",
        "",
        "| Property | JEPA-AMP R² | ESM-2 R² | Interpretation |",
        "|---|---|---|---|",
    ]
    interpretations = {
        "net_charge":  "direct conditioning target",
        "gravy":       "entangled with charge residues",
        "helix":       "secondary structure signal",
        "length":      "EOS decision (not global)",
        "mol_weight":  "correlated with length",
        "aromaticity": "compositional fraction",
    }
    for p in prop_names:
        j = metrics["jepa"][p]
        e = metrics["esm2"][p]
        interp = interpretations.get(p, "")
        lines.append(f"| {p} | {j:+.3f} | {e:+.3f} | {interp} |")

    lines += [
        "",
        "## Key finding",
        "",
        "Properties with high R² in the embedding are candidates for direct conditioning.",
        "Properties with low R² require decoupled objectives or different architectures.",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\n  wrote {OUT_DIR / 'SUMMARY.md'}")

    plot(metrics, prop_names)
    print(f"\nAll done. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
