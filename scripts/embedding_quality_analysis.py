"""
JEPA vs ESM2 embedding quality analysis.

Tests the hypothesis that JEPA's self-supervised objective (predicting masked
patches in *embedding space*) produces representations more aligned with
antimicrobial activity than ESM2's masked language modelling objective
(predicting masked *tokens* in sequence space).

Analyses:
  1. k-NN MIC prediction from frozen embeddings (k=5,10,20) on GRAMPA test set
  2. Linear probe: AUROC for AMP/non-AMP from frozen embeddings (AMPlify test set)
  3. Silhouette score: do AMP vs non-AMP cluster better in JEPA or ESM2 space?
  4. MIC rank correlation: Spearman ρ between embedding-space nearest-neighbour
     mean MIC and true MIC — measures how well local structure reflects activity
  5. Intra-class vs inter-class embedding distance ratio

Outputs:
  eval_results/embedding_quality/
    metrics.json      — all numeric results
    SUMMARY.md        — human-readable table
    knn_mic_scatter_jepa.png
    knn_mic_scatter_esm2.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = PROJECT_ROOT / "eval_results" / "embedding_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"


# ── encoders ──────────────────────────────────────────────────────────────────

def _batch_encode(seqs: list[str]) -> torch.Tensor:
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:50]) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=DEVICE)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out


def encode_jepa_batch(seqs: list[str]) -> np.ndarray:
    from src.models.jepa import JEPA

    ckpt = torch.load(PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
                      map_location=DEVICE, weights_only=False)
    jepa = JEPA(**ckpt["cfg"]["model"]).to(DEVICE)
    jepa.load_state_dict(ckpt["model_state"])
    enc = jepa.context_encoder.eval()

    all_emb = []
    for i in range(0, len(seqs), 256):
        ids = _batch_encode(seqs[i:i+256])
        with torch.no_grad():
            h = enc(ids)
        pad = (ids == 0)
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float()
        all_emb.append((h.sum(1) / lengths.clamp(min=1)).cpu().float().numpy())
    return np.concatenate(all_emb, 0)


def encode_esm2_batch(seqs: list[str]) -> np.ndarray:
    """Mean-pool ESM-2 (35M) via fair-esm."""
    from src.models.esm_head import load_esm2

    esm_model, alphabet, _ = load_esm2("esm2_t12_35M")
    esm_model = esm_model.to(DEVICE).eval()
    bc = alphabet.get_batch_converter()

    all_emb = []
    for i in range(0, len(seqs), 64):
        data = [(f"s{j}", s) for j, s in enumerate(seqs[i:i+64])]
        _, _, tokens = bc(data)
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[12], return_contacts=False)
        h = out["representations"][12]
        pad = (tokens == alphabet.padding_idx)
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float()
        all_emb.append((h.sum(1) / lengths.clamp(min=1)).cpu().float().numpy())
    return np.concatenate(all_emb, 0)


# ── data ──────────────────────────────────────────────────────────────────────

def load_grampa_splits() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Returns (train_seqs, train_mic, test_seqs, test_mic) mean log2 MIC."""
    import csv
    seq_vals: dict[str, list[float]] = defaultdict(list)
    with open(PROJECT_ROOT / "data" / "grampa.csv") as f:
        for r in csv.DictReader(f):
            seq_vals[r["sequence"]].append(float(r["value"]))
    seqs = list(seq_vals.keys())
    mic  = np.array([np.mean(v) for v in seq_vals.values()])
    rng  = np.random.default_rng(42)
    idx  = rng.permutation(len(seqs))
    n_test = int(0.1 * len(seqs))
    test_seqs  = [seqs[i] for i in idx[:n_test]]
    test_mic   = mic[idx[:n_test]]
    train_seqs = [seqs[i] for i in idx[n_test:]]
    train_mic  = mic[idx[n_test:]]
    return train_seqs, train_mic, test_seqs, test_mic


def load_amp_classification() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Returns (train_seqs, train_labels, test_seqs, test_labels)."""
    import json
    pred_path = (PROJECT_ROOT / "eval_results" / "amp_classification_evidence"
                 / "predictions.jsonl")
    recs   = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]
    seen, seqs, label_list = set(), [], []
    for r in recs:
        s = r.get("sequence", "")
        if s and s not in seen:
            seen.add(s); seqs.append(s); label_list.append(int(r["label"]))
    labels = np.array(label_list)
    # 70/30 split for probe training vs evaluation
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(seqs))
    n_tr = int(0.7 * len(seqs))
    tr_seqs = [seqs[i] for i in idx[:n_tr]];  tr_lbl = labels[idx[:n_tr]]
    te_seqs = [seqs[i] for i in idx[n_tr:]];  te_lbl = labels[idx[n_tr:]]
    return tr_seqs, tr_lbl, te_seqs, te_lbl


# ── analyses ──────────────────────────────────────────────────────────────────

def knn_mic_prediction(train_emb, train_mic, test_emb, test_mic,
                       ks=(5, 10, 20)) -> dict:
    """k-NN MIC prediction: Pearson + Spearman for each k."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics.pairwise import cosine_distances
    results = {}
    D = cosine_distances(test_emb, train_emb)   # (n_test, n_train)
    for k in ks:
        nn_idx = np.argsort(D, axis=1)[:, :k]
        pred   = train_mic[nn_idx].mean(axis=1)
        r, _   = pearsonr(pred, test_mic)
        rho, _ = spearmanr(pred, test_mic)
        results[f"k{k}"] = {"pearson": float(r), "spearman": float(rho)}
    return results


def linear_probe_auroc(train_emb, train_labels, test_emb, test_labels) -> float:
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(train_emb)
    X_te   = scaler.transform(test_emb)
    clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr, train_labels)
    proba  = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(test_labels, proba))


def silhouette(emb, labels, sample=2000) -> float:
    if len(emb) > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(emb), sample, replace=False)
        emb, labels = emb[idx], labels[idx]
    return float(silhouette_score(emb, labels, metric="cosine"))


def mic_knn_rank_correlation(train_emb, train_mic, test_emb, test_mic, k=10):
    """Spearman ρ between kNN-mean-MIC prediction and true MIC."""
    from scipy.stats import spearmanr
    from sklearn.metrics.pairwise import cosine_distances
    D   = cosine_distances(test_emb, train_emb)
    idx = np.argsort(D, axis=1)[:, :k]
    pred = train_mic[idx].mean(1)
    rho, _ = spearmanr(pred, test_mic)
    return float(rho)


def plot_knn_scatter(pred, true, label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, pred, s=6, alpha=0.4, c="#1f77b4")
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    from scipy.stats import pearsonr, spearmanr
    r, _ = pearsonr(pred, true)
    rho, _ = spearmanr(pred, true)
    ax.set_xlabel("True mean log₂ MIC"); ax.set_ylabel("k-NN predicted log₂ MIC")
    ax.set_title(f"{label}\nPearson={r:.3f}  Spearman={rho:.3f}")
    plt.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  scatter saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    metrics: dict = {"jepa": {}, "esm2": {}}

    # ── MIC panel ─────────────────────────────────────────────────────────────
    print("Loading GRAMPA …")
    tr_seqs, tr_mic, te_seqs, te_mic = load_grampa_splits()
    # subsample train for speed (cosine dist matrix)
    cap = 3000
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(tr_seqs), min(cap, len(tr_seqs)), replace=False)
    tr_seqs_s = [tr_seqs[i] for i in tr_idx]
    tr_mic_s  = tr_mic[tr_idx]

    print(f"  train_sub={len(tr_seqs_s)}, test={len(te_seqs)}")

    for name, enc_fn in [("jepa", encode_jepa_batch), ("esm2", encode_esm2_batch)]:
        print(f"\n── {name.upper()} embeddings ──")
        print("  Encoding train …")
        tr_emb = enc_fn(tr_seqs_s)
        print("  Encoding test  …")
        te_emb = enc_fn(te_seqs)

        print("  k-NN MIC …")
        knn_res = knn_mic_prediction(tr_emb, tr_mic_s, te_emb, te_mic)
        metrics[name]["knn_mic"] = knn_res

        # best-k scatter
        k_best = max(knn_res, key=lambda k: knn_res[k]["pearson"])
        from sklearn.metrics.pairwise import cosine_distances
        D = cosine_distances(te_emb, tr_emb)
        nn_idx = np.argsort(D, axis=1)[:, :int(k_best[1:])]
        pred = tr_mic_s[nn_idx].mean(1)
        plot_knn_scatter(pred, te_mic, f"{name.upper()} k-NN MIC (k={k_best[1:]})",
                         OUT_DIR / f"knn_mic_scatter_{name}.png")

    # ── AMP classification panel ───────────────────────────────────────────────
    print("\nLoading AMPlify classification set …")
    tr_amp_s, tr_amp_l, te_amp_s, te_amp_l = load_amp_classification()

    # (split already done in load_amp_classification)

    print(f"  train={len(tr_amp_s)}, test={len(te_amp_s)}")

    for name, enc_fn in [("jepa", encode_jepa_batch), ("esm2", encode_esm2_batch)]:
        print(f"\n── {name.upper()} classification ──")
        tr_emb = enc_fn(tr_amp_s)
        te_emb = enc_fn(te_amp_s)

        print("  Linear probe AUROC …")
        auc = linear_probe_auroc(tr_emb, tr_amp_l, te_emb, te_amp_l)
        metrics[name]["linear_probe_auroc"] = auc
        print(f"  AUROC = {auc:.4f}")

        print("  Silhouette …")
        combined_emb = np.concatenate([tr_emb, te_emb], 0)
        combined_lbl = np.concatenate([tr_amp_l, te_amp_l], 0)
        sil = silhouette(combined_emb, combined_lbl)
        metrics[name]["silhouette_amp"] = sil
        print(f"  Silhouette = {sil:.4f}")

    # ── summary ───────────────────────────────────────────────────────────────
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "# Embedding Quality: JEPA-AMP vs ESM-2\n",
        "## k-NN MIC Prediction (GRAMPA, Pearson / Spearman)\n",
        "| Model | k=5 | k=10 | k=20 |",
        "|---|---|---|---|",
    ]
    for name in ("jepa", "esm2"):
        r = metrics[name]["knn_mic"]
        lines.append(
            f"| {name.upper()} | "
            f"{r['k5']['pearson']:.3f} / {r['k5']['spearman']:.3f} | "
            f"{r['k10']['pearson']:.3f} / {r['k10']['spearman']:.3f} | "
            f"{r['k20']['pearson']:.3f} / {r['k20']['spearman']:.3f} |"
        )
    lines += [
        "\n## Linear Probe AUROC (AMPlify test set)\n",
        "| Model | AUROC | Silhouette (AMP/non-AMP) |",
        "|---|---|---|",
    ]
    for name in ("jepa", "esm2"):
        lines.append(
            f"| {name.upper()} | "
            f"{metrics[name]['linear_probe_auroc']:.4f} | "
            f"{metrics[name]['silhouette_amp']:.4f} |"
        )
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\nAll done. Results in", OUT_DIR)


if __name__ == "__main__":
    main()
