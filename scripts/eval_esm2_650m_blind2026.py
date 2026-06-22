"""
ESM-2 650M evaluation on blind-2026 temporal test set (eLife 2025 Supplementary File 2).

Protocol: frozen ESM-2 650M embeddings + MLP regression head trained on GRAMPA
E. coli train split (same protocol as cross_species_transfer.py for fair comparison).
Evaluates on 104 held-out sequences from eLife 2025 Supp. File 2 (zero GRAMPA overlap).

Appends esm2_650m entry to eval_results/external_elife2025_supp2_mic.json.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SUPP2_JSON = PROJECT_ROOT / "eval_results" / "external_elife2025_supp2_mic.json"
GRAMPA_CSV = PROJECT_ROOT / "data" / "grampa.csv"
AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS = [42, 123, 7]


# ── data ──────────────────────────────────────────────────────────────────────

def load_grampa_ecoli(seed: int = 42) -> tuple[list, list]:
    recs = []
    with open(GRAMPA_CSV) as f:
        for r in csv.DictReader(f):
            seq = r["sequence"].strip().upper()
            if (
                r["is_modified"].strip() == "False"
                and r["bacterium"].strip() == "E. coli"
                and 3 <= len(seq) <= 50
                and all(c in AA for c in seq)
            ):
                try:
                    recs.append({"seq": seq, "log2_mic": float(r["value"])})
                except ValueError:
                    continue

    unique_seqs = sorted({r["seq"] for r in recs})
    rng = random.Random(seed)
    rng.shuffle(unique_seqs)
    n = len(unique_seqs)
    n_test = max(1, int(n * 0.15))
    n_val  = max(1, int(n * 0.10))
    test_seqs = set(unique_seqs[:n_test])
    val_seqs  = set(unique_seqs[n_test:n_test + n_val])

    train, val = [], []
    for r in recs:
        if r["seq"] in test_seqs or r["seq"] in val_seqs:
            val.append(r)
        else:
            train.append(r)
    return train, val


# ── embedder ──────────────────────────────────────────────────────────────────

class ESM2Embedder650M(nn.Module):
    def __init__(self, device):
        super().__init__()
        from src.models.esm_head import load_esm2
        self.esm, self.alphabet, d = load_esm2("esm2_t33_650M")
        self.bc = self.alphabet.get_batch_converter()
        self.d_model = d
        self.num_layers = self.esm.num_layers
        for p in self.esm.parameters():
            p.requires_grad_(False)

    def forward(self, seqs: list[str], device) -> torch.Tensor:
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = self.bc(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = self.esm(tokens, repr_layers=[self.num_layers], return_contacts=False)
        h = out["representations"][self.num_layers]
        pad = tokens == self.alphabet.padding_idx
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
        return h.sum(1) / lengths


# ── head ──────────────────────────────────────────────────────────────────────

class MICHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x): return self.net(x).squeeze(-1)


def train_head(embedder, head, train_recs, val_recs, device,
               epochs: int = 60, batch_size: int = 64, lr: float = 3e-4,
               patience: int = 12) -> None:
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    best_val, wait = float("inf"), 0
    best_state = None

    def run_epoch(recs, train: bool):
        random.shuffle(recs)
        losses = []
        for i in range(0, len(recs), batch_size):
            batch = recs[i:i + batch_size]
            seqs = [r["seq"] for r in batch]
            y = torch.tensor([r["log2_mic"] for r in batch],
                              dtype=torch.float32, device=device)
            with torch.set_grad_enabled(train):
                with torch.no_grad():
                    emb = embedder(seqs, device)
                pred = head(emb)
                loss = F.huber_loss(pred, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    head.train()
    for ep in range(epochs):
        run_epoch(train_recs, train=True)
        head.eval()
        val_loss = run_epoch(val_recs, train=False)
        head.train()
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        head.load_state_dict(best_state)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    supp2 = json.loads(SUPP2_JSON.read_text())
    test_recs = [
        {"seq": r["sequence"], "log2_mic": r["log2_mic_true"]}
        for r in supp2["per_sequence"]
    ]
    print(f"Blind-2026 test: {len(test_recs)} sequences")

    print("Loading ESM-2 650M...")
    embedder = ESM2Embedder650M(device).to(device).eval()
    print(f"  d_model={embedder.d_model}, layers={embedder.num_layers}")

    seed_pearsons, seed_spearmans = [], []
    per_seq_preds: list[list[float]] = []

    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        train_recs, val_recs = load_grampa_ecoli(seed=seed)
        print(f"  GRAMPA E. coli train={len(train_recs)}, val={len(val_recs)}")

        head = MICHead(embedder.d_model).to(device)
        train_head(embedder, head, train_recs, val_recs, device)

        head.eval()
        preds = []
        for i in range(0, len(test_recs), 32):
            batch = test_recs[i:i + 32]
            seqs = [r["seq"] for r in batch]
            with torch.no_grad():
                emb = embedder(seqs, device)
                p = head(emb).cpu().numpy()
            preds.extend(p.tolist())

        trues = [r["log2_mic"] for r in test_recs]
        r, _   = pearsonr(preds, trues)
        rho, _ = spearmanr(preds, trues)
        print(f"  Pearson={r:.4f}  Spearman={rho:.4f}")
        seed_pearsons.append(r)
        seed_spearmans.append(rho)
        per_seq_preds.append(preds)

    mean_r   = float(np.mean(seed_pearsons))
    std_r    = float(np.std(seed_pearsons))
    mean_rho = float(np.mean(seed_spearmans))
    std_rho  = float(np.std(seed_spearmans))

    print(f"\n=== ESM-2 650M Blind-2026 ===")
    print(f"Pearson:  {mean_r:.4f} ± {std_r:.4f}")
    print(f"Spearman: {mean_rho:.4f} ± {std_rho:.4f}")

    # Update supp2 JSON with 650M results
    supp2["metrics"]["ESM-2 650M (frozen+head)"] = {
        "pearson":  mean_r,
        "spearman": mean_rho,
        "pearson_std":  std_r,
        "spearman_std": std_rho,
        "seeds": {
            str(s): {"pearson": float(p), "spearman": float(r)}
            for s, p, r in zip(SEEDS, seed_pearsons, seed_spearmans)
        },
        "protocol": "frozen ESM-2 650M + MLP head (256 hidden, Huber loss, 60 epochs) trained on GRAMPA E.coli, 3-seed mean",
    }
    # Append per-sequence predictions
    avg_preds = np.mean(per_seq_preds, axis=0).tolist()
    for i, r in enumerate(supp2["per_sequence"]):
        r["esm2_650m_pred"] = round(avg_preds[i], 4)

    SUPP2_JSON.write_text(json.dumps(supp2, indent=2))
    print(f"\nSaved results to {SUPP2_JSON}")


if __name__ == "__main__":
    main()
