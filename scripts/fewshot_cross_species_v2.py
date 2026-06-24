"""
Few-shot cross-species MIC transfer — v2: expanded species, saves per-sample predictions.

Protocol: warmstart only (source-trained head fine-tuned on k target examples).
k=0 = zero-shot (source head evaluated on target as-is).

Species: top-6 bacteria by count in GRAMPA (excluding fungi).
  E. coli (5465), S. aureus (5070), P. aeruginosa (2523),
  B. subtilis (1323), S. typhimurium (715), M. luteus (651)
  → 6×5 = 30 ordered pairs

Outputs:
    eval_results/fewshot_v2/{model}/
        metrics.json       -- {pair: {seed: {k: {pearson, spearman, rmse, n}}}}
        predictions.json   -- {pair: {seed: {k: {pred: [...], actual: [...]}}}}

Usage:
    uv run python scripts/fewshot_cross_species_v2.py --gpu 0 --model jepa
    uv run python scripts/fewshot_cross_species_v2.py --gpu 0 --model esm2
    uv run python scripts/fewshot_cross_species_v2.py --gpu 0 --model esm2_650m
    uv run python scripts/fewshot_cross_species_v2.py --gpu 0 --model mlm
"""

from __future__ import annotations

import argparse
import copy
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

AA = "ACDEFGHIKLMNPQRSTVWY"

SPECIES = [
    "E. coli",
    "S. aureus",
    "P. aeruginosa",
    "B. subtilis",
    "S. typhimurium",
    "M. luteus",
]
SPECIES_PAIRS = [
    (src, tgt) for src in SPECIES for tgt in SPECIES if src != tgt
]  # 30 pairs

SEEDS    = [42, 123, 7]
K_VALUES = [0, 5, 10, 20, 50, 100]


# ── data ──────────────────────────────────────────────────────────────────────

def load_species(csv_path: Path, species: str, seed: int = 42,
                 max_len: int = 50) -> tuple[list, list, list]:
    recs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            seq = r["sequence"].strip().upper()
            if (r["is_modified"].strip() == "False"
                    and r["bacterium"].strip() == species
                    and 3 <= len(seq) <= max_len
                    and all(c in AA for c in seq)):
                try:
                    recs.append({"seq": seq, "log2_mic": float(r["value"])})
                except ValueError:
                    continue

    unique_seqs = sorted({r["seq"] for r in recs})
    rng = random.Random(seed)
    rng.shuffle(unique_seqs)
    n      = len(unique_seqs)
    n_test = max(1, int(n * 0.15))
    n_val  = max(1, int(n * 0.10))
    test_s = set(unique_seqs[:n_test])
    val_s  = set(unique_seqs[n_test:n_test + n_val])

    train, val, test = [], [], []
    for r in recs:
        if r["seq"] in test_s:   test.append(r)
        elif r["seq"] in val_s:  val.append(r)
        else:                    train.append(r)
    return train, val, test


# ── embedders ─────────────────────────────────────────────────────────────────

def _batch_encode_jepa(seqs: list[str], device) -> torch.Tensor:
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:50]) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out


class JEPAEmbedder(nn.Module):
    def __init__(self, device):
        super().__init__()
        from src.models.jepa import JEPA
        ckpt = torch.load(
            PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
            map_location=device, weights_only=False,
        )
        jepa = JEPA(**ckpt["cfg"]["model"])
        jepa.load_state_dict(ckpt["model_state"])
        self.enc = jepa.context_encoder
        self.d_model = ckpt["cfg"]["model"]["d_model"]
        for p in self.enc.parameters():
            p.requires_grad_(False)

    def forward(self, seqs: list[str], device) -> torch.Tensor:
        ids = _batch_encode_jepa(seqs, device)
        h = self.enc(ids)
        pad = ids == 0
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
        return h.sum(1) / lengths


class ESM2Embedder(nn.Module):
    def __init__(self, device, model_key: str = "esm2_t12_35M"):
        super().__init__()
        from src.models.esm_head import load_esm2
        self.esm, self.alphabet, d = load_esm2(model_key)
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
            out = self.esm(tokens, repr_layers=[self.num_layers],
                           return_contacts=False)
        h = out["representations"][self.num_layers]
        pad = tokens == self.alphabet.padding_idx
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
        return h.sum(1) / lengths


class MLMEmbedder(nn.Module):
    def __init__(self, device):
        super().__init__()
        from src.models.mlm import MLMModel
        ckpt_path = PROJECT_ROOT / "checkpoints/mlm_pretrain_868k/best_mlm.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = MLMModel(**ckpt["cfg"]["model"])
        enc_state = {k[len("encoder."):]: v for k, v in ckpt["model_state"].items()
                     if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_state)
        self.enc = model.encoder
        self.d_model = ckpt["cfg"]["model"]["d_model"]
        for p in self.enc.parameters():
            p.requires_grad_(False)

    def forward(self, seqs: list[str], device) -> torch.Tensor:
        ids = _batch_encode_jepa(seqs, device)
        h = self.enc(ids)
        pad = ids == 0
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


# ── train / eval ──────────────────────────────────────────────────────────────

def precompute_embs(embedder, recs, device, batch=256):
    embs = []
    for i in range(0, len(recs), batch):
        seqs = [r["seq"] for r in recs[i:i + batch]]
        embs.append(embedder(seqs, device))
    return torch.cat(embs, 0)


def train_head(embs_tr, y_tr, embs_val, y_val, d_model, device,
               epochs=60, batch_size=128, lr=3e-4, patience=12):
    head = MICHead(d_model).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    best_val, wait, best_state = float("inf"), 0, None
    idx = list(range(len(y_tr)))

    for _ in range(epochs):
        head.train()
        random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            loss = F.huber_loss(head(embs_tr[b]), y_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vl = F.huber_loss(head(embs_val), y_val).item()
        if vl < best_val - 1e-4:
            best_val = vl; wait = 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        head.load_state_dict(best_state)
    return head


def finetune_warmstart(source_head, embs_sup, y_sup, device,
                       epochs=200, lr=5e-5):
    head = copy.deepcopy(source_head).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    head.train()
    for _ in range(epochs):
        loss = F.huber_loss(head(embs_sup), y_sup)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def eval_head(head, embs_te, y_te_np):
    preds = head(embs_te).cpu().numpy()
    r,   _ = pearsonr(preds, y_te_np)
    rho, _ = spearmanr(preds, y_te_np)
    rmse   = float(np.sqrt(np.mean((preds - y_te_np) ** 2)))
    return (
        {"pearson": float(r), "spearman": float(rho), "rmse": rmse, "n": len(y_te_np)},
        {"pred": preds.tolist(), "actual": y_te_np.tolist()},
    )


# ── main ──────────────────────────────────────────────────────────────────────

MODEL_FACTORIES = {
    "jepa":      lambda dev: JEPAEmbedder(dev),
    "esm2":      lambda dev: ESM2Embedder(dev, "esm2_t12_35M"),
    "esm2_650m": lambda dev: ESM2Embedder(dev, "esm2_t33_650M"),
    "mlm":       lambda dev: MLMEmbedder(dev),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",   type=int, default=0)
    parser.add_argument("--model", default="jepa",
                        choices=list(MODEL_FACTORIES))
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Optionally restrict to specific pairs, e.g. 'E. coli→S. aureus'")
    args = parser.parse_args()

    device  = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grampa  = PROJECT_ROOT / "data" / "grampa.csv"
    out_dir = PROJECT_ROOT / "eval_results" / "fewshot_v2" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = out_dir / "metrics.json"
    preds_file   = out_dir / "predictions.json"

    metrics_all: dict = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    preds_all:   dict = json.loads(preds_file.read_text())   if preds_file.exists()   else {}

    pairs_to_run = SPECIES_PAIRS
    if args.pairs:
        pairs_to_run = [(s, t) for s, t in SPECIES_PAIRS
                        if f"{s}→{t}" in args.pairs]

    print(f"Model: {args.model}  |  device: {device}")
    print(f"Pairs: {len(pairs_to_run)}  |  Seeds: {SEEDS}  |  k: {K_VALUES}")

    embedder = MODEL_FACTORIES[args.model](device).to(device).eval()
    d        = embedder.d_model

    for src_sp, tgt_sp in pairs_to_run:
        pair_key  = f"{src_sp}→{tgt_sp}"
        pair_met  = metrics_all.setdefault(pair_key, {})
        pair_pred = preds_all.setdefault(pair_key, {})

        for seed in SEEDS:
            sk       = str(seed)
            seed_met  = pair_met.setdefault(sk, {})
            seed_pred = pair_pred.setdefault(sk, {})

            needed = [k for k in K_VALUES if str(k) not in seed_met]
            if not needed:
                print(f"  [skip] {pair_key}  seed={seed}")
                continue

            print(f"\n  {pair_key}  seed={seed}")
            src_tr, src_val, _ = load_species(grampa, src_sp, seed=seed)
            tgt_tr, _,      tgt_te = load_species(grampa, tgt_sp, seed=seed)
            print(f"    src_train={len(src_tr)}  tgt_train={len(tgt_tr)}  tgt_test={len(tgt_te)}")

            # precompute embeddings
            embs_src_tr  = precompute_embs(embedder, src_tr,  device)
            embs_src_val = precompute_embs(embedder, src_val, device)
            embs_tgt_te  = precompute_embs(embedder, tgt_te,  device)
            y_src_tr  = torch.tensor([r["log2_mic"] for r in src_tr],  dtype=torch.float32, device=device)
            y_src_val = torch.tensor([r["log2_mic"] for r in src_val], dtype=torch.float32, device=device)
            y_tgt_te  = np.array([r["log2_mic"] for r in tgt_te])

            # train source head
            src_head = train_head(embs_src_tr, y_src_tr, embs_src_val, y_src_val,
                                  d, device)

            # k=0: zero-shot
            if 0 in needed:
                m, p = eval_head(src_head, embs_tgt_te, y_tgt_te)
                seed_met["0"]  = m
                seed_pred["0"] = p
                print(f"    k=0   Pearson={m['pearson']:.3f}")
                metrics_file.write_text(json.dumps(metrics_all, indent=2))
                preds_file.write_text(json.dumps(preds_all, indent=2))

            # k>0: warmstart fine-tune
            rng = random.Random(seed)
            tgt_tr_shuf = tgt_tr.copy(); rng.shuffle(tgt_tr_shuf)

            for k in sorted(kk for kk in needed if kk > 0):
                support = tgt_tr_shuf[:k]
                embs_sup = precompute_embs(embedder, support, device)
                y_sup    = torch.tensor([r["log2_mic"] for r in support],
                                        dtype=torch.float32, device=device)
                head_ft = finetune_warmstart(src_head, embs_sup, y_sup, device)
                m, p = eval_head(head_ft, embs_tgt_te, y_tgt_te)
                seed_met[str(k)]  = m
                seed_pred[str(k)] = p
                print(f"    k={k:<3}  Pearson={m['pearson']:.3f}  "
                      f"Spearman={m['spearman']:.3f}  RMSE={m['rmse']:.3f}")
                metrics_file.write_text(json.dumps(metrics_all, indent=2))
                preds_file.write_text(json.dumps(preds_all, indent=2))

    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone. → {out_dir}")


if __name__ == "__main__":
    main()
