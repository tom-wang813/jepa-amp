"""
Few-shot cross-species MIC transfer via bacteria-embedding adaptation.

Protocol:
  1. Train JEPAMICPredictor on source species only (encoder frozen,
     adapter + bacteria_emb[source] + head all trained).
  2. For k-shot: add a fresh bacteria_emb[target] (64-dim, random init).
     Fine-tune ONLY bacteria_emb[target] on k labeled target examples
     (everything else frozen). k = 0 (zero-shot), 5, 10, 20, 50, 100.
  3. Evaluate on target species test split.

Compares against: plain MICHead (no bacteria conditioning) baseline.

Outputs:
    eval_results/fewshot_bact_emb/
        metrics.json
        SUMMARY.md

Usage:
    uv run python scripts/fewshot_bact_emb.py --gpu 0
    uv run python scripts/fewshot_bact_emb.py --gpu 0 --model esm2_650m
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
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "eval_results" / "fewshot_bact_emb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIES_PAIRS = [
    ("E. coli",       "S. aureus"),
    ("E. coli",       "P. aeruginosa"),
    ("S. aureus",     "E. coli"),
    ("S. aureus",     "P. aeruginosa"),
    ("P. aeruginosa", "E. coli"),
]
SEEDS    = [42, 123, 7]
K_VALUES = [0, 5, 10, 20, 50, 100]

BACTERIA_DIM = 64
HIDDEN       = 256
DROPOUT      = 0.3


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
    n       = len(unique_seqs)
    n_test  = max(1, int(n * 0.15))
    n_val   = max(1, int(n * 0.10))
    test_s  = set(unique_seqs[:n_test])
    val_s   = set(unique_seqs[n_test:n_test + n_val])

    train, val, test = [], [], []
    for r in recs:
        if r["seq"] in test_s:   test.append(r)
        elif r["seq"] in val_s:  val.append(r)
        else:                    train.append(r)
    return train, val, test


# ── encoder ───────────────────────────────────────────────────────────────────

def load_jepa_encoder(device):
    from src.models.jepa import JEPA
    ckpt = torch.load(
        PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
        map_location=device, weights_only=False,
    )
    jepa = JEPA(**ckpt["cfg"]["model"])
    jepa.load_state_dict(ckpt["model_state"])
    enc = jepa.context_encoder
    d   = ckpt["cfg"]["model"]["d_model"]
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.to(device).eval(), d


class _ESM2Enc(nn.Module):
    """Thin wrapper so ESM-2 behaves like JEPA encoder for encode_seqs."""
    def __init__(self, esm, alphabet, d, num_layers):
        super().__init__()
        self.esm = esm; self.alphabet = alphabet
        self.d_model = d; self.num_layers = num_layers
        self.bc = alphabet.get_batch_converter()

    def forward(self, seqs_or_ids):
        # seqs_or_ids is a list of strings here
        raise NotImplementedError("use encode_seqs_esm2 instead")


def load_esm2_encoder(device, model_key: str = "esm2_t33_650M"):
    from src.models.esm_head import load_esm2
    esm, alphabet, d = load_esm2(model_key)
    for p in esm.parameters():
        p.requires_grad_(False)
    esm = esm.to(device).eval()
    return esm, alphabet, d


def encode_seqs(enc, seqs, device, alphabet=None):
    """Works for JEPA encoder (alphabet=None) or ESM-2 (pass alphabet)."""
    if alphabet is not None:
        bc = alphabet.get_batch_converter()
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = bc(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = enc(tokens, repr_layers=[enc.num_layers], return_contacts=False)
        h   = out["representations"][enc.num_layers]
        pad = tokens == alphabet.padding_idx
        h   = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
        return h.sum(1) / lengths
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:50]) for s in seqs]
    L   = max(len(e) for e in encoded)
    ids = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    with torch.no_grad():
        h   = enc(ids)
    pad = ids == 0
    h   = h.masked_fill(pad.unsqueeze(-1), 0.0)
    lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
    return h.sum(1) / lengths   # (B, d_model)


# ── conditioned MIC model ─────────────────────────────────────────────────────

class BactCondHead(nn.Module):
    """Frozen encoder + FiLM bacteria conditioning + MLP head.

    Adapter: linear bottleneck (same idea as JEPAMICPredictor).
    bacteria_emb: Embedding(n_bact, BACTERIA_DIM); only the target-species
    row is trained during few-shot adaptation.
    """

    def __init__(self, d_model: int, n_bact: int = 1):
        super().__init__()
        # FiLM: zero-init → identity at start of training
        self.bact_emb = nn.Embedding(n_bact, BACTERIA_DIM)
        self.film     = nn.Linear(BACTERIA_DIM, 2 * d_model)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.head = nn.Sequential(
            nn.Linear(d_model, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, emb: torch.Tensor, bact_id: int) -> torch.Tensor:
        """emb: (B, d_model)  →  returns (B,)"""
        bact = self.bact_emb(torch.tensor([bact_id], device=emb.device))  # (1, BD)
        gb   = self.film(bact)                                             # (1, 2D)
        gamma, beta = gb.chunk(2, dim=-1)                                  # (1, D)
        x = emb * (1 + gamma) + beta                                       # (B, D)
        return self.head(x).squeeze(-1)

    def add_target_species(self):
        """Append one new random embedding row for the target species.
        Returns the index of the new row."""
        old_w = self.bact_emb.weight.data
        new_w = torch.randn(1, BACTERIA_DIM, device=old_w.device) * 0.01
        self.bact_emb = nn.Embedding(old_w.shape[0] + 1, BACTERIA_DIM)
        with torch.no_grad():
            self.bact_emb.weight[:old_w.shape[0]] = old_w
            self.bact_emb.weight[-1]              = new_w
        return old_w.shape[0]   # new index


# ── train / eval ──────────────────────────────────────────────────────────────

def precompute_embs(enc, recs, device, batch=256, alphabet=None):
    """Cache encoder outputs to avoid repeated forward passes."""
    embs = []
    for i in range(0, len(recs), batch):
        seqs = [r["seq"] for r in recs[i:i+batch]]
        embs.append(encode_seqs(enc, seqs, device, alphabet=alphabet))
    return torch.cat(embs, 0)


def train_source_head(model: BactCondHead, enc, src_tr, src_val, device,
                      bact_id: int = 0, epochs: int = 80,
                      lr: float = 3e-4, patience: int = 15, alphabet=None):
    opt = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "bact_emb" not in n or True],
        lr=lr, weight_decay=0.05,
    )
    # only train non-encoder params
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.05)

    src_embs = precompute_embs(enc, src_tr,  device, alphabet=alphabet)
    val_embs = precompute_embs(enc, src_val, device, alphabet=alphabet)
    src_y    = torch.tensor([r["log2_mic"] for r in src_tr],  dtype=torch.float32, device=device)
    val_y    = torch.tensor([r["log2_mic"] for r in src_val], dtype=torch.float32, device=device)

    best_val, wait, best_state = float("inf"), 0, None
    idx = list(range(len(src_tr)))
    for _ in range(epochs):
        model.train()
        random.shuffle(idx)
        for i in range(0, len(idx), 128):
            b = idx[i:i+128]
            loss = F.huber_loss(model(src_embs[b], bact_id), src_y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.huber_loss(model(val_embs, bact_id), val_y).item()
        if vl < best_val - 1e-4:
            best_val = vl; wait = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_state:
        model.load_state_dict(best_state)


def finetune_bact_emb(model: BactCondHead, enc, support_recs, device,
                      tgt_bact_id: int, epochs: int = 200, lr: float = 5e-3,
                      alphabet=None):
    """Fine-tune ONLY bact_emb[tgt_bact_id] on k support examples."""
    # freeze everything except target bacteria embedding row
    for p in model.parameters():
        p.requires_grad_(False)
    model.bact_emb.weight.requires_grad_(True)

    # fine-tune only the target row via a masked gradient trick:
    # we'll manually zero other rows' grads after backward
    opt = torch.optim.Adam([model.bact_emb.weight], lr=lr)

    if not support_recs:
        return
    embs = precompute_embs(enc, support_recs, device, alphabet=alphabet)
    y    = torch.tensor([r["log2_mic"] for r in support_recs],
                        dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        pred = model(embs, tgt_bact_id)
        loss = F.huber_loss(pred, y)
        opt.zero_grad(); loss.backward()
        # zero out gradient for all rows except tgt_bact_id
        with torch.no_grad():
            mask = torch.zeros_like(model.bact_emb.weight.grad)
            mask[tgt_bact_id] = 1.0
            model.bact_emb.weight.grad *= mask
        opt.step()
    model.eval()


@torch.no_grad()
def eval_model(model: BactCondHead, enc, test_recs, device, bact_id: int,
               alphabet=None) -> dict:
    from scipy.stats import pearsonr, spearmanr
    embs = precompute_embs(enc, test_recs, device, alphabet=alphabet)
    y    = np.array([r["log2_mic"] for r in test_recs])
    model.eval()
    preds = model(embs, bact_id).cpu().numpy()
    r,   _ = pearsonr(preds, y)
    rho, _ = spearmanr(preds, y)
    rmse   = float(np.sqrt(np.mean((preds - y)**2)))
    return {"pearson": float(r), "spearman": float(rho),
            "rmse": rmse, "n": len(y)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default="jepa",
                        choices=["jepa", "esm2_650m"],
                        help="Encoder backbone to use")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grampa = PROJECT_ROOT / "data" / "grampa.csv"

    out_dir = PROJECT_ROOT / "eval_results" / f"fewshot_bact_emb_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"

    results: dict = {}
    if out_file.exists():
        results = json.loads(out_file.read_text())

    alphabet = None
    if args.model == "jepa":
        enc, d_model = load_jepa_encoder(device)
    else:
        enc, alphabet, d_model = load_esm2_encoder(device, "esm2_t33_650M")

    for src_species, tgt_species in SPECIES_PAIRS:
        pair_key = f"{src_species}→{tgt_species}"
        pair_res = results.setdefault(pair_key, {})

        for seed in SEEDS:
            seed_key = str(seed)
            seed_res = pair_res.setdefault(seed_key, {})

            needed = [k for k in K_VALUES if str(k) not in seed_res]
            if not needed:
                print(f"  [skip] {pair_key} seed={seed}")
                continue

            print(f"\n  {pair_key}  seed={seed}")
            src_tr, src_val, src_te = load_species(grampa, src_species, seed=seed)
            tgt_tr, _,      tgt_te  = load_species(grampa, tgt_species, seed=seed)
            print(f"    src train={len(src_tr)}  tgt train={len(tgt_tr)}  tgt test={len(tgt_te)}")

            # build and train source model (bact_id=0 = source)
            model = BactCondHead(d_model, n_bact=1).to(device)
            train_source_head(model, enc, src_tr, src_val, device,
                              bact_id=0, alphabet=alphabet)

            # add target-species bacteria embedding (bact_id=1)
            tgt_id = model.add_target_species()
            model = model.to(device)

            # k=0: zero-shot – target emb is random, just evaluate
            if 0 in needed:
                m = eval_model(model, enc, tgt_te, device, bact_id=tgt_id,
                               alphabet=alphabet)
                seed_res["0"] = m
                print(f"    k=0  (zero-shot)  Pearson={m['pearson']:.3f}")
                out_file.write_text(json.dumps(results, indent=2))

            # k>0: fine-tune only bact_emb[tgt_id] on k support examples
            rng = random.Random(seed)
            tgt_tr_shuf = tgt_tr.copy(); rng.shuffle(tgt_tr_shuf)

            for k in sorted(k for k in needed if k > 0):
                # reset target embedding to random before each k
                with torch.no_grad():
                    model.bact_emb.weight[tgt_id] = \
                        torch.randn(BACTERIA_DIM, device=device) * 0.01

                support = tgt_tr_shuf[:k]
                finetune_bact_emb(model, enc, support, device,
                                  tgt_bact_id=tgt_id, alphabet=alphabet)
                m = eval_model(model, enc, tgt_te, device, bact_id=tgt_id,
                               alphabet=alphabet)
                seed_res[str(k)] = m
                print(f"    k={k:<3}  Pearson={m['pearson']:.3f}  "
                      f"Spearman={m['spearman']:.3f}  RMSE={m['rmse']:.3f}")
                out_file.write_text(json.dumps(results, indent=2))

    _write_summary(results, out_dir)
    print(f"\nDone. Results → {out_dir}")


def _agg(seed_res: dict, k: int) -> tuple[float, float]:
    vals = [v["pearson"] for v in seed_res.values()
            if str(k) in seed_res.get(list(seed_res.keys())[0] if seed_res else "", {})]
    vals = []
    for sd in seed_res.values():
        e = sd.get(str(k))
        if e and "pearson" in e:
            vals.append(e["pearson"])
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _write_summary(results: dict, out_dir: Path) -> None:
    lines = [
        "# Few-Shot Cross-Species via Bacteria-Embedding Adaptation",
        "",
        "Only the 64-dim bacteria embedding for the target species is fine-tuned.",
        "k=0: random target embedding (zero-shot). 3-seed mean ± std (Pearson).",
        "",
    ]
    for src, tgt in SPECIES_PAIRS:
        pk = f"{src}→{tgt}"
        sr = results.get(pk, {})
        lines.append(f"## {pk}")
        lines.append("| k | Pearson (mean ± std) |")
        lines.append("|---|---|")
        for k in K_VALUES:
            m, s = _agg(sr, k)
            lines.append(f"| {k} | {m:.3f} ± {s:.3f} |" if not np.isnan(m) else f"| {k} | — |")
        lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
