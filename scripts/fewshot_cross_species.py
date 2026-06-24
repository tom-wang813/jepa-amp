"""
Few-shot cross-species MIC transfer.

Two protocols (select with --warmstart):

  cold-start (default):
    k=0  → source-trained head evaluated on target (zero-shot baseline)
    k>0  → FRESH head trained only on k target examples

  warm-start (--warmstart):
    k=0  → source-trained head evaluated on target (zero-shot baseline)
    k>0  → source-trained head CONTINUED on k target examples (fine-tuned)

Models:   JEPA-AMP, ESM-2 35M, MLM (same-arch baseline)
k values: 0, 5, 10, 20, 50, 100

Outputs:
    eval_results/fewshot_cross_species/          (cold-start)
    eval_results/fewshot_cross_species_warmstart/ (warm-start)

Usage:
    uv run python scripts/fewshot_cross_species.py --gpu 0
    uv run python scripts/fewshot_cross_species.py --gpu 0 --warmstart
    uv run python scripts/fewshot_cross_species.py --gpu 0 --models jepa esm2
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

OUT_DIR = PROJECT_ROOT / "eval_results" / "fewshot_cross_species"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIES_PAIRS = [
    ("E. coli",       "S. aureus"),
    ("E. coli",       "P. aeruginosa"),
    ("S. aureus",     "E. coli"),
    ("S. aureus",     "P. aeruginosa"),
    ("P. aeruginosa", "E. coli"),
]
SEEDS = [42, 123, 7]
K_VALUES = [0, 5, 10, 20, 50, 100]


# ── data ──────────────────────────────────────────────────────────────────────

def load_species(csv_path: Path, species: str, max_len: int = 50,
                 seed: int = 42) -> tuple[list, list, list]:
    recs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            seq = r["sequence"].strip().upper()
            if (
                r["is_modified"].strip() == "False"
                and r["bacterium"].strip() == species
                and 3 <= len(seq) <= max_len
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
    test_set = set(unique_seqs[:n_test])
    val_set  = set(unique_seqs[n_test:n_test + n_val])

    train, val, test = [], [], []
    for r in recs:
        if r["seq"] in test_set:
            test.append(r)
        elif r["seq"] in val_set:
            val.append(r)
        else:
            train.append(r)
    return train, val, test


# ── encoder wrappers ──────────────────────────────────────────────────────────

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
            out = self.esm(tokens, repr_layers=[self.num_layers], return_contacts=False)
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
        # load only encoder weights
        enc_state = {k[len("encoder."):]: v for k, v in ckpt["model_state"].items()
                     if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_state)
        self.enc = model.encoder
        self.d_model = ckpt["cfg"]["model"]["d_model"]
        for p in self.enc.parameters():
            p.requires_grad_(False)

    def forward(self, seqs: list[str], device) -> torch.Tensor:
        ids = _batch_encode_jepa(seqs, device)   # same tokenizer
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

def train_head(embedder, head, train_recs, val_recs, device,
               epochs: int = 60, batch_size: int = 128, lr: float = 3e-4,
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
    for _ in range(epochs):
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


def finetune_head(embedder, d_model: int, support_recs, device,
                  epochs: int = 200, lr: float = 1e-3,
                  source_head: nn.Module | None = None) -> nn.Module:
    """Fine-tune a head on k target-species support examples.

    source_head=None  → cold-start: fresh random init
    source_head=<head> → warm-start: copy source head weights, then adapt
    """
    if source_head is not None:
        import copy
        head_ft = copy.deepcopy(source_head).to(device)
        lr = 5e-5   # lower lr for warm-start to avoid forgetting
    else:
        head_ft = MICHead(d_model).to(device)
    opt = torch.optim.AdamW(head_ft.parameters(), lr=lr, weight_decay=0.01)
    seqs = [r["seq"] for r in support_recs]
    y = torch.tensor([r["log2_mic"] for r in support_recs],
                     dtype=torch.float32, device=device)
    with torch.no_grad():
        emb = embedder(seqs, device)

    head_ft.train()
    for _ in range(epochs):
        pred = head_ft(emb)
        loss = F.huber_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
    head_ft.eval()
    return head_ft


def eval_head(embedder, head, test_recs, device) -> dict:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error

    head.eval()
    preds, trues = [], []
    for i in range(0, len(test_recs), 256):
        batch = test_recs[i:i + 256]
        seqs = [r["seq"] for r in batch]
        y = [r["log2_mic"] for r in batch]
        with torch.no_grad():
            emb = embedder(seqs, device)
            pred = head(emb).cpu().numpy()
        preds.extend(pred.tolist())
        trues.extend(y)

    preds, trues = np.array(preds), np.array(trues)
    r, _   = pearsonr(preds, trues)
    rho, _ = spearmanr(preds, trues)
    rmse   = float(np.sqrt(mean_squared_error(trues, preds)))
    return {"pearson": float(r), "spearman": float(rho), "rmse": rmse, "n": len(trues)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", nargs="+",
                        default=["jepa", "esm2", "esm2_650m", "mlm"],
                        choices=["jepa", "esm2", "esm2_650m", "mlm"])
    parser.add_argument("--k_values", nargs="+", type=int, default=K_VALUES)
    parser.add_argument("--warmstart", action="store_true",
                        help="Warm-start: fine-tune source head instead of fresh init")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grampa = PROJECT_ROOT / "data" / "grampa.csv"

    out_dir  = PROJECT_ROOT / "eval_results" / (
        "fewshot_cross_species_warmstart" if args.warmstart else "fewshot_cross_species"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"

    results: dict = {}
    if out_file.exists():
        results = json.loads(out_file.read_text())

    model_factories = {
        "jepa":     JEPAEmbedder,
        "esm2":     ESM2Embedder,
        "esm2_650m": lambda dev: ESM2Embedder(dev, model_key="esm2_t33_650M"),
        "mlm":      MLMEmbedder,
    }

    for model_name in args.models:
        print(f"\n{'='*60}\n{model_name.upper()}\n{'='*60}")
        embedder = model_factories[model_name](device).to(device).eval()
        d = embedder.d_model
        model_res = results.setdefault(model_name, {})

        for src_species, tgt_species in SPECIES_PAIRS:
            pair_key = f"{src_species}→{tgt_species}"
            pair_res = model_res.setdefault(pair_key, {})

            for seed in SEEDS:
                seed_key = str(seed)
                seed_res = pair_res.setdefault(seed_key, {})

                # check if all k already done
                needed_ks = [k for k in args.k_values
                             if str(k) not in seed_res]
                if not needed_ks:
                    print(f"  [skip] {pair_key} seed={seed}")
                    continue

                print(f"\n  {pair_key}  seed={seed}")

                src_tr, src_val, src_te = load_species(grampa, src_species, seed=seed)
                tgt_tr, _,      tgt_te  = load_species(grampa, tgt_species, seed=seed)

                print(f"    src train={len(src_tr)}  tgt train={len(tgt_tr)}  tgt test={len(tgt_te)}")

                # train base head on source species
                head = MICHead(d).to(device)
                train_head(embedder, head, src_tr, src_val, device)

                # zero-shot baseline (k=0)
                if 0 in needed_ks:
                    m = eval_head(embedder, head, tgt_te, device)
                    seed_res["0"] = m
                    print(f"    k=0  (zero-shot)  Pearson={m['pearson']:.3f}")
                    out_file.write_text(json.dumps(results, indent=2))

                # few-shot: sample k from target train, fine-tune, evaluate
                rng = random.Random(seed)
                tgt_tr_shuffled = tgt_tr.copy()
                rng.shuffle(tgt_tr_shuffled)

                for k in sorted(k for k in needed_ks if k > 0):
                    support = tgt_tr_shuffled[:k]
                    head_ft = finetune_head(
                        embedder, d, support, device,
                        source_head=head if args.warmstart else None,
                    )
                    m = eval_head(embedder, head_ft, tgt_te, device)
                    seed_res[str(k)] = m
                    print(f"    k={k:<3}  Pearson={m['pearson']:.3f}  "
                          f"Spearman={m['spearman']:.3f}  RMSE={m['rmse']:.3f}")
                    out_file.write_text(json.dumps(results, indent=2))

        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_summary(results, args.k_values, out_dir)
    print(f"\nDone. Results → {out_dir}")


def _agg(seed_results: dict, k: int) -> tuple[float, float]:
    # seed_results is {seed_str: {k_str: {pearson, spearman, ...}}}
    vals = []
    for seed_data in seed_results.values():
        entry = seed_data.get(str(k))
        if entry and "pearson" in entry:
            vals.append(entry["pearson"])
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _write_summary(results: dict, k_values: list[int], out_dir: Path) -> None:
    lines = [
        "# Few-Shot Cross-Species MIC Transfer",
        "",
        "Head trained on source species, then fine-tuned on k target-species examples.",
        "k=0 is the zero-shot baseline. 3-seed mean ± std (Pearson).",
        "",
    ]

    for src, tgt in SPECIES_PAIRS:
        pair_key = f"{src}→{tgt}"
        lines.append(f"## {pair_key}")
        header = "| k | " + " | ".join(results.keys()) + " |"
        sep    = "|---|" + "---|" * len(results)
        lines += [header, sep]
        for k in k_values:
            row = [str(k)]
            for model_name, model_res in results.items():
                seed_res = model_res.get(pair_key, {})
                m, s = _agg(seed_res, k)
                row.append(f"{m:.3f} ± {s:.3f}" if not np.isnan(m) else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
