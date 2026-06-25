"""
Retrain JEPA SpecFiLM on esAMPMIC's exact train/val split, evaluate on their test split.

This gives a fair head-to-head comparison:
  - Same training data  (esAMPMIC's EC/SA/PA train+val CSVs)
  - Same test data      (esAMPMIC's EC/SA/PA test CSVs)
  - Their architecture  vs  ours (JEPA SpecFiLM)

esAMPMIC published:  E.coli 0.781 | S.aureus 0.756 | P.aeruginosa 0.802
Our target:          retrain JEPA on same data, evaluate on same test

Outputs: eval_results/esampmic_retrain/metrics.json
         eval_results/esampmic_retrain/SUMMARY.md

Usage:
    uv run python scripts/train_on_esampmic.py --gpu 0
"""

from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import encode, PAD_ID
from src.models.pretrain_utils import load_pretrained_encoder
from src.models.supervised_head import JEPAMICPredictor

PRETRAIN_CKPT = PROJECT_ROOT / "checkpoints" / "jepa_pretrain_868k" / "last_jepa.pt"
OUT_DIR       = PROJECT_ROOT / "eval_results" / "esampmic_retrain"
CKPT_DIR      = PROJECT_ROOT / "checkpoints" / "esampmic_retrain_jepa"
ESAMPMIC_BASE = "https://raw.githubusercontent.com/chungcr/esAMPMIC/main/data"
VALID_AA      = set("ACDEFGHIKLMNPQRSTVWY")
MAX_LEN       = 42   # their sequences up to 40 + 2 special tokens

# 3 species: assign fresh indices 0/1/2 for this run
SPECIES = [
    ("EC", "E. coli",       0),
    ("SA", "S. aureus",     1),
    ("PA", "P. aeruginosa", 2),
]
ESAMPMIC_PUBLISHED = {"E. coli": 0.781, "S. aureus": 0.756, "P. aeruginosa": 0.802}

# hyperparams matching formal GRAMPA run
BATCH      = 256
EPOCHS     = 60
LR         = 3e-4
LR_ENC     = 5e-5   # smaller LR for encoder when unfrozen
WD         = 0.1
PATIENCE   = 12
LABEL_NOISE = 0.3
N_BACTERIA  = 3
D_MODEL     = 384


# ── data ─────────────────────────────────────────────────────────────────────

def download_csv(prefix: str, split: str) -> list[dict]:
    url = f"{ESAMPMIC_BASE}/{prefix}_X_{split}_40.csv"
    print(f"    {url} ...", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    print(f"{len(rows)} rows")
    return rows


def parse_rows(rows: list[dict], bact_idx: int) -> list[dict]:
    out = []
    for r in rows:
        seq = r.get("SEQUENCE", "").strip().upper()
        try:
            val = float(r["NEW-CONCENTRATION"])
        except (KeyError, ValueError):
            continue
        if not seq or len(seq) > MAX_LEN - 2 or not all(c in VALID_AA for c in seq):
            continue
        out.append({"seq": seq, "log2_mic": val, "bact_idx": bact_idx})
    return out


class MICSeqDataset(Dataset):
    def __init__(self, records: list[dict], noise_std: float = 0.0):
        self.records   = records
        self.noise_std = noise_std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r   = self.records[i]
        ids = torch.tensor(encode(r["seq"]), dtype=torch.long)
        val = torch.tensor(r["log2_mic"], dtype=torch.float32)
        if self.noise_std > 0:
            val = val + torch.randn(()) * self.noise_std
        return {"input_ids": ids,
                "bacteria_idx": torch.tensor(r["bact_idx"], dtype=torch.long),
                "log2_mic": val}


def collate(batch):
    max_l = max(b["input_ids"].shape[0] for b in batch)
    ids = torch.full((len(batch), max_l), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :b["input_ids"].shape[0]] = b["input_ids"]
    return {
        "input_ids":   ids,
        "bacteria_idx": torch.stack([b["bacteria_idx"] for b in batch]),
        "log2_mic":    torch.stack([b["log2_mic"] for b in batch]),
    }


# ── train ─────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, opt=None, use_fp16=False):
    is_train = opt is not None
    model.train() if is_train else model.eval()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            bidx  = batch["bacteria_idx"].to(device)
            y     = batch["log2_mic"].to(device)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                pred = model(ids, bidx)
                loss = F.huber_loss(pred, y, delta=1.0)
            if is_train:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            total_loss += loss.item() * len(y)
            n += len(y)
    return total_loss / max(n, 1)


def _pearson(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r)


def train(model, tr_loader, va_loader, device, use_fp16, unfreeze=False):
    if unfreeze:
        # differential LR: encoder gets LR_ENC, everything else gets LR
        enc_ids = set(id(p) for p in model.encoder.parameters())
        opt = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) in enc_ids],     "lr": LR_ENC},
            {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": LR},
        ], weight_decay=WD)
    else:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD
        )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(EPOCHS):
        tr_loss = run_epoch(model, tr_loader, device, opt, use_fp16)
        va_loss = run_epoch(model, va_loader, device)
        sched.step()
        if va_loss < best_val - 1e-4:
            best_val  = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop ep={ep+1}")
                break
        if (ep + 1) % 10 == 0:
            print(f"  ep={ep+1:3d}  tr={tr_loss:.4f}  va={va_loss:.4f}  best={best_val:.4f}")
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, loader, device, use_fp16=False):
    model.eval()
    preds, trues = [], []
    with torch.cuda.amp.autocast(enabled=use_fp16):
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            bidx = batch["bacteria_idx"].to(device)
            p    = model(ids, bidx).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(batch["log2_mic"].numpy().tolist())
    preds, trues = np.array(preds), np.array(trues)
    r,_   = pearsonr(trues, preds)
    rho,_ = spearmanr(trues, preds)
    rmse  = float(np.sqrt(np.mean((trues - preds)**2)))
    mae   = float(np.mean(np.abs(trues - preds)))
    return {"pearson": round(float(r),4), "spearman": round(float(rho),4),
            "rmse": round(rmse,4), "mae": round(mae,4), "n": int(len(trues))}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--unfreeze", action="store_true", help="Unfreeze JEPA encoder (lower LR)")
    ap.add_argument("--head", default="transformer", choices=["transformer","mlp"])
    ap.add_argument("--tag", default="", help="suffix for output dirs")
    args = ap.parse_args()

    device  = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not args.no_fp16
    tag = args.tag or f"{'unfreeze' if args.unfreeze else 'frozen'}_{args.head}"
    out_dir  = OUT_DIR.parent / f"esampmic_retrain_{tag}"
    ckpt_dir = CKPT_DIR.parent / f"esampmic_retrain_jepa_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  fp16={use_fp16}  unfreeze={args.unfreeze}  head={args.head}")
    print(f"Output: {out_dir}")

    # ── load data ──────────────────────────────────────────────────────────
    print("\nDownloading esAMPMIC splits...")
    tr_recs, va_recs, te_recs = [], [], []
    for prefix, sp_name, bact_idx in SPECIES:
        print(f"  {sp_name}:")
        tr_recs.extend(parse_rows(download_csv(prefix, "train"), bact_idx))
        va_recs.extend(parse_rows(download_csv(prefix, "val"),   bact_idx))
        te_recs.extend(parse_rows(download_csv(prefix, "test"),  bact_idx))

    print(f"\nTotal — train: {len(tr_recs)}  val: {len(va_recs)}  test: {len(te_recs)}")

    tr_ds = MICSeqDataset(tr_recs, noise_std=LABEL_NOISE)
    va_ds = MICSeqDataset(va_recs, noise_std=0.0)
    te_ds = MICSeqDataset(te_recs, noise_std=0.0)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  collate_fn=collate, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=0)

    # ── build model ────────────────────────────────────────────────────────
    print(f"\nBuilding JEPA SpecFiLM (n_bacteria=3, head={args.head}, unfreeze={args.unfreeze})...")
    encoder, pretrain_cfg = load_pretrained_encoder(str(PRETRAIN_CKPT), device)
    encoder = encoder.to(device)
    d_model = pretrain_cfg["model"]["d_model"]

    model = JEPAMICPredictor(
        encoder=encoder, d_model=d_model, n_bacteria=N_BACTERIA,
        bacteria_dim=64, head_type=args.head,
        hidden=256, dropout=0.4, adapter_bottleneck=64,
        freeze_encoder=not args.unfreeze,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # ── train ──────────────────────────────────────────────────────────────
    print(f"\nTraining (epochs={EPOCHS}, patience={PATIENCE}, batch={BATCH}, unfreeze={args.unfreeze})...")
    model = train(model, tr_loader, va_loader, device, use_fp16, unfreeze=args.unfreeze)

    # save checkpoint
    torch.save({"model_state": model.state_dict()}, ckpt_dir / "best_model.pt")
    print(f"  Checkpoint saved: {ckpt_dir}/best_model.pt")

    # ── evaluate: overall + per species ────────────────────────────────────
    print("\n=== Test Set Results ===")
    overall = evaluate(model, te_loader, device, use_fp16)
    print(f"  Overall  Pearson={overall['pearson']:.4f}  RMSE={overall['rmse']:.4f}  n={overall['n']}")

    per_species = {}
    for prefix, sp_name, bact_idx in SPECIES:
        sp_recs = [r for r in te_recs if r["bact_idx"] == bact_idx]
        sp_ds   = MICSeqDataset(sp_recs)
        sp_loader = DataLoader(sp_ds, batch_size=BATCH, shuffle=False,
                               collate_fn=collate, num_workers=0)
        m = evaluate(model, sp_loader, device, use_fp16)
        per_species[sp_name] = m
        pub = ESAMPMIC_PUBLISHED[sp_name]
        delta = m["pearson"] - pub
        print(f"  {sp_name:20s}  Pearson={m['pearson']:.4f}  RMSE={m['rmse']:.4f}  n={m['n']}"
              f"  vs esAMPMIC {pub:.3f}  Δ={delta:+.3f}")

    results = {"overall": overall, "per_species": per_species,
               "esampmic_published": ESAMPMIC_PUBLISHED,
               "training": {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                            "label_noise": LABEL_NOISE, "n_bacteria": N_BACTERIA,
                            "head": args.head, "unfreeze": args.unfreeze,
                            "data": "esAMPMIC GitHub EC/SA/PA"}}

    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2))

    # ── summary ────────────────────────────────────────────────────────────
    lines = [
        "# JEPA SpecFiLM retrained on esAMPMIC data — Head-to-Head\n",
        "Same train/val/test splits as esAMPMIC (GitHub: chungcr/esAMPMIC).",
        "JEPA encoder: jepa_pretrain_868k (frozen). Adapter + bacteria_emb + TransformerHead trained.\n",
        "",
        "## Per-Species Pearson Correlation",
        "",
        "| Species | esAMPMIC (published) | JEPA SpecFiLM (same data) | Δ | n_test |",
        "|---|---:|---:|---:|---:|",
    ]
    for sp_name, m in per_species.items():
        pub   = ESAMPMIC_PUBLISHED[sp_name]
        delta = m["pearson"] - pub
        lines.append(f"| {sp_name} | {pub:.3f} | **{m['pearson']:.3f}** | {delta:+.3f} | {m['n']} |")

    lines += ["", f"Overall (3 species concat): **{overall['pearson']:.4f}**"]

    summary = out_dir / "SUMMARY.md"
    summary.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {out_path}")
    print(f"Saved: {summary}")
    print("\n" + "\n".join(lines[5:]))


if __name__ == "__main__":
    main()
