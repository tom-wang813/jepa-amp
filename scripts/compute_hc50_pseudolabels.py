"""
Pre-compute HC50 pseudo-labels for the AMP training corpus using the frozen
QMAP HC50 predictor (MeanPoolRegressor trained on split 0).

Output: data/processed/hc50_pseudolabels.json  {sequence: log10_HC50}

Usage:
    uv run python scripts/compute_hc50_pseudolabels.py --gpu 0
    uv run python scripts/compute_hc50_pseudolabels.py --gpu 0 --batch_size 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pretrain_utils import load_pretrained_encoder
from src.data.tokenizer import encode, BOS_ID, EOS_ID, PAD_ID

HC50_CKPT = PROJECT_ROOT / "eval_results/qmap_jepa_hc50_head_finetune_seed42/split_0/best_model.pt"
PRETRAIN_CKPT = PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt"
FASTA_PATH = PROJECT_ROOT / "data/processed/amp_corpus.fasta"
OUT_PATH   = PROJECT_ROOT / "data/processed/hc50_pseudolabels.json"


class _HC50Oracle(nn.Module):
    def __init__(self, encoder, d_model: int, hidden: int = 512, dropout: float = 0.25):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids)     # (B, L, D)
        pooled = torch.stack([
            h[i, 1:lengths[i] - 1].mean(0) for i in range(len(lengths))
        ])
        return self.head(pooled).squeeze(-1)


class SeqDataset(Dataset):
    def __init__(self, seqs: list[str], max_seq_len: int = 52):
        self.seqs = seqs
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = [BOS_ID] + encode(seq, add_special_tokens=False) + [EOS_ID]
        ids = ids[:self.max_seq_len]
        length = len(ids)
        padded = ids + [PAD_ID] * (self.max_seq_len - length)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--fasta", default=str(FASTA_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load HC50 model
    ckpt = torch.load(HC50_CKPT, map_location=device, weights_only=False)
    enc, pt_cfg = load_pretrained_encoder(str(PRETRAIN_CKPT), device)
    d_model = pt_cfg["model"]["d_model"]
    model = _HC50Oracle(enc, d_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"HC50 oracle loaded from {HC50_CKPT}")

    # Load sequences
    AA = set("ACDEFGHIKLMNPQRSTVWY")
    seqs = []
    for rec in SeqIO.parse(args.fasta, "fasta"):
        s = str(rec.seq).upper()
        if 3 <= len(s) <= args.max_len and all(c in AA for c in s):
            seqs.append(s)
    seqs = sorted(set(seqs))
    print(f"Sequences to label: {len(seqs):,}")

    ds = SeqDataset(seqs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results: dict[str, float] = {}
    done = 0
    with torch.no_grad():
        for ids, lengths in loader:
            ids, lengths = ids.to(device), lengths.to(device)
            preds = model(ids, lengths).cpu().tolist()
            batch_seqs = seqs[done:done + len(preds)]
            for seq, val in zip(batch_seqs, preds):
                results[seq] = float(val)
            done += len(preds)
            if done % 50000 == 0:
                print(f"  {done:,} / {len(seqs):,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f)

    vals = list(results.values())
    import statistics
    print(f"\nDone. {len(results):,} sequences labelled → {out}")
    print(f"HC50 log10 stats: mean={statistics.mean(vals):.3f} "
          f"std={statistics.stdev(vals):.3f} "
          f"min={min(vals):.2f} max={max(vals):.2f}")


if __name__ == "__main__":
    main()
