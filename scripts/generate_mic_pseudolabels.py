"""
Run MIC Transformer predictor on all 868k AMP training sequences.
Saves predicted log2_MIC for all 20 bacteria as a numpy memmap.

Output:
  data/processed/mic_pseudolabels.npy   shape (N, 20) float32
  data/processed/mic_pseudolabels_seqs.txt  one sequence per line (same order)

Usage:
  uv run python -u scripts/generate_mic_pseudolabels.py [--gpu 1]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIC_CKPT = PROJECT_ROOT / "checkpoints/mic_868k_transformer/best_model.pt"
MIC_CFG  = PROJECT_ROOT / "configs/mic_868k_transformer.yaml"
FASTA    = PROJECT_ROOT / "data/processed/amp_corpus.fasta"
OUT_NPY  = PROJECT_ROOT / "data/processed/mic_pseudolabels.npy"
OUT_SEQS = PROJECT_ROOT / "data/processed/mic_pseudolabels_seqs.txt"


def load_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur: seqs.append("".join(cur).upper())
                cur = []
            else:
                cur.append(line)
    if cur: seqs.append("".join(cur).upper())
    return seqs


class SeqDataset(Dataset):
    def __init__(self, seqs, max_len=48):
        from src.data.tokenizer import encode
        self.ids = [
            torch.tensor(encode(s[:max_len-2], add_special_tokens=True), dtype=torch.long)
            for s in seqs
        ]

    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.ids[i]


def collate(batch):
    from src.data.tokenizer import PAD_ID
    max_l = max(x.shape[0] for x in batch)
    out = torch.full((len(batch), max_l), PAD_ID, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, :len(x)] = x
    return out


def load_mic_model(device):
    from src.models.jepa import JEPA
    from src.models.supervised_head import JEPAMICPredictor
    from src.data.supervised_dataset import N_BACTERIA

    with open(MIC_CFG) as f:
        cfg = yaml.safe_load(f)
    pt_ckpt = torch.load(PROJECT_ROOT / cfg["pretrain_checkpoint"],
                          map_location=device, weights_only=False)
    jepa = JEPA(**pt_ckpt["cfg"]["model"])
    jepa.load_state_dict(pt_ckpt["model_state"])

    head_cfg = cfg["head"].copy()
    head_type = head_cfg.pop("head_type", "transformer")
    model = JEPAMICPredictor(
        encoder=jepa.context_encoder,
        d_model=pt_ckpt["cfg"]["model"]["d_model"],
        n_bacteria=N_BACTERIA,
        head_type=head_type,
        freeze_encoder=True,
        **head_cfg,
    ).to(device)
    ckpt = torch.load(MIC_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, seqs, device, batch_size=512):
    from src.data.supervised_dataset import N_BACTERIA
    ds = SeqDataset(seqs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=4, pin_memory=True)

    all_preds = []
    for step, ids in enumerate(loader):
        ids = ids.to(device)
        # predict for all bacteria by looping over bacteria indices
        B = ids.shape[0]
        preds = torch.zeros(B, N_BACTERIA)
        for b in range(N_BACTERIA):
            bidx = torch.full((B,), b, dtype=torch.long, device=device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                p = model(ids, bidx)
            preds[:, b] = p.cpu().float()
        all_preds.append(preds)
        if (step + 1) % 50 == 0:
            print(f"  {(step+1)*batch_size:>8,} / {len(seqs):,} sequences done")
    return torch.cat(all_preds, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading sequences from {FASTA} ...")
    seqs = load_fasta(FASTA)
    # filter to max_len=50 (same as training)
    seqs = [s for s in seqs if 3 <= len(s) <= 50]
    print(f"  {len(seqs):,} sequences (len 3-50)")

    print("Loading MIC predictor ...")
    model = load_mic_model(device)

    print("Predicting MIC pseudo-labels ...")
    preds = predict_all(model, seqs, device)  # (N, 20)
    print(f"Predictions shape: {preds.shape}  range [{preds.min():.2f}, {preds.max():.2f}]")

    np.save(OUT_NPY, preds)
    with open(OUT_SEQS, "w") as f:
        f.write("\n".join(seqs))
    print(f"Saved:\n  {OUT_NPY}\n  {OUT_SEQS}")


if __name__ == "__main__":
    main()
