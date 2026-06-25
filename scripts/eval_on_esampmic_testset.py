"""
Evaluate JEPA SpecFiLM on esAMPMIC's published test sets.

Downloads esAMPMIC test CSVs from GitHub, runs our formal JEPA SpecFiLM
checkpoint (formal_mic_868k_transformer) on each species' test sequences,
and reports Pearson / Spearman / RMSE — directly comparable to their
published numbers (E.coli 0.781, S.aureus 0.756, P.aeruginosa 0.802).

Also flags sequence overlap with our GRAMPA training set so the comparison
can be reported both ways (all test / overlap-removed).

Outputs: eval_results/esampmic_comparison/metrics.json
         eval_results/esampmic_comparison/SUMMARY.md

Usage:
    uv run python scripts/eval_on_esampmic_testset.py --gpu 0
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
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.supervised_dataset import BACTERIA_TO_IDX, load_grampa
from src.data.tokenizer import encode, PAD_ID
from src.models.pretrain_utils import load_pretrained_encoder
from src.models.supervised_head import JEPAMICPredictor
from src.models.generator import Adapter

FORMAL_CKPT = PROJECT_ROOT / "checkpoints" / "formal_mic_868k_transformer" / "best_model.pt"
GRAMPA_CSV  = PROJECT_ROOT / "data" / "grampa.csv"
OUT_DIR     = PROJECT_ROOT / "eval_results" / "esampmic_comparison"
MAX_LEN     = 50
D_MODEL     = 384
VALID_AA    = set("ACDEFGHIKLMNPQRSTVWY")

ESAMPMIC_BASE = "https://raw.githubusercontent.com/chungcr/esAMPMIC/main/data"
SPECIES_MAP = {
    "E. coli":        ("EC",  BACTERIA_TO_IDX["E. coli"]),
    "S. aureus":      ("SA",  BACTERIA_TO_IDX["S. aureus"]),
    "P. aeruginosa":  ("PA",  BACTERIA_TO_IDX["P. aeruginosa"]),
}
# esAMPMIC published Pearson on their own test set
ESAMPMIC_PUBLISHED = {
    "E. coli":       0.781,
    "S. aureus":     0.756,
    "P. aeruginosa": 0.802,
}


def download_csv(url: str) -> list[dict]:
    print(f"  Downloading {url} ...", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    print(f"{len(rows)} rows")
    return rows


def parse_esampmic(rows: list[dict], max_len: int = MAX_LEN) -> list[tuple[str, float]]:
    """Return (sequence, log2_mic) pairs; filter non-canonical and too-long."""
    out = []
    for r in rows:
        seq = r.get("SEQUENCE", "").strip().upper()
        try:
            val = float(r["NEW-CONCENTRATION"])
        except (KeyError, ValueError):
            continue
        if not seq or len(seq) > max_len or not all(c in VALID_AA for c in seq):
            continue
        out.append((seq, val))
    return out


def load_grampa_train_seqs() -> set[str]:
    """Return set of all sequences in GRAMPA training+val split to flag overlap."""
    train_ds, val_ds, _ = load_grampa(GRAMPA_CSV, max_len=MAX_LEN,
                                       val_ratio=0.1, test_ratio=0.1, seed=42)
    seqs = set()
    for rec in list(train_ds) + list(val_ds):
        ids = rec["input_ids"].numpy().tolist()
        # decode back: ids are token indices; we just store the set as token tuples
        seqs.add(tuple(ids))
    return seqs


def load_grampa_train_raw_seqs() -> set[str]:
    """Load raw sequence strings from GRAMPA (train + val)."""
    import csv as _csv
    seqs = set()
    with open(GRAMPA_CSV) as f:
        for r in _csv.DictReader(f):
            seq = r.get("sequence", "").strip().upper()
            if seq:
                seqs.add(seq)
    return seqs


def build_model(device: torch.device):
    encoder, pretrain_cfg = load_pretrained_encoder(str(
        PROJECT_ROOT / "checkpoints" / "jepa_pretrain_868k" / "last_jepa.pt"
    ), device)
    encoder = encoder.to(device)
    d_model = pretrain_cfg["model"]["d_model"]
    n_bact  = 20

    model = JEPAMICPredictor(
        encoder=encoder, d_model=d_model, n_bacteria=n_bact,
        bacteria_dim=64, head_type="transformer",
        hidden=256, dropout=0.4, adapter_bottleneck=64,
        freeze_encoder=True,
    ).to(device)
    state = torch.load(FORMAL_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, seqs: list[str], bact_idx: int, device: torch.device,
            batch_size: int = 256) -> np.ndarray:
    all_preds = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        max_l = max(len(encode(s)) for s in batch)
        ids = torch.full((len(batch), max_l), PAD_ID, dtype=torch.long, device=device)
        for j, seq in enumerate(batch):
            enc = encode(seq)
            ids[j, :len(enc)] = torch.tensor(enc, dtype=torch.long, device=device)
        bidx = torch.full((len(batch),), bact_idx, dtype=torch.long, device=device)
        preds = model(ids, bidx).cpu().numpy()
        all_preds.append(preds)
    return np.concatenate(all_preds)


def metrics(trues, preds):
    trues, preds = np.array(trues), np.array(preds)
    r,_   = pearsonr(trues, preds)
    rho,_ = spearmanr(trues, preds)
    rmse  = float(np.sqrt(np.mean((trues - preds)**2)))
    return {"pearson": round(float(r),4), "spearman": round(float(rho),4),
            "rmse": round(rmse,4), "n": int(len(trues))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load GRAMPA sequences for overlap check
    print("Loading GRAMPA sequences for overlap detection...")
    grampa_seqs = load_grampa_train_raw_seqs()
    print(f"  GRAMPA total unique seqs: {len(grampa_seqs)}")

    # build model
    print("Loading JEPA SpecFiLM checkpoint...")
    model = build_model(device)
    print("  Model loaded.")

    results = {}

    for sp_name, (prefix, bact_idx) in SPECIES_MAP.items():
        print(f"\n=== {sp_name} (bact_idx={bact_idx}) ===")
        url = f"{ESAMPMIC_BASE}/{prefix}_X_test_40.csv"
        try:
            rows = download_csv(url)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        pairs = parse_esampmic(rows)
        print(f"  Valid sequences after filter: {len(pairs)}")

        seqs  = [p[0] for p in pairs]
        trues = [p[1] for p in pairs]

        # overlap detection
        in_grampa = [s in grampa_seqs for s in seqs]
        n_overlap = sum(in_grampa)
        print(f"  Sequences in GRAMPA train+val: {n_overlap}/{len(seqs)} ({100*n_overlap/len(seqs):.1f}%)")

        # predict on ALL test seqs
        preds = predict(model, seqs, bact_idx, device)

        m_all = metrics(trues, preds)
        print(f"  All test   Pearson={m_all['pearson']:.4f}  RMSE={m_all['rmse']:.4f}  n={m_all['n']}")

        # predict on overlap-removed subset
        novel_idx = [i for i,v in enumerate(in_grampa) if not v]
        if len(novel_idx) >= 10:
            t_nov = [trues[i] for i in novel_idx]
            p_nov = preds[novel_idx]
            m_nov = metrics(t_nov, p_nov)
            print(f"  Novel only Pearson={m_nov['pearson']:.4f}  RMSE={m_nov['rmse']:.4f}  n={m_nov['n']}")
        else:
            m_nov = None
            print("  Novel-only: too few sequences after overlap removal")

        results[sp_name] = {
            "esampmic_published": ESAMPMIC_PUBLISHED[sp_name],
            "our_all_test": m_all,
            "our_novel_only": m_nov,
            "n_overlap": n_overlap,
            "n_total": len(seqs),
        }

    # save
    out_path = OUT_DIR / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    # summary table
    lines = [
        "# JEPA SpecFiLM vs esAMPMIC — same test set\n",
        "esAMPMIC test CSVs from https://github.com/chungcr/esAMPMIC\n",
        "Our model: formal_mic_868k_transformer (trained on GRAMPA, curated DBAASP subset)\n",
        "",
        "## Pearson Correlation on esAMPMIC Test Set",
        "",
        "| Species | esAMPMIC (published) | JEPA SpecFiLM (all test) | JEPA (novel only) | Overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for sp, r in results.items():
        nov = r["our_novel_only"]["pearson"] if r["our_novel_only"] else "—"
        pct = f"{100*r['n_overlap']/r['n_total']:.0f}%"
        lines.append(
            f"| {sp} | {r['esampmic_published']:.3f} | "
            f"{r['our_all_test']['pearson']:.3f} | {nov if isinstance(nov,str) else f'{nov:.3f}'} | {pct} |"
        )

    summary = OUT_DIR / "SUMMARY.md"
    summary.write_text("\n".join(lines) + "\n")
    print(f"Saved: {summary}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
