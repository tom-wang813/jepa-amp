"""
Temperature sweep for the conditional generator.

Generates N sequences at each (temperature, top_p) combo and reports:
  - mean/std sequence length
  - length histogram
  - JEPA-probe AMP score (fraction > 0.5)
  - uniqueness

Usage:
  uv run python -u scripts/temp_sweep.py [--gpu 0] [--n 200]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
EVAL_DIR      = PROJECT_ROOT / "eval_results"
PRETRAIN_CFG  = PROJECT_ROOT / "configs/jepa_pretrain_868k.yaml"
FINETUNE_CFG  = PROJECT_ROOT / "configs/finetune_868k.yaml"
PRETRAIN_CKPT = PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt"
GEN_CKPT      = PROJECT_ROOT / "checkpoints/generator_868k/best_generator.pt"
EVAL_DIR      = PROJECT_ROOT / "eval_results"

SWEEP = [
    (0.5,  0.9),
    (0.7,  0.9),
    (0.9,  0.9),
    (1.0,  0.9),   # baseline
    (1.2,  0.9),
    (1.0,  0.7),   # tighter nucleus
    (1.0,  0.95),  # wider nucleus
]

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MAX_PROBE_SEQS = 5_000


def _ids_to_seq(ids: list[int], eos_id: int, pad_id: int) -> str:
    aa = "ACDEFGHIKLMNPQRSTVWY"
    out = []
    for i in ids:
        if i == eos_id or i == pad_id:
            break
        if 2 <= i <= 21:
            out.append(aa[i - 2])
    return "".join(out)


def generate(generator, dataset, device, n: int, temperature: float, top_p: float,
             batch_size: int = 64) -> list[str]:
    from src.models.generator import ConditionalGenerator
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    it = iter(loader)
    seqs = []
    with tqdm(total=n, desc=f"T={temperature} top_p={top_p}", leave=False) as pbar:
        while len(seqs) < n:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            ctx = batch["context_ids"].to(device)
            cond = batch["conditions"].to(device) if "conditions" in batch else None
            with torch.no_grad():
                out = generator.generate(ctx, conditions=cond, max_new_tokens=55,
                                         temperature=temperature, top_p=top_p)
            for row in out:
                seq = _ids_to_seq(row.tolist(), eos_id=1, pad_id=0)
                if len(seq) >= 3:
                    seqs.append(seq)
                    pbar.update(1)
                    if len(seqs) >= n:
                        break
    return seqs[:n]


def jepa_amp_score(seqs: list[str], encoder, device, train_pos: list[str],
                   neg_seqs: list[str]) -> np.ndarray:
    from src.eval.amp_classifier import JEPAAMPClassifier
    clf = JEPAAMPClassifier(encoder=encoder, device=str(device))
    pos_cap = random.sample(train_pos, min(len(train_pos), MAX_PROBE_SEQS))
    neg_cap = random.sample(neg_seqs,  min(len(neg_seqs),  MAX_PROBE_SEQS))
    clf.fit(pos_seqs=pos_cap, neg_seqs=neg_cap)
    return clf.predict_proba(seqs)


def summarise(seqs: list[str], scores: np.ndarray) -> dict:
    lengths = [len(s) for s in seqs]
    hist, edges = np.histogram(lengths, bins=range(1, 52, 5))
    return {
        "n": len(seqs),
        "mean_length": float(np.mean(lengths)),
        "std_length":  float(np.std(lengths)),
        "min_length":  int(np.min(lengths)),
        "max_length":  int(np.max(lengths)),
        "length_hist": {"counts": hist.tolist(),
                        "bin_edges": edges.tolist()},
        "uniqueness":  len(set(seqs)) / len(seqs),
        "amp_mean":    float(np.mean(scores)),
        "amp_frac_gt05": float(np.mean(scores > 0.5)),
        "samples": seqs[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n",   type=int, default=200)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- load configs ---
    with open(PRETRAIN_CFG) as f:
        pretrain_cfg = yaml.safe_load(f)
    with open(FINETUNE_CFG) as f:
        finetune_cfg = yaml.safe_load(f)

    # --- load JEPA + generator ---
    from src.models.jepa import JEPA
    from src.models.generator import ConditionalGenerator
    from src.models.encoder import TransformerEncoder
    from src.data.dataset import build_seq2seq_datasets

    gen_ckpt = torch.load(GEN_CKPT, map_location=device, weights_only=False)
    pretrain_model_cfg = gen_ckpt["pretrain_cfg"]["model"]
    gen_model_cfg      = gen_ckpt["cfg"]["generator"]

    jepa = JEPA(**pretrain_model_cfg)
    pt_ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    jepa.load_state_dict(pt_ckpt["model_state"])
    jepa.to(device).eval()
    print(f"JEPA: epoch {pt_ckpt['epoch']}, val_loss={pt_ckpt['val_loss']:.4f}")

    encoder = TransformerEncoder(
        d_model=pretrain_model_cfg["d_model"],
        nhead=pretrain_model_cfg["nhead"],
        num_layers=pretrain_model_cfg["num_layers"],
        dim_feedforward=pretrain_model_cfg["dim_feedforward"],
        dropout=pretrain_model_cfg["dropout"],
        max_seq_len=pretrain_model_cfg["max_seq_len"],
    )
    encoder.load_state_dict(jepa.context_encoder.state_dict())

    generator = ConditionalGenerator(encoder=encoder, d_model=pretrain_model_cfg["d_model"],
                                     freeze_encoder=True, **gen_model_cfg)
    generator.load_state_dict(gen_ckpt["model_state"])
    generator.to(device).eval()
    print(f"Generator: epoch {gen_ckpt['epoch']}, val_loss={gen_ckpt['val_loss']:.4f}")

    # --- datasets ---
    pretrain_data_cfg = pretrain_cfg["data"]
    fasta_paths = [PROJECT_ROOT / p for p in pretrain_data_cfg["fasta_paths"]]
    train_ds, _ = build_seq2seq_datasets(
        fasta_paths=fasta_paths,
        max_len=pretrain_data_cfg["max_len"],
        val_ratio=pretrain_data_cfg["val_ratio"],
        seed=42,
        prefix_ratio=finetune_cfg["data"].get("prefix_ratio", 0.5),
        min_prefix_len=finetune_cfg["data"].get("min_prefix_len", 3),
        max_seq_len=finetune_cfg["generator"]["max_seq_len"],
    )
    train_sequences = train_ds.sequences

    # --- fetch negatives once ---
    print("Fetching non-AMP negatives for probe …")
    from src.eval.amp_classifier import _fetch_non_amp_sequences
    neg_seqs = _fetch_non_amp_sequences(max_seqs=MAX_PROBE_SEQS)

    # --- sweep ---
    results = {}
    for temp, top_p in SWEEP:
        key = f"T{temp}_p{top_p}"
        print(f"\n[{key}] generating {args.n} sequences …")
        seqs = generate(generator, train_ds, device, args.n, temp, top_p)
        print(f"  -> {len(seqs)} seqs, mean_len={np.mean([len(s) for s in seqs]):.1f}")

        print(f"  -> scoring with JEPA probe …")
        scores = jepa_amp_score(seqs, jepa.context_encoder, device, train_sequences, neg_seqs)
        results[key] = {"temperature": temp, "top_p": top_p, **summarise(seqs, scores)}
        r = results[key]
        print(f"  -> length {r['mean_length']:.1f}±{r['std_length']:.1f}  "
              f"AMP={r['amp_frac_gt05']:.3f}  uniq={r['uniqueness']:.3f}")

    # --- save ---
    out_path = EVAL_DIR / "temp_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- summary table ---
    print(f"\n{'Key':<18} {'mean_len':>9} {'std':>6} {'AMP>0.5':>8} {'uniq':>6}")
    print("-" * 52)
    for key, r in results.items():
        print(f"{key:<18} {r['mean_length']:>9.1f} {r['std_length']:>6.1f} "
              f"{r['amp_frac_gt05']:>8.3f} {r['uniqueness']:>6.3f}")


if __name__ == "__main__":
    main()
