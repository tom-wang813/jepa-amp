"""
Evaluate quality of conditionally generated sequences.

For each v4 condition target, generates 200 sequences and reports:
  - AMP classifier score (fraction predicted AMP)
  - Mean physicochemical properties vs target
  - Sequence diversity (mean pairwise Levenshtein distance / mean length)
  - Novelty (fraction not in training set, by exact match)
  - Sample sequences

Usage:
  uv run python -u scripts/eval_generator_quality.py [--gpu 1]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_CKPT = PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt"
CLASSIFIER_CKPT = PROJECT_ROOT / "checkpoints/amp_classifier_v6/best_model.pt"
CLASSIFIER_CFG  = PROJECT_ROOT / "configs/amp_classifier_v6.yaml"
GENERATOR_CKPT  = PROJECT_ROOT / "checkpoints/generator_868k_v4/best_generator.pt"
GENERATOR_CFG   = PROJECT_ROOT / "configs/finetune_868k_v4.yaml"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
POSITIVE = set("KR")
NEGATIVE = set("DE")
KD_SCALE = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
             "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
             "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}


def _ids_to_seq(ids):
    aa = "ACDEFGHIKLMNPQRSTVWY"
    out = []
    for i in ids:
        if i in (0, 1):
            break
        if 2 <= i <= 21:
            out.append(aa[i - 2])
    return "".join(out)


def physchem(seq):
    n = len(seq)
    charge = sum(1 if c in POSITIVE else -1 if c in NEGATIVE else 0 for c in seq)
    gravy = sum(KD_SCALE.get(c, 0) for c in seq) / max(n, 1)
    return {"length": n, "charge": charge, "gravy": gravy}


def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]


def diversity(seqs, sample=100):
    """Mean pairwise normalised edit distance on a random sample."""
    if len(seqs) > sample:
        idx = np.random.choice(len(seqs), sample, replace=False)
        seqs = [seqs[i] for i in idx]
    dists = []
    for i in range(len(seqs)):
        for j in range(i+1, len(seqs)):
            ml = max(len(seqs[i]), len(seqs[j]), 1)
            dists.append(levenshtein(seqs[i], seqs[j]) / ml)
    return float(np.mean(dists)) if dists else 0.0


def load_training_seqs():
    import yaml
    with open(GENERATOR_CFG) as f:
        cfg = yaml.safe_load(f)
    seqs = set()
    for p in cfg["data"]["fasta_paths"]:
        path = PROJECT_ROOT / p
        cur = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if cur:
                        seqs.add("".join(cur).upper())
                    cur = []
                else:
                    cur.append(line)
            if cur:
                seqs.add("".join(cur).upper())
    return seqs


def load_generator(device):
    import yaml
    from src.models.jepa import JEPA
    from src.models.generator import ConditionalGeneratorV4
    from src.models.encoder import TransformerEncoder

    gen_ckpt = torch.load(GENERATOR_CKPT, map_location=device, weights_only=False)
    pm_cfg   = gen_ckpt["pretrain_cfg"]["model"]
    gm_cfg   = gen_ckpt["cfg"]["generator"]

    pt_ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    jepa = JEPA(**pm_cfg); jepa.load_state_dict(pt_ckpt["model_state"])
    enc  = TransformerEncoder(**{k: pm_cfg[k] for k in
           ["d_model","nhead","num_layers","dim_feedforward","dropout","max_seq_len"]})
    enc.load_state_dict(jepa.context_encoder.state_dict())

    gen = ConditionalGeneratorV4(encoder=enc, d_model=pm_cfg["d_model"],
                                  freeze_encoder=True, **gm_cfg)
    gen.load_state_dict(gen_ckpt["model_state"])
    gen.to(device).eval()
    print(f"Generator v4 loaded (epoch {gen_ckpt['epoch']}, val_loss={gen_ckpt['val_loss']:.4f})")
    return gen


def load_classifier(device):
    import yaml
    from src.models.jepa import JEPA
    from src.models.supervised_head import JEPAClassifier

    with open(CLASSIFIER_CFG) as f:
        cfg = yaml.safe_load(f)
    pt_ckpt = torch.load(PROJECT_ROOT / cfg["pretrain_checkpoint"],
                          map_location=device, weights_only=False)
    jepa = JEPA(**pt_ckpt["cfg"]["model"])
    jepa.load_state_dict(pt_ckpt["model_state"])

    model = JEPAClassifier(encoder=jepa.context_encoder,
                            d_model=pt_ckpt["cfg"]["model"]["d_model"],
                            freeze_encoder=True, n_tox=0, **cfg["head"]).to(device)
    ckpt = torch.load(CLASSIFIER_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, pt_ckpt["cfg"]["model"].get("max_seq_len", 52)


@torch.no_grad()
def score_amp(classifier, max_seq_len, seqs, device, batch_size=256):
    from src.data.tokenizer import encode
    from src.data.supervised_dataset import collate_supervised

    scores = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        max_aa = max_seq_len - 2
        items = [{"input_ids": torch.tensor(encode(s[:max_aa], add_special_tokens=True),
                               dtype=torch.long), "amp_label": torch.tensor(0.0)}
                 for s in batch]
        b = collate_supervised(items)
        ids = b["input_ids"].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = classifier(ids)
        scores.extend(torch.sigmoid(out["amp_logit"]).cpu().float().tolist())
    return np.array(scores)


def generate_for_cond(gen, loader, cond_tensor, device, n=200):
    seqs = []
    loader_iter = iter(loader)
    while len(seqs) < n:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        bs = batch["context_ids"].shape[0]
        ctx  = batch["context_ids"].to(device)
        cond = cond_tensor.unsqueeze(0).expand(bs, -1).to(device)
        with torch.no_grad():
            out = gen.generate(ctx, conditions=cond, max_new_tokens=50,
                               temperature=0.9, top_p=0.9, cfg_scale=0.0)
        for row in out:
            s = _ids_to_seq(row.tolist())
            if len(s) >= 3:
                seqs.append(s)
    return seqs[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import yaml
    from src.data.dataset import build_seq2seq_datasets

    with open(PROJECT_ROOT / "configs/jepa_pretrain_868k.yaml") as f:
        pcfg = yaml.safe_load(f)
    with open(GENERATOR_CFG) as f:
        gcfg = yaml.safe_load(f)
    pdc = pcfg["data"]

    train_ds, _ = build_seq2seq_datasets(
        [PROJECT_ROOT / p for p in pdc["fasta_paths"]],
        max_len=pdc["max_len"], val_ratio=pdc["val_ratio"], seed=42,
        prefix_ratio=gcfg["data"].get("prefix_ratio", 0.5),
        min_prefix_len=gcfg["data"].get("min_prefix_len", 3),
        max_seq_len=gcfg["generator"]["max_seq_len"],
    )
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)

    gen = load_generator(device)
    clf, max_seq_len = load_classifier(device)

    print("Loading training sequences for novelty check...")
    train_seqs = load_training_seqs()
    print(f"  {len(train_seqs):,} training sequences loaded")

    targets = [
        ("AMP-like (+charge, hydrophob)",   0.40,  0.76,  0.30),
        ("High +charge, short",             0.20,  0.95,  0.00),
        ("Very high +charge",               0.40,  0.99,  0.00),
        ("Neutral, hydrophobic",            0.30,  0.00,  0.70),
        ("Strongly hydrophobic",            0.30,  0.00,  0.90),
        ("Anionic (−charge)",               0.30, -0.76,  0.00),
        ("Strong anionic",                  0.40, -0.95,  0.00),
        ("Long, neutral, hydrophilic",      0.60,  0.00, -0.60),
        ("Cationic + hydrophobic (AMP)",    0.30,  0.76,  0.60),
        ("Anionic + hydrophilic",           0.40, -0.60, -0.60),
    ]

    results = {}
    print(f"\n{'Condition':<35} {'tgt_chg':>8} {'act_chg':>8} {'tgt_gvy':>8} {'act_gvy':>8} "
          f"{'frac_AMP':>9} {'diversity':>10} {'novelty':>9}")
    print("-" * 105)

    for label, ln, tc, tg in targets:
        cond = torch.tensor([ln, tc, tg], dtype=torch.float32)
        tgt_charge = math.atanh(tc) * 5
        tgt_gravy  = math.atanh(max(min(tg, 0.9999), -0.9999))

        seqs = generate_for_cond(gen, loader, cond, device, n=200)
        pc   = [physchem(s) for s in seqs]

        amp_scores = score_amp(clf, max_seq_len, seqs, device)
        div  = diversity(seqs)
        nov  = float(np.mean([s not in train_seqs for s in seqs]))

        mean_charge = float(np.mean([p["charge"] for p in pc]))
        mean_gravy  = float(np.mean([p["gravy"]  for p in pc]))
        frac_amp    = float(np.mean(amp_scores >= 0.5))

        print(f"{label:<35} {tgt_charge:>8.1f} {mean_charge:>8.2f} "
              f"{tgt_gravy:>8.2f} {mean_gravy:>8.2f} "
              f"{frac_amp:>9.3f} {div:>10.3f} {nov:>9.3f}")

        results[label] = {
            "target": {"charge": tgt_charge, "gravy": tgt_gravy, "len": ln * 50},
            "actual": {"charge": mean_charge, "gravy": mean_gravy,
                       "len": float(np.mean([p["length"] for p in pc]))},
            "frac_amp": frac_amp, "diversity": div, "novelty": nov,
            "mean_amp_score": float(amp_scores.mean()),
            "samples": seqs[:5],
        }

    out = PROJECT_ROOT / "eval_results/generator_v4_quality.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
