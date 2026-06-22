"""
Controllability audit: requested vs achieved physicochemical properties.

Applies a unified R²/MAE/Pearson audit to any set of generated peptides
with target property annotations. Can audit our own models OR sequences
from other papers (HydrAMP, DLFea4AMPGen, etc.) using their reported targets.

Input formats:
  --our_model:   run our generator live (requires checkpoint + config)
  --fasta:       FASTA file where headers encode targets, e.g.
                 >seq_0 charge=5.0 gravy=0.3 hc50=2.5
  --jsonl:       JSONL file, each line: {"seq": "...", "charge": 5.0, "gravy": 0.3}
  --hydra_amp:   HydrAMP-style CSV with columns: sequence, charge, hydrophobicity

Usage:
    # Audit our V4 checkpoint
    uv run python scripts/audit_generation_controllability.py \\
        --our_model --config configs/generation_control_v7.yaml --gpu 0

    # Audit sequences from another paper (provide their sequences + targets as JSONL)
    uv run python scripts/audit_generation_controllability.py \\
        --jsonl data/external/hydramp_generated.jsonl --label HydrAMP

    # Compare both at once
    uv run python scripts/audit_generation_controllability.py \\
        --our_model --config configs/generation_control_v7.yaml --gpu 0 \\
        --jsonl data/external/hydramp_generated.jsonl --label HydrAMP

Output: eval_results/controllability_audit/SUMMARY.md + metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

AA = "ACDEFGHIKLMNPQRSTVWY"
VALID_AA = set(AA)
POSITIVE = set("KRH")
NEGATIVE = set("DE")
KD_SCALE = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}
OUT_DIR = PROJECT_ROOT / "eval_results" / "controllability_audit"


# ── physicochemical computations ──────────────────────────────────────────────

def seq_charge(seq: str) -> float:
    return sum(1.0 if aa in POSITIVE else -1.0 if aa in NEGATIVE else 0.0 for aa in seq)


def seq_gravy(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(KD_SCALE.get(aa, 0.0) for aa in seq) / len(seq)


def is_valid(seq: str) -> bool:
    return len(seq) >= 3 and all(c in VALID_AA for c in seq)


# ── R² / metrics ──────────────────────────────────────────────────────────────

def r2(targets: list[float], actuals: list[float]) -> float:
    if len(targets) < 2:
        return float("nan")
    t = np.array(targets)
    a = np.array(actuals)
    ss_res = ((t - a) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    x_, y_ = np.array(x), np.array(y)
    denom = np.std(x_) * np.std(y_)
    return float(np.corrcoef(x_, y_)[0, 1]) if denom > 1e-12 else float("nan")


def mae(targets: list[float], actuals: list[float]) -> float:
    return float(np.mean(np.abs(np.array(targets) - np.array(actuals))))


def audit_property(targets: list[float], actuals: list[float], name: str) -> dict:
    return {
        "property": name,
        "n": len(targets),
        "r2_target_actual": r2(targets, actuals),
        "pearson": pearson(targets, actuals),
        "mae": mae(targets, actuals),
        "target_range": [float(min(targets)), float(max(targets))],
        "actual_mean": float(np.mean(actuals)),
        "actual_std": float(np.std(actuals)),
    }


# ── load sequences from external sources ──────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Each line: {"seq": "...", "charge": float, "gravy": float, "hc50": float}"""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} records from {path.name}")
    return records


def load_fasta_with_targets(path: Path) -> list[dict]:
    """
    FASTA where header encodes targets:
      >seq_0 charge=5.0 gravy=0.3 hc50=2.5
    """
    records = []
    seq, header = "", ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    records.append(_parse_fasta_record(header, seq))
                header, seq = line[1:], ""
            else:
                seq += line
    if seq:
        records.append(_parse_fasta_record(header, seq))
    print(f"  Loaded {len(records)} sequences from {path.name}")
    return records


def _parse_fasta_record(header: str, seq: str) -> dict:
    parts = header.split()
    rec = {"seq": seq.upper()}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = v
    return rec


def load_hydramp_csv(path: Path) -> list[dict]:
    """
    HydrAMP-style CSV: sequence, charge, hydrophobicity (GRAVY), activity
    Column names may vary — we try common variants.
    """
    import csv
    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("sequence", row.get("peptide", row.get("Sequence", ""))).strip().upper()
            if not is_valid(seq):
                continue
            rec = {"seq": seq}
            for col, key in [
                ("charge", "charge"), ("Charge", "charge"),
                ("hydrophobicity", "gravy"), ("GRAVY", "gravy"), ("Hydrophobicity", "gravy"),
                ("hc50", "hc50"), ("HC50", "hc50"),
                ("activity", "activity"), ("Activity", "activity"),
            ]:
                if col in row:
                    try:
                        rec[key] = float(row[col])
                    except ValueError:
                        pass
            records.append(rec)
    print(f"  Loaded {len(records)} records from {path.name} (HydrAMP CSV format)")
    return records


# ── run our own generator ─────────────────────────────────────────────────────

def run_our_generator(cfg_path: Path, gpu: int) -> list[dict]:
    """Generate sequences using our conditional generator and return records with targets."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    gen_cfg = cfg.get("generation", {})
    n_per   = gen_cfg.get("n_per_condition", 200)
    targets = cfg.get("targets", [])

    # Load model
    from src.models.pretrain_utils import load_pretrained_encoder
    from src.models.generator import ConditionalGeneratorV4
    from src.data.dataset import load_fasta
    from src.data.tokenizer import encode, BOS_ID, PAD_ID

    variant = cfg["variants"][-1]  # use last (best) variant
    impl    = variant["implementation"]
    ckpt_path = PROJECT_ROOT / variant["checkpoint"]

    if not ckpt_path.exists():
        print(f"  WARNING: checkpoint not found at {ckpt_path}, skipping our model")
        return []

    enc, pt_cfg = load_pretrained_encoder(
        str(PROJECT_ROOT / cfg["data"]["pretrain_config"].replace("configs/", "")
            .replace(".yaml", "")), device)

    gen_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen_model_cfg = gen_ckpt["cfg"]["generator"]
    gen = ConditionalGeneratorV4(
        encoder=enc,
        d_model=pt_cfg["model"]["d_model"],
        freeze_encoder=True,
        **gen_model_cfg,
    ).to(device)
    gen.load_state_dict(gen_ckpt["model_state"])
    gen.eval()
    print(f"  Generator loaded: {variant['key']}")

    # Build context sequences
    fasta_paths = cfg["data"].get("fasta_paths", ["data/processed/amp_corpus.fasta"])
    all_seqs = []
    for fp in fasta_paths:
        all_seqs.extend(load_fasta(PROJECT_ROOT / fp, max_len=50))
    random.shuffle(all_seqs)
    context_seqs = all_seqs[:cfg["data"].get("max_context_batches", 64) * cfg["data"].get("batch_size", 64)]

    def encode_contexts(seqs, max_len=52):
        from src.data.tokenizer import PAD_ID
        ids_list = [[BOS_ID] + encode(s[:25], add_special_tokens=False) for s in seqs]
        padded = torch.full((len(ids_list), max_len), PAD_ID, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            l = min(len(ids), max_len)
            padded[i, :l] = torch.tensor(ids[:l])
        return padded

    records = []
    num_conditions = gen_model_cfg.get("num_conditions", 3)

    for tgt in targets:
        charge_t = tgt.get("charge", 0.0)
        gravy_t  = tgt.get("gravy",  0.0)
        hc50_t   = tgt.get("hc50_log10", 2.3)
        length_t = tgt.get("length", 18)

        # Build condition vector
        if num_conditions == 4:
            cond_vals = [length_t / 50.0,
                         math.tanh(charge_t / 5.0),
                         math.tanh(gravy_t),
                         math.tanh(hc50_t / 3.0)]
        else:
            cond_vals = [length_t / 50.0,
                         math.tanh(charge_t / 5.0),
                         math.tanh(gravy_t)]

        ctx = encode_contexts(context_seqs[:n_per]).to(device)
        cond = torch.tensor([cond_vals] * n_per, dtype=torch.float32, device=device)

        with torch.no_grad():
            out = gen.generate(ctx, conditions=cond,
                               max_new_tokens=50,
                               temperature=gen_cfg.get("temperature", 0.9),
                               top_p=gen_cfg.get("top_p", 0.9))

        from src.data.tokenizer import EOS_ID
        for row in out:
            toks = row.tolist()
            seq_ids = []
            for t in toks:
                if t == EOS_ID: break
                if 2 <= t <= 21: seq_ids.append(AA[t - 2])
            seq = "".join(seq_ids)
            if is_valid(seq):
                records.append({
                    "seq": seq,
                    "target_key": tgt["key"],
                    "charge_target": charge_t,
                    "gravy_target":  gravy_t,
                    "hc50_target":   hc50_t,
                    "charge_actual": seq_charge(seq),
                    "gravy_actual":  seq_gravy(seq),
                })

    print(f"  Generated {len(records)} valid sequences across {len(targets)} targets")
    return records


# ── core audit ────────────────────────────────────────────────────────────────

def run_audit(records: list[dict], label: str) -> dict:
    """
    Given records with *_target and *_actual fields (or raw seq),
    compute requested-vs-achieved R²/Pearson/MAE for each property.
    """
    # Fill in actual properties if not pre-computed
    for r in records:
        seq = r["seq"]
        if "charge_actual" not in r:
            r["charge_actual"] = seq_charge(seq)
        if "gravy_actual" not in r:
            r["gravy_actual"] = seq_gravy(seq)

    results = {"label": label, "n_sequences": len(records), "properties": {}}

    for prop, target_key, actual_key in [
        ("charge",  "charge_target",  "charge_actual"),
        ("gravy",   "gravy_target",   "gravy_actual"),
        ("hc50",    "hc50_target",    "hc50_actual"),
    ]:
        has_target = [r for r in records if target_key in r and actual_key in r]
        if not has_target:
            continue
        tgts = [r[target_key] for r in has_target]
        acts = [r[actual_key] for r in has_target]
        if len(set(tgts)) < 2:
            continue  # no variation in targets → R² undefined
        results["properties"][prop] = audit_property(tgts, acts, prop)

    return results


def print_audit(res: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"  {res['label']}  (n={res['n_sequences']})")
    print(f"{'─'*60}")
    print(f"  {'Property':<10} {'R²':>7}  {'Pearson':>8}  {'MAE':>7}  {'target range'}")
    for prop, m in res["properties"].items():
        rng = f"[{m['target_range'][0]:.1f}, {m['target_range'][1]:.1f}]"
        r2v  = f"{m['r2_target_actual']:+.3f}" if not math.isnan(m['r2_target_actual']) else "  nan"
        pr   = f"{m['pearson']:+.3f}"           if not math.isnan(m['pearson'])          else "  nan"
        mav  = f"{m['mae']:.3f}"
        print(f"  {prop:<10} {r2v:>7}  {pr:>8}  {mav:>7}  {rng}")


def write_summary(all_results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    lines = [
        "# Controllability Audit: Requested vs Achieved Properties",
        "",
        "R² = correlation between *requested* target and *actual* computed property.",
        "R² ≈ 1.0 = perfect control. R² ≈ 0 or negative = no control.",
        "This metric is applied uniformly to all models / papers.",
        "",
        "| Model / Paper | Property | R² | Pearson | MAE | n |",
        "|---|---|---|---|---|---|",
    ]
    for res in all_results:
        for prop, m in res["properties"].items():
            r2v = f"{m['r2_target_actual']:+.3f}" if not math.isnan(m['r2_target_actual']) else "—"
            pr  = f"{m['pearson']:+.3f}"           if not math.isnan(m['pearson'])          else "—"
            mav = f"{m['mae']:.3f}"
            lines.append(f"| {res['label']} | {prop} | {r2v} | {pr} | {mav} | {m['n']} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- Charge R² > 0.80: reliable control (confirmed for JEPA-AMP V4+)",
        "- GRAVY R² ≈ 0:    no control (V4 failure; V7 targets R² > 0.40)",
        "- Papers that do not report this metric cannot be directly compared.",
        "  Their sequences can be audited by running this script on their FASTA/CSV output.",
    ]

    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {out_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",        type=int,  default=0)
    parser.add_argument("--our_model",  action="store_true",
                        help="Run our own generator (requires --config)")
    parser.add_argument("--config",     default="configs/generation_control_v7.yaml",
                        help="Generation control config for our model")
    parser.add_argument("--jsonl",      default=None,
                        help="External sequences in JSONL format (with target fields)")
    parser.add_argument("--fasta",      default=None,
                        help="External sequences in FASTA format (targets in header)")
    parser.add_argument("--hydramp",    default=None,
                        help="HydrAMP-style CSV file")
    parser.add_argument("--label",      default="External",
                        help="Label for external sequences in report")
    args = parser.parse_args()

    all_results = []

    if args.our_model:
        cfg_path = PROJECT_ROOT / args.config
        print(f"\n[Our model — {cfg_path.name}]")
        records = run_our_generator(cfg_path, args.gpu)
        if records:
            res = run_audit(records, label="JEPA-AMP V7")
            print_audit(res)
            all_results.append(res)

    if args.jsonl:
        print(f"\n[{args.label} — JSONL]")
        records = load_jsonl(Path(args.jsonl))
        res = run_audit(records, label=args.label)
        print_audit(res)
        all_results.append(res)

    if args.fasta:
        print(f"\n[{args.label} — FASTA]")
        records = load_fasta_with_targets(Path(args.fasta))
        res = run_audit(records, label=args.label)
        print_audit(res)
        all_results.append(res)

    if args.hydramp:
        print(f"\n[{args.label} — HydrAMP CSV]")
        records = load_hydramp_csv(Path(args.hydramp))
        res = run_audit(records, label=args.label)
        print_audit(res)
        all_results.append(res)

    if not all_results:
        parser.print_help()
        return

    write_summary(all_results, OUT_DIR)


if __name__ == "__main__":
    main()
