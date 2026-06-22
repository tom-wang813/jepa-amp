"""
Comprehensive generator comparison: controllability + quality + diversity.

Evaluation dimensions:
  Controllability: charge R², GRAVY R², HC50 R² (V7), MAE, Pearson for each property
  Quality:        AMP score (fraction predicted AMP by classifier), HC50 risk (mean predicted)
  Diversity:      mean pairwise normalised edit distance within each condition group
  Novelty:        exact-match novelty vs training corpus
  Validity:       fraction of sequences with only canonical AAs, length ≥ 3

Outputs:
  eval_results/comparison_formal/
    metrics.json          machine-readable full metrics
    SUMMARY.md            human-readable tables
    per_target.jsonl      per-generated-sequence records
    latex_tables.tex      ready-to-paste LaTeX tables for paper

Usage:
    uv run python scripts/compare_all_generators_formal.py \\
        --config configs/comparison_formal.yaml --gpu 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

AA = "ACDEFGHIKLMNPQRSTVWY"
VALID_AA = set(AA)
POSITIVE = set("KR")
NEGATIVE = set("DE")
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve(p): return Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def ids_to_seq(ids):
    out = []
    for t in ids:
        if t in (0, 1): break
        if 2 <= t <= 21: out.append(AA[t - 2])
    return "".join(out)

def seq_charge(seq): return sum(1 if c in POSITIVE else -1 if c in NEGATIVE else 0 for c in seq)
def seq_gravy(seq):  return sum(KD_SCALE.get(c, 0) for c in seq) / max(len(seq), 1)
def is_valid(seq):   return len(seq) >= 3 and all(c in VALID_AA for c in seq)

def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]

def diversity_score(seqs, n_sample=80):
    if len(seqs) > n_sample:
        seqs = random.sample(seqs, n_sample)
    dists = [levenshtein(seqs[i], seqs[j]) / max(len(seqs[i]), len(seqs[j]), 1)
             for i in range(len(seqs)) for j in range(i+1, len(seqs))]
    return float(np.mean(dists)) if dists else 0.0

def condition_vector(target, device, num_conditions=3):
    dims = [
        float(target["length"]) / 50.0,
        math.tanh(float(target["charge"]) / 5.0),
        math.tanh(float(target["gravy"])),
    ]
    if num_conditions >= 4:
        dims.append(math.tanh(float(target.get("hc50_log10", 2.3)) / 3.0))
    return torch.tensor(dims, dtype=torch.float32, device=device)


# ── metrics ───────────────────────────────────────────────────────────────────

def r2(targets, actuals):
    t, a = np.array(targets, float), np.array(actuals, float)
    ss_tot = ((t - t.mean())**2).sum()
    if ss_tot < 1e-12 or len(t) < 2: return float("nan")
    return float(1 - ((t - a)**2).sum() / ss_tot)

def pearson(x, y):
    x_, y_ = np.array(x, float), np.array(y, float)
    if np.std(x_) < 1e-9 or np.std(y_) < 1e-9: return float("nan")
    return float(np.corrcoef(x_, y_)[0, 1])

def mae(t, a): return float(np.mean(np.abs(np.array(t, float) - np.array(a, float))))


# ── model loading ─────────────────────────────────────────────────────────────

def load_generator(spec, device):
    from src.models.encoder import TransformerEncoder
    from src.models.generator import ConditionalGenerator, ConditionalGeneratorV3, ConditionalGeneratorV4
    from src.models.jepa import JEPA
    from src.models.pretrain_utils import load_pretrained_encoder

    ckpt_path = resolve(spec["checkpoint"])
    if not ckpt_path.exists():
        print(f"  WARNING: checkpoint missing → {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pt_cfg  = ckpt["pretrain_cfg"]["model"]
    gen_cfg = ckpt["cfg"]["generator"]
    impl    = spec.get("implementation", "v4")

    # Detect encoder type from checkpoint: MLM variant uses mlm pretrain checkpoint
    pretrain_ckpt_path = ckpt["cfg"].get("pretrain_checkpoint",
                                          "checkpoints/jepa_pretrain_868k/last_jepa.pt")
    enc, _ = load_pretrained_encoder(str(resolve(pretrain_ckpt_path)), device)

    # Dispatch to correct generator class based on implementation version
    if impl == "v2":
        gen = ConditionalGenerator(encoder=enc, d_model=pt_cfg["d_model"],
                                   freeze_encoder=True, **gen_cfg)
    elif impl == "v3":
        gen = ConditionalGeneratorV3(encoder=enc, d_model=pt_cfg["d_model"],
                                     freeze_encoder=True, **gen_cfg)
    else:  # v4, v7, mlm_v4
        gen = ConditionalGeneratorV4(encoder=enc, d_model=pt_cfg["d_model"],
                                     freeze_encoder=True, **gen_cfg)

    gen.load_state_dict(ckpt["model_state"])
    return gen.to(device).eval()


def load_amp_classifier(cfg_path, ckpt_path, device):
    from src.models.jepa import JEPA
    from src.models.supervised_head import JEPAClassifier
    with open(resolve(cfg_path)) as f:
        clf_cfg = yaml.safe_load(f)
    pt_ckpt = torch.load(resolve(clf_cfg["pretrain_checkpoint"]),
                          map_location=device, weights_only=False)
    jepa = JEPA(**pt_ckpt["cfg"]["model"])
    jepa.load_state_dict(pt_ckpt["model_state"])
    clf = JEPAClassifier(encoder=jepa.context_encoder,
                          d_model=pt_ckpt["cfg"]["model"]["d_model"],
                          freeze_encoder=True, n_tox=0,
                          **clf_cfg["head"]).to(device)
    ckpt = torch.load(resolve(ckpt_path), map_location=device, weights_only=False)
    clf.load_state_dict(ckpt["model_state"])
    return clf.eval()


def load_hc50_oracle(ckpt_path, device):
    """Returns callable: seqs (list[str]) → list[float] of predicted log10_HC50."""
    import torch.nn as nn
    from src.models.encoder import TransformerEncoder
    from src.models.jepa import JEPA
    from src.data.tokenizer import encode, BOS_ID, EOS_ID, PAD_ID

    pt_ckpt = torch.load(resolve("checkpoints/jepa_pretrain_868k/last_jepa.pt"),
                          map_location=device, weights_only=False)
    pt_cfg = pt_ckpt["cfg"]["model"]
    jepa = JEPA(**pt_cfg); jepa.load_state_dict(pt_ckpt["model_state"])
    enc = TransformerEncoder(**{k: pt_cfg[k] for k in
                                ["d_model","nhead","num_layers","dim_feedforward","dropout","max_seq_len"]})
    enc.load_state_dict(jepa.context_encoder.state_dict())

    d = pt_cfg["d_model"]
    head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,512), nn.GELU(), nn.Dropout(0.25),
                         nn.Linear(512,512), nn.GELU(), nn.Dropout(0.25), nn.Linear(512,1))
    ckpt = torch.load(resolve(ckpt_path), map_location=device, weights_only=False)
    full_state = ckpt["model_state"]
    enc_state  = {k[8:]: v for k, v in full_state.items() if k.startswith("encoder.")}
    head_state = {k[5:]: v for k, v in full_state.items() if k.startswith("head.")}
    enc.load_state_dict(enc_state)
    head.load_state_dict(head_state)
    enc.to(device).eval(); head.to(device).eval()

    for p in enc.parameters(): p.requires_grad_(False)
    for p in head.parameters(): p.requires_grad_(False)

    def predict_hc50(seqs_batch, max_len=52):
        ids_list = [[BOS_ID] + encode(s[:50], add_special_tokens=False) + [EOS_ID] for s in seqs_batch]
        lengths  = [len(x) for x in ids_list]
        padded   = torch.full((len(ids_list), max_len), PAD_ID, dtype=torch.long, device=device)
        for i, ids in enumerate(ids_list):
            l = min(len(ids), max_len)
            padded[i, :l] = torch.tensor(ids[:l], device=device)
        with torch.no_grad():
            h = enc(padded)
            pooled = torch.stack([h[i, 1:lengths[i]-1].mean(0) for i in range(len(lengths))])
            preds  = head(pooled).squeeze(-1)
        return preds.cpu().tolist()

    return predict_hc50


def load_context_loader(cfg, device):
    from src.data.dataset import build_seq2seq_datasets
    with open(resolve(cfg["data"]["pretrain_config"])) as f:
        pt_cfg = yaml.safe_load(f)
    fasta_paths = [resolve(p) for p in pt_cfg["data"]["fasta_paths"]]
    with open(resolve("configs/finetune_868k_v4.yaml")) as f:
        ft_cfg = yaml.safe_load(f)
    train_ds, _ = build_seq2seq_datasets(
        fasta_paths=fasta_paths,
        max_len=pt_cfg["data"]["max_len"],
        val_ratio=pt_cfg["data"]["val_ratio"],
        seed=int(cfg.get("seed", 42)),
        prefix_ratio=ft_cfg["data"].get("prefix_ratio", 0.5),
        min_prefix_len=ft_cfg["data"].get("min_prefix_len", 3),
        max_seq_len=ft_cfg["generator"]["max_seq_len"],
    )
    max_items = min(len(train_ds), int(cfg["data"].get("max_context_batches", 64)) *
                                   int(cfg["data"].get("batch_size", 64)))
    loader = DataLoader(Subset(train_ds, list(range(max_items))),
                        batch_size=int(cfg["data"]["batch_size"]),
                        shuffle=False, num_workers=0, drop_last=False)
    corpus_seqs = set(getattr(train_ds, "sequences", []))
    return loader, corpus_seqs


# ── generation ────────────────────────────────────────────────────────────────

def generate_for_target(generator, loader, target, variant, gen_cfg, device):
    n = int(gen_cfg["n_per_condition"])
    num_conditions = int(variant.get("num_conditions", 3))
    cond_single = condition_vector(target, device, num_conditions)
    cfg_scale   = float(variant.get("cfg_scale", 0.0))
    seqs = []
    while len(seqs) < n:
        for batch in loader:
            ctx = batch["context_ids"].to(device)
            cond = cond_single.unsqueeze(0).expand(ctx.shape[0], -1)
            with torch.no_grad():
                gen_kwargs = dict(max_new_tokens=int(gen_cfg["max_new_tokens"]),
                                  temperature=float(gen_cfg["temperature"]),
                                  top_p=float(gen_cfg["top_p"]))
                import inspect
                if "cfg_scale" in inspect.signature(generator.generate).parameters:
                    gen_kwargs["cfg_scale"] = cfg_scale
                out = generator.generate(ctx, conditions=cond, **gen_kwargs)
            for row in out:
                s = row if isinstance(row, str) else ids_to_seq(row.tolist())
                if len(s) >= 1:
                    seqs.append(s)
                    if len(seqs) >= n: break
            if len(seqs) >= n: break
    return seqs[:n]


def score_amp(seqs, classifier, device, batch_size=256):
    """Return list of AMP probability scores for each sequence."""
    from src.data.tokenizer import encode
    from src.data.supervised_dataset import collate_supervised
    scores = []
    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i:i+batch_size]
        items = [{"input_ids": torch.tensor(encode(s[:50], add_special_tokens=True), dtype=torch.long),
                  "amp_label": torch.tensor(0.0)} for s in batch_seqs]
        b = collate_supervised(items)
        ids = b["input_ids"].to(device)
        with torch.no_grad():
            out = classifier(ids)
            probs = torch.sigmoid(out["amp_logit"]).cpu().float().tolist()
        scores.extend(probs if isinstance(probs, list) else [probs])
    return scores


# ── per-variant summary ───────────────────────────────────────────────────────

def compute_variant_metrics(rows, corpus_seqs):
    out = {"by_property": {}, "by_group": {}, "by_target": {}}

    # Compute R² per property ONLY within its own dedicated sweep group.
    # Mixing other groups (where that property is fixed) collapses variance and
    # makes R² meaningless (any constant predictor beats the "target").
    group_for_prop = {
        "charge": "charge_sweep",
        "gravy":  "gravy_sweep",
        "hc50":   "hc50_sweep",
        "length": None,           # compute across all (length is always a target)
    }

    def _prop_metrics(target_key, actual_key, filter_group=None):
        subset = [r for r in rows
                  if target_key in r and actual_key in r
                  and (filter_group is None or r.get("group") == filter_group)]
        if len(subset) < 10 or len({r[target_key] for r in subset}) < 2:
            return None
        tgts = [r[target_key] for r in subset]
        acts = [r[actual_key]  for r in subset]
        return {"n": len(subset), "r2": r2(tgts, acts), "pearson": pearson(tgts, acts),
                "mae": mae(tgts, acts),
                "mean_target": float(np.mean(tgts)), "mean_actual": float(np.mean(acts))}

    for prop, tk, ak in [
        ("charge", "target_charge", "actual_charge"),
        ("gravy",  "target_gravy",  "actual_gravy"),
        ("length", "target_length", "actual_length"),
        ("hc50",   "target_hc50",   "predicted_hc50"),
    ]:
        m = _prop_metrics(tk, ak, filter_group=group_for_prop[prop])
        if m: out["by_property"][prop] = m

    seqs = [r["sequence"] for r in rows]
    out["n_sequences"]            = len(seqs)
    out["valid_fraction"]         = float(np.mean([is_valid(s) for s in seqs]))
    out["unique_fraction"]        = float(len(set(seqs)) / max(len(seqs), 1))
    out["exact_novelty_fraction"] = float(np.mean([s not in corpus_seqs for s in seqs]))
    out["mean_amp_score"]         = float(np.mean([r.get("amp_score", float("nan")) for r in rows
                                                    if not math.isnan(r.get("amp_score", float("nan")))]))
    out["mean_predicted_hc50"]    = float(np.mean([r.get("predicted_hc50", float("nan")) for r in rows
                                                    if not math.isnan(r.get("predicted_hc50", float("nan")))]))

    # By group
    for group in sorted({r.get("group", "ungrouped") for r in rows}):
        sub = [r for r in rows if r.get("group") == group]
        seqs_g = [r["sequence"] for r in sub]
        grp = {
            "n": len(sub),
            "diversity": diversity_score(seqs_g),
            "mean_amp_score": float(np.mean([r.get("amp_score", float("nan")) for r in sub
                                              if not math.isnan(r.get("amp_score", float("nan")))])),
        }
        for prop, tk, ak in [
            ("charge", "target_charge", "actual_charge"),
            ("gravy",  "target_gravy",  "actual_gravy"),
        ]:
            tgts = [r[tk] for r in sub if tk in r and ak in r]
            acts = [r[ak] for r in sub if tk in r and ak in r]
            if len(set(tgts)) >= 2 and len(tgts) >= 5:
                grp[f"{prop}_r2"]  = r2(tgts, acts)
                grp[f"{prop}_mae"] = mae(tgts, acts)
        out["by_group"][group] = grp

    # By target
    for tgt_key in sorted({r["target_key"] for r in rows}):
        sub = [r for r in rows if r["target_key"] == tgt_key]
        out["by_target"][tgt_key] = {
            "n": len(sub),
            "target_charge": sub[0]["target_charge"],
            "target_gravy":  sub[0]["target_gravy"],
            "mean_charge":   float(np.mean([r["actual_charge"] for r in sub])),
            "mean_gravy":    float(np.mean([r["actual_gravy"]  for r in sub])),
            "mean_amp_score": float(np.mean([r.get("amp_score", float("nan")) for r in sub
                                              if not math.isnan(r.get("amp_score", float("nan")))])),
            "diversity":     diversity_score([r["sequence"] for r in sub]),
        }

    return out


# ── output ────────────────────────────────────────────────────────────────────

def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.{digits}f}" if isinstance(v, float) else str(v)


def write_summary_md(out_dir, all_metrics):
    lines = [
        "# Generator Comparison: Controllability + Quality",
        "",
        "## 1. Controllability (requested-vs-achieved R²)",
        "",
        "R² across all targets in that property's sweep. R²=1 = perfect control.",
        "R² ≈ 0 = uncontrolled. R² < 0 = anti-correlated (worse than mean prediction).",
        "",
    ]

    has_hc50 = any("hc50" in v.get("by_property", {}) for v in all_metrics.values())

    # Table 1: R² overview
    header = "| Variant | Charge R² | Charge MAE | GRAVY R² | GRAVY MAE"
    if has_hc50:
        header += " | HC50 R² | HC50 MAE"
    header += " | Length R² |"
    lines += [header,
              "|---|---:|---:|---:|---:" + ("|---:|---:" if has_hc50 else "") + "|---:|"]

    for var_key, var_m in all_metrics.items():
        bp = var_m.get("by_property", {})
        row = (f"| {var_key} "
               f"| {_fmt(bp.get('charge',{}).get('r2'))} "
               f"| {_fmt(bp.get('charge',{}).get('mae'))} "
               f"| {_fmt(bp.get('gravy',{}).get('r2'))} "
               f"| {_fmt(bp.get('gravy',{}).get('mae'))}")
        if has_hc50:
            row += (f" | {_fmt(bp.get('hc50',{}).get('r2'))} "
                    f"| {_fmt(bp.get('hc50',{}).get('mae'))}")
        row += f" | {_fmt(bp.get('length',{}).get('r2'))} |"
        lines.append(row)

    lines += [
        "",
        "## 2. Generation Quality",
        "",
        "| Variant | Validity | Novelty | Uniqueness | Mean AMP Score | Mean pred HC50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for var_key, var_m in all_metrics.items():
        lines.append(
            f"| {var_key} "
            f"| {var_m.get('valid_fraction',float('nan')):.3f} "
            f"| {var_m.get('exact_novelty_fraction',float('nan')):.3f} "
            f"| {var_m.get('unique_fraction',float('nan')):.3f} "
            f"| {var_m.get('mean_amp_score',float('nan')):.3f} "
            f"| {var_m.get('mean_predicted_hc50',float('nan')):.2f} |"
        )

    lines += [
        "",
        "## 3. Per-group Diversity",
        "",
        "| Variant | Group | Diversity | Charge R² | GRAVY R² | Mean AMP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for var_key, var_m in all_metrics.items():
        for grp, gm in var_m.get("by_group", {}).items():
            lines.append(
                f"| {var_key} | {grp} "
                f"| {gm.get('diversity', float('nan')):.3f} "
                f"| {_fmt(gm.get('charge_r2'))} "
                f"| {_fmt(gm.get('gravy_r2'))} "
                f"| {gm.get('mean_amp_score', float('nan')):.3f} |"
            )

    lines += [
        "",
        "## 4. Comparison with Literature",
        "",
        "### HydrAMP (Szymczak et al. 2023) — actual data from released results",
        "",
        "HydrAMP **does not condition on charge or GRAVY**.",
        "Its two generation modes are:",
        "  1. `unconstrained_generation` — samples from the AMP region of latent space (binary amp/mic conditions only)",
        "  2. `analogue_generation` — adds noise to existing peptide embeddings, optimises for amp/mic improvement",
        "",
        "Measured from `hydra_unconstrained.csv` (n=149, released by authors):",
        "  - charge: mean=+3.0, std=1.04, range=[1,6]  — always cationic, no user control",
        "  - GRAVY:  mean=-0.44, std=0.73              — mostly hydrophilic, no user control",
        "  - AMP score: mean=0.981                      — high AMP quality",
        "",
        "**Conclusion: R² is undefined for HydrAMP** — no charge/GRAVY targets exist.",
        "The model generates high-quality AMPs but cannot design to a specified charge or hydrophobicity.",
        "",
        "| Method | Conditioning targets | Charge R² | GRAVY R² | HC50 R² | Mean AMP↑ |",
        "|---|---|:---:|:---:|:---:|---:|",
        "| HydrAMP (Szymczak 2023)  | amp/mic binary only        | — (no target) | — (no target) | — | **0.981** |",
        "| PepCVAE (Das 2021)       | AMP class label            | — | — | — | — |",
        "| TG-CDDPM (Li 2023)       | MIC range, toxicity        | — | — | — | — |",
        "| JEPA-AMP V4 (ours)       | charge, GRAVY, length      | +0.782 | +0.210 | — | 0.409 |",
        "| **JEPA-AMP V7 (ours)**   | **charge, GRAVY, HC50**    | (pending) | (pending) | (pending) | **0.491** |",
        "",
        "> HydrAMP's AMP quality is high because it optimises for AMP activity.",
        "> Our model trades some AMP score for explicit physicochemical controllability.",
        "> V7's AMP score (0.491 at epoch 5) is expected to improve further with training.",
    ]

    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def write_latex_tables(out_dir, all_metrics):
    has_hc50 = any("hc50" in v.get("by_property", {}) for v in all_metrics.values())

    def nan2dash(v):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "---"
        return f"{v:.3f}"

    # Table 1: Controllability
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Generator controllability: requested-vs-achieved R\textsuperscript{2} and MAE.",
        r"  R\textsuperscript{2} computed across all targets in the sweep for each axis.",
        r"  R\textsuperscript{2}$\!\approx\!1$ = reliable control; $\approx\!0$ = uncontrolled.}",
        r"\label{tab:gen_control}",
        r"\small",
    ]
    if has_hc50:
        lines += [r"\begin{tabular}{lcccccccc}",
                  r"\toprule",
                  r"\textbf{Method} & \multicolumn{2}{c}{\textbf{Charge}} & \multicolumn{2}{c}{\textbf{GRAVY}} & \multicolumn{2}{c}{\textbf{HC50 (log)}} & \textbf{Novelty} & \textbf{AMP$\uparrow$} \\",
                  r"\cmidrule(r){2-3}\cmidrule(r){4-5}\cmidrule(r){6-7}",
                  r"& R\textsuperscript{2} & MAE & R\textsuperscript{2} & MAE & R\textsuperscript{2} & MAE & & \\",
                  r"\midrule"]
    else:
        lines += [r"\begin{tabular}{lccccccc}",
                  r"\toprule",
                  r"\textbf{Method} & \multicolumn{2}{c}{\textbf{Charge}} & \multicolumn{2}{c}{\textbf{GRAVY}} & \textbf{Length R\textsuperscript{2}} & \textbf{Novelty} & \textbf{AMP$\uparrow$} \\",
                  r"\cmidrule(r){2-3}\cmidrule(r){4-5}",
                  r"& R\textsuperscript{2} & MAE & R\textsuperscript{2} & MAE & & & \\",
                  r"\midrule"]

    for var_key, var_m in all_metrics.items():
        bp = var_m.get("by_property", {})
        c_r2  = nan2dash(bp.get("charge",{}).get("r2"))
        c_mae = nan2dash(bp.get("charge",{}).get("mae"))
        g_r2  = nan2dash(bp.get("gravy",{}).get("r2"))
        g_mae = nan2dash(bp.get("gravy",{}).get("mae"))
        nov   = f"{var_m.get('exact_novelty_fraction', float('nan')):.3f}"
        amp   = f"{var_m.get('mean_amp_score', float('nan')):.3f}"
        bold  = var_key.startswith("JEPA-AMP")

        if has_hc50:
            h_r2  = nan2dash(bp.get("hc50",{}).get("r2"))
            h_mae = nan2dash(bp.get("hc50",{}).get("mae"))
            key_str = f"\\textbf{{{var_key}}}" if bold else var_key
            lines.append(f"{key_str} & {c_r2} & {c_mae} & {g_r2} & {g_mae} & {h_r2} & {h_mae} & {nov} & {amp} \\\\")
        else:
            l_r2 = nan2dash(bp.get("length",{}).get("r2"))
            key_str = f"\\textbf{{{var_key}}}" if bold else var_key
            lines.append(f"{key_str} & {c_r2} & {c_mae} & {g_r2} & {g_mae} & {l_r2} & {nov} & {amp} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table 2: Literature comparison
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Methodology comparison with existing AMP conditional generators.",
        r"  ``Req.-vs.-Ach.\ R\textsuperscript{2}'' = whether systematic requested-vs-achieved",
        r"  correlation is reported across a sweep of targets (not just mean statistics).}",
        r"\label{tab:lit_comparison}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Conditioning} & \textbf{R\textsuperscript{2} reported?} & \textbf{Charge ctrl} & \textbf{GRAVY ctrl} & \textbf{HC50 ctrl} \\",
        r"\midrule",
        r"HydrAMP~\cite{szymczak2023hydramp}     & amp/mic binary$^\dagger$   & \ding{55} & \multicolumn{2}{c}{--- (untargeted)} & \textbf{0.981} \\",
        r"PepCVAE~\cite{das2021accelerated}       & AMP class label            & \ding{55} & \multicolumn{2}{c}{---} & --- \\",
        r"TG-CDDPM~\cite{li2023tgcddpm}           & MIC range, toxicity        & \ding{55} & \multicolumn{2}{c}{---} & --- \\",
        r"\midrule",
        r"\textbf{JEPA-AMP V4 (ours)} & charge, GRAVY, len  & \ding{51} & 0.782 & 0.210 & 0.409 \\",
        r"\textbf{JEPA-AMP V7 (ours)} & charge, GRAVY, HC50 & \ding{51} & \textit{pending} & \textit{pending} & 0.491$^*$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (out_dir / "latex_tables.tex").write_text("\n".join(lines) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_missing", action="store_true",
                        help="Skip variants whose checkpoint is missing (useful if V7 not done yet)")
    args = parser.parse_args()

    with open(resolve(args.config)) as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve(args.config), out_dir / "config_resolved.yaml")

    # Load shared utilities
    loader, corpus_seqs = load_context_loader(cfg, device)
    print(f"Context loader: {sum(1 for _ in loader) * cfg['data']['batch_size']} sequences")

    amp_clf = None
    qual_cfg = cfg.get("quality", {})
    if qual_cfg.get("amp_classifier_ckpt") and resolve(qual_cfg["amp_classifier_ckpt"]).exists():
        try:
            amp_clf = load_amp_classifier(qual_cfg["amp_classifier_cfg"],
                                           qual_cfg["amp_classifier_ckpt"], device)
            print("AMP classifier loaded.")
        except Exception as e:
            print(f"WARNING: could not load AMP classifier: {e}")

    hc50_fn = None
    if qual_cfg.get("hc50_oracle_ckpt") and resolve(qual_cfg["hc50_oracle_ckpt"]).exists():
        try:
            hc50_fn = load_hc50_oracle(qual_cfg["hc50_oracle_ckpt"], device)
            print("HC50 oracle loaded.")
        except Exception as e:
            print(f"WARNING: could not load HC50 oracle: {e}")

    all_rows: list[dict] = []
    all_metrics: dict[str, Any] = {}

    for vi, variant in enumerate(cfg["variants"]):
        var_key = variant["key"]
        print(f"\n{'='*60}")
        print(f"Variant {vi+1}/{len(cfg['variants'])}: {var_key}")
        print(f"  ckpt: {variant['checkpoint']}")

        generator = load_generator(variant, device)
        if generator is None:
            if args.skip_missing:
                print(f"  SKIPPED (checkpoint missing)")
                continue
            else:
                raise FileNotFoundError(f"Checkpoint not found: {variant['checkpoint']}")

        variant_rows = []
        for ti, target in enumerate(cfg["targets"]):
            set_seed(int(cfg.get("seed", 42)) + 1000 * vi + ti)
            seqs = generate_for_target(generator, loader, target, variant,
                                       cfg["generation"], device)
            print(f"  {target['key']:20s}: {len(seqs)} seqs", end="")

            # AMP scores
            amp_scores = ([float("nan")] * len(seqs) if amp_clf is None
                          else score_amp(seqs, amp_clf, device))

            # HC50 predictions
            hc50_preds = ([float("nan")] * len(seqs) if hc50_fn is None
                          else hc50_fn(seqs))

            for si, (seq, amp_s, hc50_p) in enumerate(zip(seqs, amp_scores, hc50_preds)):
                row = {
                    "variant":        var_key,
                    "target_key":     target["key"],
                    "group":          target.get("group", "ungrouped"),
                    "sample_index":   si,
                    "sequence":       seq,
                    "target_charge":  float(target["charge"]),
                    "target_gravy":   float(target["gravy"]),
                    "target_length":  float(target["length"]),
                    "actual_charge":  float(seq_charge(seq)),
                    "actual_gravy":   float(seq_gravy(seq)),
                    "actual_length":  float(len(seq)),
                    "amp_score":      float(amp_s),
                    "predicted_hc50": float(hc50_p),
                    "valid":          is_valid(seq),
                    "novel":          seq not in corpus_seqs,
                }
                if "hc50_log10" in target:
                    row["target_hc50"] = float(target["hc50_log10"])
                variant_rows.append(row)
                all_rows.append(row)

            mean_amp = np.mean([r["amp_score"] for r in variant_rows
                                 if not math.isnan(r["amp_score"])]) if variant_rows else float("nan")
            print(f"  | AMP={mean_amp:.2f}")

        all_metrics[var_key] = compute_variant_metrics(variant_rows, corpus_seqs)
        del generator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save outputs
    with open(out_dir / "per_target.jsonl", "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    full_metrics = {
        "experiment_id":       cfg["experiment_id"],
        "research_decision":   cfg["research_decision"],
        "n_rows":              len(all_rows),
        "n_targets":           len(cfg["targets"]),
        "n_variants":          len(all_metrics),
        "variants":            all_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    write_summary_md(out_dir, all_metrics)
    write_latex_tables(out_dir, all_metrics)

    print(f"\nArtifacts written to {out_dir}/")
    print(f"  metrics.json  per_target.jsonl  SUMMARY.md  latex_tables.tex")

    # Quick print
    print("\n── Controllability Summary ──────────────────────────────")
    print(f"  {'Variant':<25} {'Charge R²':>10} {'GRAVY R²':>10} {'HC50 R²':>10} {'AMP↑':>8}")
    for var_key, var_m in all_metrics.items():
        bp = var_m["by_property"]
        c = bp.get("charge", {}).get("r2", float("nan"))
        g = bp.get("gravy",  {}).get("r2", float("nan"))
        h = bp.get("hc50",   {}).get("r2", float("nan"))
        a = var_m.get("mean_amp_score", float("nan"))
        print(f"  {var_key:<25} {c:>10.3f} {g:>10.3f} {h:>10.3f} {a:>8.3f}")


if __name__ == "__main__":
    main()
