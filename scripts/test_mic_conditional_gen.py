"""
Test MIC-conditioned generation (v5).

For each selectivity scenario, generate 200 sequences and score with the
MIC predictor to verify the model actually follows the MIC condition.

Usage:
  uv run python -u scripts/test_mic_conditional_gen.py [--gpu 1]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_CKPT = PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt"
# Default to grampa_v5 checkpoint; fall back to 868k_v5 if not present
_GRAMPA_CKPT = PROJECT_ROOT / "checkpoints/generator_grampa_v5/best_generator.pt"
_V5_CKPT     = PROJECT_ROOT / "checkpoints/generator_868k_v5/best_generator.pt"
GEN_CKPT = _GRAMPA_CKPT if _GRAMPA_CKPT.exists() else _V5_CKPT
GEN_CFG  = PROJECT_ROOT / ("configs/finetune_grampa_v5.yaml" if _GRAMPA_CKPT.exists()
                            else "configs/finetune_868k_v5.yaml")
MIC_CKPT = PROJECT_ROOT / "checkpoints/mic_868k_transformer/best_model.pt"
MIC_CFG  = PROJECT_ROOT / "configs/mic_868k_transformer.yaml"

from src.data.supervised_dataset import GRAMPA_TOP20, N_BACTERIA

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
        if i in (0, 1): break
        if 2 <= i <= 21: out.append(aa[i - 2])
    return "".join(out)


def physchem(seq):
    n = len(seq)
    charge = sum(1 if c in POSITIVE else -1 if c in NEGATIVE else 0 for c in seq)
    gravy  = sum(KD_SCALE.get(c, 0) for c in seq) / max(n, 1)
    return charge, gravy, n


def load_generator(device):
    from src.models.jepa import JEPA
    from src.models.generator import ConditionalGeneratorV5
    from src.models.encoder import TransformerEncoder

    gen_ckpt = torch.load(GEN_CKPT, map_location=device, weights_only=False)
    pm_cfg   = gen_ckpt["pretrain_cfg"]["model"]
    gm_cfg   = gen_ckpt["cfg"]["generator"]

    pt_ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    from src.models.jepa import JEPA
    jepa = JEPA(**pm_cfg); jepa.load_state_dict(pt_ckpt["model_state"])
    enc = TransformerEncoder(**{k: pm_cfg[k] for k in
          ["d_model","nhead","num_layers","dim_feedforward","dropout","max_seq_len"]})
    enc.load_state_dict(jepa.context_encoder.state_dict())

    gen = ConditionalGeneratorV5(encoder=enc, d_model=pm_cfg["d_model"],
                                  freeze_encoder=True, **gm_cfg)
    gen.load_state_dict(gen_ckpt["model_state"])
    gen.to(device).eval()
    print(f"Generator v5 loaded (epoch {gen_ckpt['epoch']}, val_loss={gen_ckpt['val_loss']:.4f})")
    return gen


def load_mic_model(device):
    from src.models.jepa import JEPA
    from src.models.supervised_head import JEPAMICPredictor

    with open(MIC_CFG) as f:
        cfg = yaml.safe_load(f)
    pt_ckpt = torch.load(PROJECT_ROOT / cfg["pretrain_checkpoint"],
                          map_location=device, weights_only=False)
    jepa = JEPA(**pt_ckpt["cfg"]["model"])
    jepa.load_state_dict(pt_ckpt["model_state"])

    head_cfg  = cfg["head"].copy()
    head_type = head_cfg.pop("head_type", "transformer")
    model = JEPAMICPredictor(
        encoder=jepa.context_encoder, d_model=pt_ckpt["cfg"]["model"]["d_model"],
        n_bacteria=N_BACTERIA, head_type=head_type, freeze_encoder=True, **head_cfg,
    ).to(device)
    ckpt = torch.load(MIC_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_mic(mic_model, seqs, device, bacteria_indices, batch_size=256):
    """Returns dict {bact_idx: np.array of predicted log2_MIC (len=n_seqs)}"""
    from src.data.tokenizer import encode, PAD_ID
    results = {b: [] for b in bacteria_indices}
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        max_len = 48
        ids = []
        for s in batch:
            enc = [0] + [2 + "ACDEFGHIKLMNPQRSTVWY".index(c)
                         for c in s[:max_len-2] if c in "ACDEFGHIKLMNPQRSTVWY"] + [1]
            ids.append(torch.tensor(enc, dtype=torch.long))
        max_l = max(x.shape[0] for x in ids)
        padded = torch.full((len(ids), max_l), PAD_ID, dtype=torch.long)
        for j, x in enumerate(ids):
            padded[j, :len(x)] = x
        padded = padded.to(device)
        for b in bacteria_indices:
            bidx = torch.full((len(batch),), b, dtype=torch.long, device=device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                p = mic_model(padded, bidx)
            results[b].extend(p.cpu().float().tolist())
    return {b: np.array(v) for b, v in results.items()}


def generate_batch(gen, loader, conditions, device, n=200):
    """Generate n sequences given a fixed condition tensor."""
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
        cond = conditions.unsqueeze(0).expand(bs, -1).to(device)
        with torch.no_grad():
            out = gen.generate(ctx, conditions=cond, max_new_tokens=50,
                               temperature=0.9, top_p=0.9, cfg_scale=0.0)
        for row in out:
            s = _ids_to_seq(row.tolist())
            if len(s) >= 3:
                seqs.append(s)
    return seqs[:n]


def make_condition(physchem_vec, target_mic_dict, n_bacteria=N_BACTERIA):
    """Build a 43-dim condition tensor from physchem + sparse MIC targets."""
    mic_vals = torch.zeros(n_bacteria)
    mic_mask = torch.zeros(n_bacteria)
    for bact_idx, log2_mic in target_mic_dict.items():
        mic_vals[bact_idx] = log2_mic
        mic_mask[bact_idx] = 1.0
    return torch.cat([
        torch.tensor(physchem_vec, dtype=torch.float32),
        mic_vals, mic_mask
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    with open(GEN_CFG) as f:
        gcfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs/jepa_pretrain_868k.yaml") as f:
        pcfg = yaml.safe_load(f)

    pdc = pcfg["data"]
    gdc = gcfg["data"]
    ds_kwargs = dict(
        max_len=pdc["max_len"], val_ratio=pdc["val_ratio"], seed=42,
        prefix_ratio=gdc.get("prefix_ratio", 0.5),
        min_prefix_len=gdc.get("min_prefix_len", 3),
        max_seq_len=gcfg["generator"]["max_seq_len"],
    )
    if gcfg.get("generator_version") == "grampa_v5":
        from src.data.dataset import build_seq2seq_datasets_grampa_v5
        train_ds, _ = build_seq2seq_datasets_grampa_v5(
            grampa_csv=PROJECT_ROOT / gdc["grampa_csv"],
            n_repeats=1,
            **ds_kwargs,
        )
    else:
        from src.data.dataset import build_seq2seq_datasets_v5
        train_ds, _ = build_seq2seq_datasets_v5(
            fasta_paths=[PROJECT_ROOT / p for p in pdc["fasta_paths"]],
            mic_pseudolabel_npy=PROJECT_ROOT / gdc["mic_pseudolabel_npy"],
            mic_pseudolabel_seqs=PROJECT_ROOT / gdc["mic_pseudolabel_seqs"],
            mic_mask_prob=0.0,
            **ds_kwargs,
        )
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)

    gen       = load_generator(device)
    mic_model = load_mic_model(device)

    # Bacteria indices of interest
    ECOLI    = 0   # E. coli
    SAUREUS  = 1   # S. aureus
    PAERU    = 2   # P. aeruginosa
    CALB     = 3   # C. albicans (fungus)

    # Physchem baseline: moderate length, cationic (typical AMP)
    phys_amp = [0.30, 0.76, 0.30]

    scenarios = [
        # label, physchem, {bacteria_idx: log2_MIC_target}
        ("Unconditioned (no MIC)",
         [0.30, 0.0, 0.0], {}),

        ("Broad-spectrum (E.coli + S.aureus + P.aeru potent)",
         phys_amp, {ECOLI: -1.0, SAUREUS: -1.0, PAERU: -1.0}),

        ("E.coli-selective (strong E.coli, weak S.aureus)",
         phys_amp, {ECOLI: -1.5, SAUREUS: 3.0}),

        ("S.aureus-selective (strong S.aureus, weak E.coli)",
         phys_amp, {SAUREUS: -1.5, ECOLI: 3.0}),

        ("Gram-neg only (E.coli + P.aeru strong, S.aureus weak)",
         phys_amp, {ECOLI: -1.5, PAERU: -1.5, SAUREUS: 3.0}),

        ("Antifungal-selective (C.albicans strong, bacteria weak)",
         phys_amp, {CALB: -1.5, ECOLI: 3.0, SAUREUS: 3.0}),

        ("High-potency E.coli (MIC = 0.25 µg/mL)",
         phys_amp, {ECOLI: -2.0}),

        ("Inactive against all (high MIC everywhere)",
         [0.30, 0.0, 0.0], {i: 3.0 for i in range(N_BACTERIA)}),
    ]

    eval_bacteria = [ECOLI, SAUREUS, PAERU, CALB]
    bact_names = {ECOLI: "E.coli", SAUREUS: "S.aureus",
                  PAERU: "P.aeru", CALB: "C.albicans"}

    print(f"{'Scenario':<48} "
          + "  ".join(f"{bact_names[b]:>12}" for b in eval_bacteria))
    print(f"{'':48} "
          + "  ".join(f"{'tgt→act':>12}" for b in eval_bacteria))
    print("-" * 100)

    all_results = {}
    for label, phys, mic_targets in scenarios:
        cond = make_condition(phys, mic_targets)
        seqs = generate_batch(gen, loader, cond, device, n=200)
        mic_preds = predict_mic(mic_model, seqs, device, eval_bacteria)

        row = f"{label:<48}"
        res_entry = {"physchem": phys, "mic_targets": {str(k): v for k, v in mic_targets.items()},
                     "n_seqs": len(seqs), "predicted_mic": {}}
        for b in eval_bacteria:
            act = float(mic_preds[b].mean())
            tgt = mic_targets.get(b, None)
            tgt_str = f"{tgt:+.1f}" if tgt is not None else "  - "
            row += f"  {tgt_str:>4}→{act:>+5.2f}  "
            res_entry["predicted_mic"][bact_names[b]] = {
                "target": tgt, "actual_mean": act,
                "actual_std": float(mic_preds[b].std()),
            }
        print(row)

        # physicochemical stats
        pc = [physchem(s) for s in seqs]
        res_entry["mean_charge"] = float(np.mean([p[0] for p in pc]))
        res_entry["mean_gravy"]  = float(np.mean([p[1] for p in pc]))
        res_entry["mean_len"]    = float(np.mean([p[2] for p in pc]))
        res_entry["samples"]     = seqs[:5]
        all_results[label] = res_entry

    # Selectivity summary
    print(f"\n{'='*100}")
    print("SELECTIVITY ANALYSIS (E.coli vs S.aureus)")
    print(f"{'Scenario':<48} {'E.coli MIC':>12} {'S.aureus MIC':>14} {'Δ (selectivity)':>16}")
    print("-" * 95)
    for label, res in all_results.items():
        ec = res["predicted_mic"]["E.coli"]["actual_mean"]
        sa = res["predicted_mic"]["S.aureus"]["actual_mean"]
        print(f"{label:<48} {ec:>12.3f} {sa:>14.3f} {sa-ec:>+16.3f}")

    out = PROJECT_ROOT / "eval_results/mic_conditional_gen.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
