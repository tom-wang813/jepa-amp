"""
Test whether the conditional generator actually respects its conditions.

Conditions (3-vector):
  [length/50,  tanh(charge/5),  tanh(gravy)]

We sweep over target charge and hydrophobicity, generate 100 sequences each,
and report the actual physicochemical properties of the generated sequences.

Usage:
  uv run python -u scripts/test_conditional_gen.py [--gpu 0]
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
PRETRAIN_CFG  = PROJECT_ROOT / "configs/jepa_pretrain_868k.yaml"

# v2 = original single-token condition; v3 = AdaLN per-layer condition
GENERATORS = {
    "v2": {
        "ckpt": PROJECT_ROOT / "checkpoints/generator_868k_v2/best_generator.pt",
        "cfg":  PROJECT_ROOT / "configs/finetune_868k_v2.yaml",
    },
    "v3": {
        "ckpt": PROJECT_ROOT / "checkpoints/generator_868k_v3/best_generator.pt",
        "cfg":  PROJECT_ROOT / "configs/finetune_868k_v3.yaml",
    },
    "v4": {
        "ckpt": PROJECT_ROOT / "checkpoints/generator_868k_v4/best_generator.pt",
        "cfg":  PROJECT_ROOT / "configs/finetune_868k_v4.yaml",
    },
}
GEN_CKPT     = PROJECT_ROOT / "checkpoints/generator_868k_v2/best_generator.pt"
FINETUNE_CFG = PROJECT_ROOT / "configs/finetune_868k.yaml"

VALID_AA  = set("ACDEFGHIKLMNPQRSTVWY")
POSITIVE  = set("KR")
NEGATIVE  = set("DE")
KD_SCALE  = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
              "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
              "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}


def _ids_to_seq(ids: list[int]) -> str:
    aa = "ACDEFGHIKLMNPQRSTVWY"
    out = []
    for i in ids:
        if i in (0, 1):
            break
        if 2 <= i <= 21:
            out.append(aa[i - 2])
    return "".join(out)


def physchem(seq: str) -> dict:
    n = len(seq)
    charge = sum(1 if c in POSITIVE else -1 if c in NEGATIVE else 0 for c in seq)
    gravy = sum(KD_SCALE.get(c, 0) for c in seq) / max(n, 1)
    return {"length": n, "charge": charge, "gravy": gravy}


def generate_with_cond(generator, cond_tensor: torch.Tensor, device: torch.device,
                       train_loader, n: int = 100, cfg_scale: float = 0.0) -> list[str]:
    """Generate n sequences with optional CFG guidance.
    cfg_scale=0: standard conditional generation
    cfg_scale>0: logits = uncond + cfg_scale * (cond - uncond)
    """
    seqs = []
    loader_iter = iter(train_loader)
    null_cond = torch.zeros(1, cond_tensor.shape[0], dtype=torch.float32)  # uncond = zeros

    while len(seqs) < n:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        bs = batch["context_ids"].shape[0]
        ctx  = batch["context_ids"].to(device)
        cond = cond_tensor.unsqueeze(0).expand(bs, -1).to(device)

        is_v3 = hasattr(generator, 'cond_encoder')
        if cfg_scale > 0 and not is_v3:
            uncond = null_cond.expand(bs, -1).to(device)
            out = _generate_cfg(generator, ctx, cond, uncond, cfg_scale, max_new_tokens=50)
        else:
            with torch.no_grad():
                kw = dict(cfg_scale=cfg_scale) if is_v3 else {}
                out = generator.generate(ctx, conditions=cond, max_new_tokens=50,
                                         temperature=0.9, top_p=0.9, **kw)
        for row in out:
            s = _ids_to_seq(row.tolist())
            if len(s) >= 3:
                seqs.append(s)
    return seqs[:n]


def _generate_cfg(generator, ctx, cond, uncond, scale, max_new_tokens=55):
    """CFG decoding: guided = uncond + scale*(cond - uncond), greedy."""
    import torch.nn.functional as F
    B = ctx.shape[0]
    device = ctx.device
    EOS_ID, BOS_ID = 1, 0

    h_ctx = generator.encoder(ctx)
    h_ctx = generator.adapter(h_ctx)

    mem_cond   = torch.cat([generator.condition_proj(cond),   h_ctx], dim=1)
    mem_uncond = torch.cat([generator.condition_proj(uncond), h_ctx], dim=1)

    generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    finished  = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits_c = generator.decoder(generated, memory=mem_cond)[:, -1, :]
        logits_u = generator.decoder(generated, memory=mem_uncond)[:, -1, :]
        logits   = logits_u + scale * (logits_c - logits_u)

        # greedy decoding for CFG (avoids numerical issues with guided logits)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
        next_tok = logits.argmax(dim=-1, keepdim=True).clamp(0, logits.shape[-1] - 1)

        finished |= (next_tok.squeeze(1) == EOS_ID)
        generated = torch.cat([generated, next_tok], dim=1)
        if finished.all():
            break

    return generated[:, 1:]  # strip BOS


def summarise(seqs: list[str]) -> dict:
    pc = [physchem(s) for s in seqs]
    return {
        "n":           len(seqs),
        "mean_len":    float(np.mean([p["length"] for p in pc])),
        "mean_charge": float(np.mean([p["charge"] for p in pc])),
        "mean_gravy":  float(np.mean([p["gravy"]  for p in pc])),
        "frac_positive_charge": float(np.mean([p["charge"] > 0 for p in pc])),
        "samples": seqs[:3],
    }


def _load_generator(version: str, device: torch.device):
    import yaml
    from src.models.jepa import JEPA
    from src.models.generator import ConditionalGenerator, ConditionalGeneratorV3, ConditionalGeneratorV4
    from src.models.encoder import TransformerEncoder

    spec = GENERATORS[version]
    gen_ckpt = torch.load(spec["ckpt"], map_location=device, weights_only=False)
    pretrain_model_cfg = gen_ckpt["pretrain_cfg"]["model"]
    gen_model_cfg      = gen_ckpt["cfg"]["generator"]

    pt_ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    jepa = JEPA(**pretrain_model_cfg)
    jepa.load_state_dict(pt_ckpt["model_state"])

    encoder = TransformerEncoder(**{k: pretrain_model_cfg[k] for k in
                                    ["d_model","nhead","num_layers","dim_feedforward","dropout","max_seq_len"]})
    encoder.load_state_dict(jepa.context_encoder.state_dict())

    if version == "v4":
        GenClass = ConditionalGeneratorV4
    elif version == "v3":
        GenClass = ConditionalGeneratorV3
    else:
        GenClass = ConditionalGenerator
    generator = GenClass(encoder=encoder, d_model=pretrain_model_cfg["d_model"],
                         freeze_encoder=True, **gen_model_cfg)
    generator.load_state_dict(gen_ckpt["model_state"])
    generator.to(device).eval()
    print(f"Generator {version} loaded (epoch {gen_ckpt['epoch']}, val_loss={gen_ckpt['val_loss']:.4f})")
    return generator, gen_ckpt["cfg"]


def _run_cond_test(generator, train_loader, device, targets, version_label):
    """Run cfg_scale=0 (standard) and cfg_scale=5 for v3 (AdaLN supports built-in CFG)."""
    cfg_scales = [0.0, 5.0] if ("v3" in version_label or "v4" in version_label) else [0.0]
    results_by_scale = {}

    for cfg_scale in cfg_scales:
        tag = f"{version_label} cfg={cfg_scale:.0f}"
        print(f"\n{'='*60}  {tag}")
        results = {}
        for label, ln, tc, tg in targets:
            cond = torch.tensor([ln, tc, tg], dtype=torch.float32)
            tgt_charge = math.atanh(tc) * 5
            tgt_gravy  = math.atanh(max(min(tg, 0.9999), -0.9999))
            seqs = generate_with_cond(generator, cond, device, train_loader,
                                      n=100, cfg_scale=cfg_scale)
            s = summarise(seqs)
            print(f"  [{label}]  tgt chg={tgt_charge:.1f} gravy={tgt_gravy:.2f}  "
                  f"→ actual chg={s['mean_charge']:.2f} gravy={s['mean_gravy']:.2f} "
                  f"len={s['mean_len']:.1f}")
            results[label] = {"target": {"charge_raw": tgt_charge, "gravy_raw": tgt_gravy,
                                          "len": ln*50}, **s}
        results_by_scale[tag] = results
    return results_by_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--versions", nargs="+", default=["v2", "v3"])
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    import yaml
    from src.data.dataset import build_seq2seq_datasets

    with open(PRETRAIN_CFG) as f:
        pretrain_cfg = yaml.safe_load(f)
    pdc = pretrain_cfg["data"]
    fasta_paths = [PROJECT_ROOT / p for p in pdc["fasta_paths"]]

    # Use v3 finetune cfg for data loader (same data setup)
    with open(GENERATORS["v3"]["cfg"]) as f:
        ft_cfg = yaml.safe_load(f)
    train_ds, _ = build_seq2seq_datasets(
        fasta_paths=fasta_paths, max_len=pdc["max_len"], val_ratio=pdc["val_ratio"],
        seed=42, prefix_ratio=ft_cfg["data"].get("prefix_ratio", 0.5),
        min_prefix_len=ft_cfg["data"].get("min_prefix_len", 3),
        max_seq_len=ft_cfg["generator"]["max_seq_len"],
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)

    targets = [
        # (label, len/50, tanh(charge/5), tanh(gravy))
        ("AMP-like  (+charge, hydrophob)",  0.40,  0.76,  0.30),
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

    all_results = {}
    for version in args.versions:
        if not GENERATORS[version]["ckpt"].exists():
            print(f"[SKIP] {version} — checkpoint not found")
            continue
        generator, _ = _load_generator(version, device)
        res = _run_cond_test(generator, train_loader, device, targets, version)
        all_results.update(res)

    # Final comparison table
    print(f"\n{'='*100}")
    print("CHARGE TARGETING SUMMARY")
    print(f"{'Condition':<35} {'tgt':>6}", end="")
    for tag in all_results:
        short = tag.replace("v2 cfg=0","v2").replace("v3 cfg=0","v3").replace("v3 cfg=5","v3+CFG")
        print(f"  {short:>10}", end="")
    print()
    print("-" * 100)
    for label, ln, tc, tg in targets:
        tc_raw = math.atanh(tc) * 5
        print(f"{label:<35} {tc_raw:>6.1f}", end="")
        for tag in all_results:
            r = all_results[tag][label]
            print(f"  {r['mean_charge']:>10.2f}", end="")
        print()

    print(f"\nGRAVY TARGETING SUMMARY")
    print(f"{'Condition':<35} {'tgt':>6}", end="")
    for tag in all_results:
        short = tag.replace("v2 cfg=0","v2").replace("v3 cfg=0","v3").replace("v3 cfg=5","v3+CFG")
        print(f"  {short:>10}", end="")
    print()
    print("-" * 100)
    for label, ln, tc, tg in targets:
        tg_raw = math.atanh(max(min(tg, 0.9999), -0.9999))
        print(f"{label:<35} {tg_raw:>6.2f}", end="")
        for tag in all_results:
            r = all_results[tag][label]
            print(f"  {r['mean_gravy']:>10.2f}", end="")
        print()

    out = PROJECT_ROOT / "eval_results/conditional_gen_v3_test.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
