"""Formal evaluation of physicochemical control in AMP generators.

The script writes:
  - metrics.json: aggregate controllability metrics by variant
  - predictions.jsonl: per generated sequence with target and observed properties
  - manifest.json: config/output linkage for paper evidence
  - SUMMARY.md: compact human-readable summary
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
VALID_AA = set(AA)
POSITIVE = set("KR")
NEGATIVE = set("DE")
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ids_to_seq(ids: list[int]) -> str:
    out = []
    for token_id in ids:
        if token_id in (0, 1):
            break
        if 2 <= token_id <= 21:
            out.append(AA[token_id - 2])
    return "".join(out)


def physchem(seq: str) -> dict[str, float]:
    valid = all(c in VALID_AA for c in seq)
    n = len(seq)
    charge = sum(1 if c in POSITIVE else -1 if c in NEGATIVE else 0 for c in seq)
    gravy = sum(KD_SCALE.get(c, 0.0) for c in seq) / max(n, 1)
    return {
        "length": float(n),
        "charge": float(charge),
        "gravy": float(gravy),
        "valid": float(valid),
    }


def condition_vector(target: dict[str, float], device: torch.device, num_conditions: int = 3) -> torch.Tensor:
    dims = [
        float(target["length"]) / 50.0,
        math.tanh(float(target["charge"]) / 5.0),
        math.tanh(float(target["gravy"])),
    ]
    if num_conditions >= 4:
        hc50 = float(target.get("hc50_log10", 2.3))
        dims.append(math.tanh(hc50 / 3.0))
    return torch.tensor(dims[:num_conditions], dtype=torch.float32, device=device)


def load_generator(spec: dict[str, Any], device: torch.device):
    from src.models.encoder import TransformerEncoder
    from src.models.generator import (
        ConditionalGenerator,
        ConditionalGeneratorV3,
        ConditionalGeneratorV4,
    )
    from src.models.generator_diffusion import ConditionalGeneratorDiffusion
    from src.models.generator_nar import ConditionalGeneratorNAR
    from src.models.jepa import JEPA

    ckpt = torch.load(resolve(spec["checkpoint"]), map_location=device, weights_only=False)
    pretrain_model_cfg = ckpt["pretrain_cfg"]["model"]
    gen_model_cfg = ckpt["cfg"]["generator"]

    pt_ckpt = torch.load(
        resolve("checkpoints/jepa_pretrain_868k/last_jepa.pt"),
        map_location=device,
        weights_only=False,
    )
    jepa = JEPA(**pretrain_model_cfg)
    jepa.load_state_dict(pt_ckpt["model_state"])

    encoder = TransformerEncoder(
        **{
            k: pretrain_model_cfg[k]
            for k in ["d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "max_seq_len"]
        }
    )
    encoder.load_state_dict(jepa.context_encoder.state_dict())

    implementation = spec["implementation"]
    if implementation == "v2":
        gen_cls = ConditionalGenerator
    elif implementation == "v3":
        gen_cls = ConditionalGeneratorV3
    elif implementation in {"v4", "v7"}:
        gen_cls = ConditionalGeneratorV4
    elif implementation == "nar":
        gen_cls = ConditionalGeneratorNAR
    elif implementation == "diffusion":
        gen_cls = ConditionalGeneratorDiffusion
    else:
        raise ValueError(f"Unknown generator implementation: {implementation}")

    generator = gen_cls(encoder=encoder, d_model=pretrain_model_cfg["d_model"], freeze_encoder=True, **gen_model_cfg)
    generator.load_state_dict(ckpt["model_state"])
    generator.to(device).eval()
    return generator


def load_context_loader(cfg: dict[str, Any]) -> tuple[DataLoader, set[str]]:
    from src.data.dataset import build_seq2seq_datasets

    with open(resolve(cfg["data"]["pretrain_config"])) as f:
        pretrain_cfg = yaml.safe_load(f)
    data_cfg = pretrain_cfg["data"]
    fasta_paths = [resolve(p) for p in data_cfg["fasta_paths"]]

    # Use a representative finetune config for prefix splitting; all compared
    # checkpoints were trained on the same sequence corpus family.
    with open(resolve("configs/finetune_868k_v3.yaml")) as f:
        ft_cfg = yaml.safe_load(f)

    train_ds, _ = build_seq2seq_datasets(
        fasta_paths=fasta_paths,
        max_len=data_cfg["max_len"],
        val_ratio=data_cfg["val_ratio"],
        seed=int(cfg.get("seed", 42)),
        prefix_ratio=ft_cfg["data"].get("prefix_ratio", 0.5),
        min_prefix_len=ft_cfg["data"].get("min_prefix_len", 3),
        max_seq_len=ft_cfg["generator"]["max_seq_len"],
    )
    max_batches = int(cfg["data"].get("max_context_batches", 0))
    batch_size = int(cfg["data"].get("batch_size", 64))
    max_items = len(train_ds) if max_batches <= 0 else min(len(train_ds), max_batches * batch_size)
    subset = Subset(train_ds, list(range(max_items)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    train_sequences = set(getattr(train_ds, "sequences", []))
    return loader, train_sequences


def generate_sequences(
    generator,
    loader: DataLoader,
    target: dict[str, float],
    variant: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
) -> list[str]:
    n_required = int(cfg["generation"]["n_per_condition"])
    max_new_tokens = int(cfg["generation"].get("max_new_tokens", 50))
    temperature = float(cfg["generation"].get("temperature", 0.9))
    top_p = float(cfg["generation"].get("top_p", 0.9))
    cfg_scale = float(variant.get("cfg_scale", 0.0))

    num_conditions = int(variant.get("num_conditions", 3))
    cond_single = condition_vector(target, device, num_conditions=num_conditions)
    seqs: list[str] = []
    while len(seqs) < n_required:
        for batch in loader:
            context_ids = batch["context_ids"].to(device)
            cond = cond_single.unsqueeze(0).expand(context_ids.shape[0], -1)
            with torch.no_grad():
                if variant["implementation"] == "v2":
                    out = generator.generate(
                        context_ids,
                        conditions=cond,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                elif variant["implementation"] in {"v3", "v4", "v7"}:
                    out = generator.generate(
                        context_ids,
                        conditions=cond,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        cfg_scale=cfg_scale,
                    )
                elif variant["implementation"] == "nar":
                    out = generator.generate(
                        context_ids,
                        conditions=cond,
                        temperature=temperature,
                    )
                elif variant["implementation"] == "diffusion":
                    out = generator.generate(
                        context_ids,
                        conditions=cond,
                        seq_len=int(target["length"]),
                        temperature=temperature,
                    )
                else:
                    raise ValueError(f"Unknown generator implementation: {variant['implementation']}")
            for row in out:
                seq = row if isinstance(row, str) else ids_to_seq(row.tolist())
                if len(seq) >= 1:
                    seqs.append(seq)
                    if len(seqs) >= n_required:
                        break
            if len(seqs) >= n_required:
                break
    return seqs[:n_required]


def r2_score(targets: np.ndarray, values: np.ndarray) -> float:
    denom = float(np.sum((values - values.mean()) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((values - targets) ** 2) / denom)


def summarize_variant(rows: list[dict[str, Any]], train_sequences: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_property": {}, "by_target": {}}
    props_to_eval = ["charge", "gravy", "length"]
    if any("target_hc50_log10" in r and "actual_hc50_log10" in r for r in rows):
        props_to_eval.append("hc50_log10")
    for prop in props_to_eval:
        target_vals = np.array([r[f"target_{prop}"] for r in rows if f"actual_{prop}" in r], dtype=float)
        actual_vals = np.array([r[f"actual_{prop}"] for r in rows if f"actual_{prop}" in r], dtype=float)
        out["by_property"][prop] = {
            "mae": float(np.mean(np.abs(actual_vals - target_vals))),
            "bias": float(np.mean(actual_vals - target_vals)),
            "r2_target_actual": r2_score(target_vals, actual_vals),
        }

    seqs = [r["sequence"] for r in rows]
    out["n_sequences"] = len(seqs)
    out["unique_fraction"] = float(len(set(seqs)) / max(len(seqs), 1))
    out["exact_novelty_fraction"] = float(np.mean([s not in train_sequences for s in seqs]))
    out["valid_fraction"] = float(np.mean([r["valid"] for r in rows]))
    out["short_fraction_len_lt_5"] = float(np.mean([len(s) < 5 for s in seqs]))

    for target_key in sorted({r["target_key"] for r in rows}):
        sub = [r for r in rows if r["target_key"] == target_key]
        entry = {
            "n": len(sub),
            "mean_charge": float(np.mean([r["actual_charge"] for r in sub])),
            "mean_gravy": float(np.mean([r["actual_gravy"] for r in sub])),
            "mean_length": float(np.mean([r["actual_length"] for r in sub])),
            "target_charge": float(sub[0]["target_charge"]),
            "target_gravy": float(sub[0]["target_gravy"]),
            "target_length": float(sub[0]["target_length"]),
        }
        hc50_actuals = [r["actual_hc50_log10"] for r in sub if "actual_hc50_log10" in r]
        if hc50_actuals:
            entry["mean_hc50_log10"] = float(np.mean(hc50_actuals))
        if "target_hc50_log10" in sub[0]:
            entry["target_hc50_log10"] = float(sub[0]["target_hc50_log10"])
        out["by_target"][target_key] = entry
    return out


def write_summary(out_dir: Path, metrics: dict[str, Any]) -> None:
    has_hc50 = any("hc50_log10" in v["by_property"] for v in metrics["variants"].values())
    header = "| Variant | Charge R² | Charge MAE | GRAVY R² | GRAVY MAE | Length R² | Length MAE"
    sep    = "|---|---:|---:|---:|---:|---:|---:"
    if has_hc50:
        header += " | HC50 R² | HC50 MAE"
        sep    += "|---:|---:"
    header += " | Novelty | Unique |"
    sep    += "|---:|---:|"

    lines = ["# Generation Control Summary", "", header, sep]
    for key, vals in metrics["variants"].items():
        props = vals["by_property"]

        def _fmt(p, metric):
            v = props.get(p, {}).get(metric, float("nan"))
            return f"{v:.3f}" if v == v else "—"

        row = (f"| {key} "
               f"| {_fmt('charge','r2_target_actual')} | {_fmt('charge','mae')} "
               f"| {_fmt('gravy','r2_target_actual')} | {_fmt('gravy','mae')} "
               f"| {_fmt('length','r2_target_actual')} | {_fmt('length','mae')}")
        if has_hc50:
            row += f" | {_fmt('hc50_log10','r2_target_actual')} | {_fmt('hc50_log10','mae')}"
        row += f" | {vals['exact_novelty_fraction']:.3f} | {vals['unique_fraction']:.3f} |"
        lines.append(row)

    lines += [
        "",
        "R² = requested-vs-achieved correlation across all targets for that property.",
        "R² ≈ 1 = good control, R² ≈ 0 = uncontrolled, R² < 0 = anticorrelated.",
        "Note: other papers do not report this metric — direct comparison is not possible without their raw sequences.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(resolve(args.config)) as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))
    gpu = int(cfg.get("gpu", 0))
    use_cuda = cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu}" if use_cuda else "cpu")

    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve(args.config), out_dir / "config_resolved.yaml")

    loader, train_sequences = load_context_loader(cfg)
    all_rows: list[dict[str, Any]] = []
    variant_metrics: dict[str, Any] = {}

    for variant_idx, variant in enumerate(cfg["variants"]):
        variant_key = variant["key"]
        print(f"Loading {variant_key} from {variant['checkpoint']}")
        generator = load_generator(variant, device)
        variant_rows: list[dict[str, Any]] = []

        # Load HC50 oracle if configured and variant uses HC50 (4-dim conditions)
        hc50_oracle = None
        if cfg.get("hc50_oracle_ckpt") and int(variant.get("num_conditions", 3)) >= 4:
            try:
                from src.models.pretrain_utils import load_pretrained_encoder
                import torch.nn as nn
                oracle_ckpt_path = resolve(cfg["hc50_oracle_ckpt"])
                oracle_ckpt = torch.load(oracle_ckpt_path, map_location=device, weights_only=False)
                oracle_args = oracle_ckpt.get("args", {})
                pt_path = oracle_args.get("checkpoint", "checkpoints/jepa_pretrain_868k/last_jepa.pt")
                enc, pt_cfg = load_pretrained_encoder(str(resolve(pt_path)), device)
                d_model = pt_cfg["model"]["d_model"]

                class _HC50Oracle(nn.Module):
                    def __init__(self, encoder, d_model, hidden, dropout):
                        super().__init__()
                        self.encoder = encoder
                        self.head = nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
                            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                            nn.Linear(hidden, 1),
                        )
                    def forward(self, input_ids, lengths):
                        h = self.encoder(input_ids)
                        pooled = torch.stack([h[i, 1:lengths[i]-1].mean(0) for i in range(len(lengths))])
                        return self.head(pooled).squeeze(-1)

                hc50_oracle = _HC50Oracle(enc, d_model,
                                          hidden=oracle_args.get("hidden", 512),
                                          dropout=oracle_args.get("dropout", 0.25)).to(device)
                hc50_oracle.load_state_dict(oracle_ckpt["model_state"])
                hc50_oracle.eval()
                for p in hc50_oracle.parameters():
                    p.requires_grad_(False)
                print(f"  [HC50 oracle loaded]")
            except Exception as e:
                print(f"  [HC50 oracle load failed: {e}]")
                hc50_oracle = None

        for target_idx, target in enumerate(cfg["targets"]):
            # Deterministic but distinct sampling stream per variant/target.
            set_seed(int(cfg.get("seed", 42)) + 1000 * variant_idx + target_idx)
            seqs = generate_sequences(generator, loader, target, variant, cfg, device)

            # Batch-score with HC50 oracle if available
            hc50_preds: list[float] = []
            if hc50_oracle is not None:
                from src.data.tokenizer import encode
                batch_ids, batch_lens = [], []
                for seq in seqs:
                    ids = torch.tensor(encode(seq[:50], add_special_tokens=True), dtype=torch.long)
                    batch_ids.append(ids)
                    batch_lens.append(len(ids))
                max_len = max(batch_lens)
                padded = torch.zeros(len(batch_ids), max_len, dtype=torch.long, device=device)
                for i, ids in enumerate(batch_ids):
                    padded[i, :len(ids)] = ids.to(device)
                lengths_t = torch.tensor(batch_lens, dtype=torch.long, device=device)
                with torch.no_grad():
                    hc50_preds = hc50_oracle(padded, lengths_t).cpu().float().tolist()

            for seq_idx, seq in enumerate(seqs):
                pc = physchem(seq)
                row = {
                    "variant": variant_key,
                    "implementation": variant["implementation"],
                    "checkpoint": str(resolve(variant["checkpoint"])),
                    "cfg_scale": float(variant.get("cfg_scale", 0.0)),
                    "target_key": target["key"],
                    "sample_index": seq_idx,
                    "sequence": seq,
                    "target_charge": float(target["charge"]),
                    "target_gravy": float(target["gravy"]),
                    "target_length": float(target["length"]),
                    "actual_charge": pc["charge"],
                    "actual_gravy": pc["gravy"],
                    "actual_length": pc["length"],
                    "valid": pc["valid"],
                    "exact_novel": float(seq not in train_sequences),
                }
                if "hc50_log10" in target:
                    row["target_hc50_log10"] = float(target["hc50_log10"])
                if hc50_preds:
                    row["actual_hc50_log10"] = float(hc50_preds[seq_idx])
                variant_rows.append(row)
                all_rows.append(row)
            print(f"  {target['key']}: generated {len(seqs)}")

        variant_metrics[variant_key] = summarize_variant(variant_rows, train_sequences)
        del generator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(out_dir / "predictions.jsonl", "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    metrics = {
        "experiment_id": cfg["experiment_id"],
        "research_decision": cfg["research_decision"],
        "n_rows": len(all_rows),
        "variants": variant_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    manifest = {
        "experiment_id": cfg["experiment_id"],
        "config": str(out_dir / "config_resolved.yaml"),
        "metrics": str(out_dir / "metrics.json"),
        "predictions": str(out_dir / "predictions.jsonl"),
        "summary": str(out_dir / "SUMMARY.md"),
        "n_rows": len(all_rows),
        "status": "formal_artifact" if "formal" in cfg["experiment_id"] else "smoke_artifact",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    write_summary(out_dir, metrics)
    print(f"Saved generation-control artifacts to {out_dir}")


if __name__ == "__main__":
    main()
