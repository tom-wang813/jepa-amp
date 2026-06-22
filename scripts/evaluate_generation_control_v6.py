"""Formal evaluation for the 7-dim v6 conditional generator.

Outputs:
  - metrics.json: aggregate control metrics by property
  - predictions.jsonl: per-sequence realized properties
  - manifest.json: config/output linkage
  - SUMMARY.md: compact reader-facing table
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
from Bio.SeqUtils.ProtParam import ProteinAnalysis
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
_CF_HELIX = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
_PKA = {"D": 3.9, "E": 4.1, "H": 6.0, "C": 8.3, "Y": 10.1, "K": 10.5, "R": 12.5}
_PKA_NT = 8.0
_PKA_CT = 3.1
_HELIX_DELTA = 100.0 * math.pi / 180.0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _net_charge_at_ph(seq: str, ph: float) -> float:
    charge = 1.0 / (1.0 + 10 ** (ph - _PKA_NT))
    charge -= 1.0 / (1.0 + 10 ** (_PKA_CT - ph))
    for aa in seq:
        pk = _PKA.get(aa)
        if pk is None:
            continue
        if aa in ("D", "E", "C", "Y"):
            charge -= 1.0 / (1.0 + 10 ** (pk - ph))
        else:
            charge += 1.0 / (1.0 + 10 ** (ph - pk))
    return charge


def compute_pi(seq: str) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if _net_charge_at_ph(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compute_hydrophobic_moment(seq: str) -> float:
    sin_sum = sum(KD_SCALE.get(aa, 0.0) * math.sin(i * _HELIX_DELTA) for i, aa in enumerate(seq))
    cos_sum = sum(KD_SCALE.get(aa, 0.0) * math.cos(i * _HELIX_DELTA) for i, aa in enumerate(seq))
    return math.sqrt(sin_sum ** 2 + cos_sum ** 2) / max(len(seq), 1)


def compute_helix_propensity(seq: str) -> float:
    return sum(_CF_HELIX.get(aa, 1.0) for aa in seq) / max(len(seq), 1)


def physchem(seq: str) -> dict[str, float]:
    valid = all(c in VALID_AA for c in seq)
    n = len(seq)
    charge = sum(1 if c in POSITIVE else -1 if c in NEGATIVE else 0 for c in seq)
    gravy = sum(KD_SCALE.get(c, 0.0) for c in seq) / max(n, 1)
    helix = compute_helix_propensity(seq)
    pI = compute_pi(seq)
    hm = compute_hydrophobic_moment(seq)
    return {
        "length": float(n),
        "charge": float(charge),
        "gravy": float(gravy),
        "helix": float(helix),
        "pI": float(pI),
        "hydrophobic_moment": float(hm),
        "valid": float(valid),
    }


def condition_vector_v6(target: dict[str, float], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            float(target["length"]) / 50.0,
            math.tanh(float(target["charge"]) / 5.0),
            math.tanh(float(target["gravy"])),
            (float(target["helix"]) - 1.0) / 0.5,
            float(target["pI"]) / 14.0,
            math.tanh(float(target["hydrophobic_moment"]) * 4.0),
            float(target["amp_score"]),
        ],
        dtype=torch.float32,
        device=device,
    )


def load_generator(spec: dict[str, Any], device: torch.device):
    from src.models.encoder import TransformerEncoder
    from src.models.generator import ConditionalGeneratorV4
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

    generator = ConditionalGeneratorV4(
        encoder=encoder,
        d_model=pretrain_model_cfg["d_model"],
        freeze_encoder=True,
        **gen_model_cfg,
    )
    generator.load_state_dict(ckpt["model_state"])
    generator.to(device).eval()
    return generator


def load_context_loader(cfg: dict[str, Any]) -> tuple[DataLoader, set[str]]:
    from src.data.dataset import build_seq2seq_datasets

    with open(resolve(cfg["data"]["pretrain_config"])) as f:
        pretrain_cfg = yaml.safe_load(f)
    data_cfg = pretrain_cfg["data"]
    fasta_paths = [resolve(p) for p in data_cfg["fasta_paths"]]

    with open(resolve(cfg["data"]["reference_finetune_config"])) as f:
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


def load_amp_classifier(device: torch.device):
    from scripts.precompute_amp_scores import load_classifier

    return load_classifier(device)


@torch.no_grad()
def score_amp_sequences(model, sequences: list[str], device: torch.device, batch_size: int = 512) -> dict[str, float]:
    from scripts.precompute_amp_scores import score_sequences

    return score_sequences(model, sequences, device, batch_size=batch_size)


def generate_sequences(
    generator,
    loader: DataLoader,
    target: dict[str, float],
    cfg: dict[str, Any],
    device: torch.device,
) -> list[str]:
    n_required = int(cfg["generation"]["n_per_condition"])
    max_new_tokens = int(cfg["generation"].get("max_new_tokens", 50))
    temperature = float(cfg["generation"].get("temperature", 0.9))
    top_p = float(cfg["generation"].get("top_p", 0.9))
    cfg_scale = float(cfg["generation"].get("cfg_scale", 0.0))

    cond_single = condition_vector_v6(target, device)
    seqs: list[str] = []
    while len(seqs) < n_required:
        for batch in loader:
            context_ids = batch["context_ids"].to(device)
            cond = cond_single.unsqueeze(0).expand(context_ids.shape[0], -1)
            with torch.no_grad():
                out = generator.generate(
                    context_ids,
                    conditions=cond,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_scale=cfg_scale,
                )
            for row in out:
                seq = "".join(AA[token_id - 2] for token_id in row.tolist() if 2 <= token_id <= 21)
                if len(seq) >= 1:
                    seqs.append(seq)
                    if len(seqs) >= n_required:
                        break
            if len(seqs) >= n_required:
                break
    return seqs[:n_required]


def r2_score(targets: np.ndarray, values: np.ndarray) -> float:
    denom = float(np.sum((targets - targets.mean()) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((values - targets) ** 2) / denom)


def summarize(rows: list[dict[str, Any]], train_sequences: set[str]) -> dict[str, Any]:
    props = ["charge", "gravy", "length", "helix", "pI", "hydrophobic_moment", "amp_score"]
    out: dict[str, Any] = {"by_property": {}, "by_target": {}}
    for prop in props:
        target_vals = np.array([r[f"target_{prop}"] for r in rows], dtype=float)
        actual_vals = np.array([r[f"actual_{prop}"] for r in rows], dtype=float)
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
    for target_key in sorted({r["target_key"] for r in rows}):
        sub = [r for r in rows if r["target_key"] == target_key]
        out["by_target"][target_key] = {
            "n": len(sub),
            "mean_charge": float(np.mean([r["actual_charge"] for r in sub])),
            "mean_helix": float(np.mean([r["actual_helix"] for r in sub])),
            "mean_pI": float(np.mean([r["actual_pI"] for r in sub])),
            "mean_hydrophobic_moment": float(np.mean([r["actual_hydrophobic_moment"] for r in sub])),
            "mean_amp_score": float(np.mean([r["actual_amp_score"] for r in sub])),
        }
    return out


def write_summary(out_dir: Path, metrics: dict[str, Any]) -> None:
    vals = metrics["variant"]["by_property"]
    lines = ["# V6 Generation Control Summary", ""]
    lines.append("| Charge R2 | Charge MAE | Helix R2 | Helix MAE | pI R2 | pI MAE | HM R2 | HM MAE | AMP R2 | AMP MAE | Unique | Novelty |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| {c_r2:.3f} | {c_mae:.3f} | {h_r2:.3f} | {h_mae:.3f} | {pi_r2:.3f} | {pi_mae:.3f} | {hm_r2:.3f} | {hm_mae:.3f} | {a_r2:.3f} | {a_mae:.3f} | {uniq:.3f} | {nov:.3f} |".format(
            c_r2=vals["charge"]["r2_target_actual"],
            c_mae=vals["charge"]["mae"],
            h_r2=vals["helix"]["r2_target_actual"],
            h_mae=vals["helix"]["mae"],
            pi_r2=vals["pI"]["r2_target_actual"],
            pi_mae=vals["pI"]["mae"],
            hm_r2=vals["hydrophobic_moment"]["r2_target_actual"],
            hm_mae=vals["hydrophobic_moment"]["mae"],
            a_r2=vals["amp_score"]["r2_target_actual"],
            a_mae=vals["amp_score"]["mae"],
            uniq=metrics["variant"]["unique_fraction"],
            nov=metrics["variant"]["exact_novelty_fraction"],
        )
    )
    lines.append("")
    lines.append("Interpretation: v6 is only paper-usable if added control dimensions improve beyond charge alone without collapsing uniqueness.")
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
    generator = load_generator(cfg["variant"], device)
    amp_classifier = load_amp_classifier(device)

    rows: list[dict[str, Any]] = []
    for target_idx, target in enumerate(cfg["targets"]):
        set_seed(int(cfg.get("seed", 42)) + target_idx)
        seqs = generate_sequences(generator, loader, target, cfg, device)
        amp_scores = score_amp_sequences(amp_classifier, seqs, device, batch_size=int(cfg["generation"].get("amp_batch_size", 512)))
        for seq_idx, seq in enumerate(seqs):
            pc = physchem(seq)
            rows.append(
                {
                    "target_key": target["key"],
                    "sample_index": seq_idx,
                    "sequence": seq,
                    "valid": pc["valid"],
                    "target_charge": float(target["charge"]),
                    "target_gravy": float(target["gravy"]),
                    "target_length": float(target["length"]),
                    "target_helix": float(target["helix"]),
                    "target_pI": float(target["pI"]),
                    "target_hydrophobic_moment": float(target["hydrophobic_moment"]),
                    "target_amp_score": float(target["amp_score"]),
                    "actual_charge": pc["charge"],
                    "actual_gravy": pc["gravy"],
                    "actual_length": pc["length"],
                    "actual_helix": pc["helix"],
                    "actual_pI": pc["pI"],
                    "actual_hydrophobic_moment": pc["hydrophobic_moment"],
                    "actual_amp_score": amp_scores[seq],
                    "exact_novel": float(seq not in train_sequences),
                }
            )
        print(f"  {target['key']}: generated {len(seqs)}")

    with open(out_dir / "predictions.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    variant_metrics = summarize(rows, train_sequences)
    metrics = {
        "experiment_id": cfg["experiment_id"],
        "research_decision": cfg["research_decision"],
        "n_rows": len(rows),
        "variant": variant_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    manifest = {
        "experiment_id": cfg["experiment_id"],
        "config": str(out_dir / "config_resolved.yaml"),
        "metrics": str(out_dir / "metrics.json"),
        "predictions": str(out_dir / "predictions.jsonl"),
        "summary": str(out_dir / "SUMMARY.md"),
        "n_rows": len(rows),
        "status": "formal_artifact",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    write_summary(out_dir, metrics)
    print(f"Saved v6 generation-control artifacts to {out_dir}")


if __name__ == "__main__":
    main()
