"""Formal evaluation of MIC-conditioned generation.

This entrypoint separates two questions that were previously mixed:
  1. Does the generator move a JEPA MIC oracle in the requested direction?
  2. Does the same movement remain visible under an independently trained ESM2 MIC scorer?

Outputs:
  - metrics.json
  - predictions.jsonl
  - manifest.json
  - SUMMARY.md
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
    n = len(seq)
    charge = sum(1 if c in POSITIVE else -1 if c in NEGATIVE else 0 for c in seq)
    gravy = sum(KD_SCALE.get(c, 0.0) for c in seq) / max(n, 1)
    return {
        "length": float(n),
        "charge": float(charge),
        "gravy": float(gravy),
        "valid": float(all(c in VALID_AA for c in seq)),
    }


def mic_condition(physchem_vec: list[float], targets: dict[Any, Any], n_bacteria: int) -> torch.Tensor:
    mic_vals = torch.zeros(n_bacteria, dtype=torch.float32)
    mic_mask = torch.zeros(n_bacteria, dtype=torch.float32)
    for key, value in targets.items():
        idx = int(key)
        mic_vals[idx] = float(value)
        mic_mask[idx] = 1.0
    return torch.cat([torch.tensor(physchem_vec, dtype=torch.float32), mic_vals, mic_mask])


def load_context_loader(cfg: dict[str, Any]) -> tuple[DataLoader, set[str]]:
    gen_cfg_path = cfg["generator"]["grampa_config"]
    if not resolve(cfg["generator"]["checkpoint"]).exists():
        gen_cfg_path = cfg["generator"]["fallback_config"]
    with open(resolve(gen_cfg_path)) as f:
        gen_cfg = yaml.safe_load(f)
    with open(resolve(cfg["data"]["pretrain_config"])) as f:
        pretrain_cfg = yaml.safe_load(f)

    pre_data = pretrain_cfg["data"]
    gen_data = gen_cfg["data"]
    ds_kwargs = dict(
        max_len=pre_data["max_len"],
        val_ratio=pre_data["val_ratio"],
        seed=int(cfg.get("seed", 42)),
        prefix_ratio=gen_data.get("prefix_ratio", 0.5),
        min_prefix_len=gen_data.get("min_prefix_len", 3),
        max_seq_len=gen_cfg["generator"]["max_seq_len"],
    )

    if gen_cfg.get("generator_version") == "grampa_v5":
        from src.data.dataset import build_seq2seq_datasets_grampa_v5

        train_ds, _ = build_seq2seq_datasets_grampa_v5(
            grampa_csv=resolve(gen_data["grampa_csv"]),
            n_repeats=1,
            **ds_kwargs,
        )
    else:
        from src.data.dataset import build_seq2seq_datasets_v5

        train_ds, _ = build_seq2seq_datasets_v5(
            fasta_paths=[resolve(p) for p in pre_data["fasta_paths"]],
            mic_pseudolabel_npy=resolve(gen_data["mic_pseudolabel_npy"]),
            mic_pseudolabel_seqs=resolve(gen_data["mic_pseudolabel_seqs"]),
            mic_mask_prob=0.0,
            **ds_kwargs,
        )

    batch_size = int(cfg["data"].get("context_batch_size", 64))
    max_batches = int(cfg["data"].get("max_context_batches", 0))
    max_items = len(train_ds) if max_batches <= 0 else min(len(train_ds), max_batches * batch_size)
    subset = Subset(train_ds, list(range(max_items)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    train_sequences = set(getattr(train_ds, "sequences", []))
    return loader, train_sequences


def load_generator(cfg: dict[str, Any], device: torch.device):
    from src.models.encoder import TransformerEncoder
    from src.models.generator import ConditionalGeneratorV5
    from src.models.jepa import JEPA

    ckpt_path = resolve(cfg["generator"]["checkpoint"])
    if not ckpt_path.exists():
        ckpt_path = resolve(cfg["generator"]["fallback_checkpoint"])
    gen_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pretrain_model_cfg = gen_ckpt["pretrain_cfg"]["model"]
    gen_model_cfg = gen_ckpt["cfg"]["generator"]

    pt_ckpt = torch.load(resolve(cfg["generator"]["pretrain_checkpoint"]), map_location=device, weights_only=False)
    jepa = JEPA(**pretrain_model_cfg)
    jepa.load_state_dict(pt_ckpt["model_state"])

    encoder = TransformerEncoder(
        **{
            k: pretrain_model_cfg[k]
            for k in ["d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "max_seq_len"]
        }
    )
    encoder.load_state_dict(jepa.context_encoder.state_dict())

    gen = ConditionalGeneratorV5(encoder=encoder, d_model=pretrain_model_cfg["d_model"], freeze_encoder=True, **gen_model_cfg)
    gen.load_state_dict(gen_ckpt["model_state"])
    gen.to(device).eval()
    return gen, ckpt_path


def load_jepa_scorer(spec: dict[str, str], device: torch.device):
    from src.data.supervised_dataset import N_BACTERIA
    from src.models.jepa import JEPA
    from src.models.supervised_head import JEPAMICPredictor

    with open(resolve(spec["config"])) as f:
        cfg = yaml.safe_load(f)
    pt_ckpt = torch.load(resolve(cfg["pretrain_checkpoint"]), map_location=device, weights_only=False)
    jepa = JEPA(**pt_ckpt["cfg"]["model"])
    jepa.load_state_dict(pt_ckpt["model_state"])

    head_cfg = cfg["head"].copy()
    head_type = head_cfg.pop("head_type", "transformer")
    model = JEPAMICPredictor(
        encoder=jepa.context_encoder,
        d_model=pt_ckpt["cfg"]["model"]["d_model"],
        n_bacteria=N_BACTERIA,
        head_type=head_type,
        freeze_encoder=cfg["train"]["freeze_encoder"],
        **head_cfg,
    ).to(device)
    ckpt = torch.load(resolve(spec["checkpoint"]), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_esm_scorer(spec: dict[str, str], device: torch.device):
    from src.data.supervised_dataset import N_BACTERIA
    from src.models.esm_head import ESMMICPredictor

    with open(resolve(spec["config"])) as f:
        cfg = yaml.safe_load(f)
    model_key = cfg.get("esm_model", "esm2_t12_35M")
    head_cfg = cfg["head"].copy()
    model = ESMMICPredictor(
        model_key=model_key,
        n_bacteria=N_BACTERIA,
        freeze_encoder=cfg["train"]["freeze_encoder"],
        **head_cfg,
    ).to(device)
    ckpt = torch.load(resolve(spec["checkpoint"]), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, model.encoder.alphabet.get_batch_converter()


@torch.no_grad()
def generate_sequences(generator, loader: DataLoader, cond: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> list[str]:
    n_required = int(cfg["generation"]["n_per_scenario"])
    seqs: list[str] = []
    cond = cond.to(device)
    while len(seqs) < n_required:
        for batch in loader:
            context_ids = batch["context_ids"].to(device)
            cond_batch = cond.unsqueeze(0).expand(context_ids.shape[0], -1)
            out = generator.generate(
                context_ids,
                conditions=cond_batch,
                max_new_tokens=int(cfg["generation"].get("max_new_tokens", 50)),
                temperature=float(cfg["generation"].get("temperature", 0.9)),
                top_p=float(cfg["generation"].get("top_p", 0.9)),
                cfg_scale=float(cfg["generation"].get("cfg_scale", 0.0)),
            )
            for row in out:
                seq = ids_to_seq(row.tolist())
                if len(seq) >= 3:
                    seqs.append(seq)
                    if len(seqs) >= n_required:
                        break
            if len(seqs) >= n_required:
                break
    return seqs[:n_required]


@torch.no_grad()
def score_jepa(model, seqs: list[str], bacteria_indices: list[int], batch_size: int, device: torch.device) -> dict[int, list[float]]:
    from src.data.tokenizer import PAD_ID

    results = {idx: [] for idx in bacteria_indices}
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        encoded = []
        for seq in batch:
            ids = [0] + [2 + AA.index(c) for c in seq[:46] if c in VALID_AA] + [1]
            encoded.append(torch.tensor(ids, dtype=torch.long))
        max_len = max(x.shape[0] for x in encoded)
        tokens = torch.full((len(encoded), max_len), PAD_ID, dtype=torch.long)
        for i, row in enumerate(encoded):
            tokens[i, : row.shape[0]] = row
        tokens = tokens.to(device)
        for idx in bacteria_indices:
            bidx = torch.full((len(batch),), idx, dtype=torch.long, device=device)
            preds = model(tokens, bidx).detach().cpu().float().tolist()
            results[idx].extend(preds)
    return results


@torch.no_grad()
def score_esm(model, batch_converter, seqs: list[str], bacteria_indices: list[int], batch_size: int, device: torch.device) -> dict[int, list[float]]:
    results = {idx: [] for idx in bacteria_indices}
    for start in range(0, len(seqs), batch_size):
        batch = [s[:48] for s in seqs[start:start + batch_size]]
        _, _, tokens = batch_converter([(f"s{i}", seq) for i, seq in enumerate(batch)])
        tokens = tokens.to(device)
        for idx in bacteria_indices:
            bidx = torch.full((len(batch),), idx, dtype=torch.long, device=device)
            preds = model(tokens, bidx).detach().cpu().float().tolist()
            results[idx].extend(preds)
    return results


def summarize(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    bacteria_names = {0: "E.coli", 1: "S.aureus", 2: "P.aeru", 3: "C.albicans"}
    by_scenario: dict[str, Any] = {}
    by_key = {s["key"]: s for s in cfg["scenarios"]}

    for scenario in cfg["scenarios"]:
        key = scenario["key"]
        sub = [r for r in rows if r["scenario_key"] == key]
        by_scenario[key] = {
            "label": scenario["label"],
            "n": len(sub),
            "unique_fraction": float(len({r["sequence"] for r in sub}) / max(len(sub), 1)),
            "valid_fraction": float(np.mean([r["valid"] for r in sub])),
            "mean_charge": float(np.mean([r["charge"] for r in sub])),
            "mean_gravy": float(np.mean([r["gravy"] for r in sub])),
            "mean_length": float(np.mean([r["length"] for r in sub])),
            "mic_targets": {str(k): float(v) for k, v in scenario.get("mic_targets", {}).items()},
            "scorers": {},
        }
        for scorer in ["jepa_oracle", "esm2_independent"]:
            scorer_summary = {}
            for idx in cfg["evaluation"]["bacteria_indices"]:
                vals = np.array([r[f"{scorer}_b{idx}"] for r in sub], dtype=float)
                scorer_summary[bacteria_names.get(idx, str(idx))] = {
                    "mean_log2_mic": float(vals.mean()),
                    "std_log2_mic": float(vals.std()),
                }
            by_scenario[key]["scorers"][scorer] = scorer_summary

    for key, scenario in by_key.items():
        control_key = scenario.get("control_key")
        if not control_key:
            continue
        for scorer in ["jepa_oracle", "esm2_independent"]:
            for name in bacteria_names.values():
                cur = by_scenario[key]["scorers"][scorer].get(name)
                ctrl = by_scenario[control_key]["scorers"][scorer].get(name)
                if cur and ctrl:
                    cur["delta_vs_control"] = float(cur["mean_log2_mic"] - ctrl["mean_log2_mic"])

    return {
        "experiment_id": cfg["experiment_id"],
        "research_decision": cfg["research_decision"],
        "n_rows": len(rows),
        "by_scenario": by_scenario,
    }


def write_summary(out_dir: Path, metrics: dict[str, Any]) -> None:
    lines = ["# MIC-Conditioned Generation Summary", ""]
    lines.append("| Scenario | JEPA E.coli | JEPA Δ | ESM2 E.coli | ESM2 Δ | JEPA S.aureus | ESM2 S.aureus | Charge | Unique |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, vals in metrics["by_scenario"].items():
        jo_ec = vals["scorers"]["jepa_oracle"]["E.coli"]
        es_ec = vals["scorers"]["esm2_independent"]["E.coli"]
        jo_sa = vals["scorers"]["jepa_oracle"]["S.aureus"]
        es_sa = vals["scorers"]["esm2_independent"]["S.aureus"]
        lines.append(
            "| {key} | {jo:.3f} | {jod:+.3f} | {es:.3f} | {esd:+.3f} | {js:.3f} | {ess:.3f} | {ch:.2f} | {uniq:.3f} |".format(
                key=key,
                jo=jo_ec["mean_log2_mic"],
                jod=jo_ec.get("delta_vs_control", 0.0),
                es=es_ec["mean_log2_mic"],
                esd=es_ec.get("delta_vs_control", 0.0),
                js=jo_sa["mean_log2_mic"],
                ess=es_sa["mean_log2_mic"],
                ch=vals["mean_charge"],
                uniq=vals["unique_fraction"],
            )
        )
    lines.append("")
    lines.append("Interpretation: negative deltas indicate lower predicted MIC than the matched control. JEPA-oracle deltas support conditioning only as model-predicted evidence; ESM2 deltas are the stricter cross-model check.")
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
    generator, generator_checkpoint = load_generator(cfg, device)
    jepa_scorer = load_jepa_scorer(cfg["scorers"]["jepa_oracle"], device)
    esm_scorer, esm_converter = load_esm_scorer(cfg["scorers"]["esm2_independent"], device)

    from src.data.supervised_dataset import N_BACTERIA

    rows: list[dict[str, Any]] = []
    bacteria_indices = [int(x) for x in cfg["evaluation"]["bacteria_indices"]]
    batch_size = int(cfg["evaluation"].get("batch_size", 128))
    for scenario_idx, scenario in enumerate(cfg["scenarios"]):
        set_seed(int(cfg.get("seed", 42)) + scenario_idx)
        cond = mic_condition(scenario["physchem"], scenario.get("mic_targets", {}), N_BACTERIA)
        seqs = generate_sequences(generator, loader, cond, cfg, device)
        jepa_scores = score_jepa(jepa_scorer, seqs, bacteria_indices, batch_size, device)
        esm_scores = score_esm(esm_scorer, esm_converter, seqs, bacteria_indices, batch_size, device)
        for sample_idx, seq in enumerate(seqs):
            pc = physchem(seq)
            row: dict[str, Any] = {
                "scenario_key": scenario["key"],
                "scenario_label": scenario["label"],
                "sample_index": sample_idx,
                "sequence": seq,
                "exact_novel": float(seq not in train_sequences),
                **pc,
            }
            for idx in bacteria_indices:
                row[f"jepa_oracle_b{idx}"] = float(jepa_scores[idx][sample_idx])
                row[f"esm2_independent_b{idx}"] = float(esm_scores[idx][sample_idx])
            rows.append(row)
        print(f"{scenario['key']}: generated and scored {len(seqs)}")

    with open(out_dir / "predictions.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    metrics = summarize(rows, cfg)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    manifest = {
        "experiment_id": cfg["experiment_id"],
        "config": str(out_dir / "config_resolved.yaml"),
        "generator_checkpoint": str(generator_checkpoint),
        "jepa_oracle_checkpoint": str(resolve(cfg["scorers"]["jepa_oracle"]["checkpoint"])),
        "esm2_independent_checkpoint": str(resolve(cfg["scorers"]["esm2_independent"]["checkpoint"])),
        "metrics": str(out_dir / "metrics.json"),
        "predictions": str(out_dir / "predictions.jsonl"),
        "summary": str(out_dir / "SUMMARY.md"),
        "n_rows": len(rows),
        "status": "formal_artifact" if "formal" in cfg["experiment_id"] else "smoke_artifact",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    write_summary(out_dir, metrics)
    print(f"Saved MIC-conditioned generation artifacts to {out_dir}")


if __name__ == "__main__":
    main()
