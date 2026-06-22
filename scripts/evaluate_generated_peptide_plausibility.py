"""Generated-peptide plausibility analysis.

This script is intentionally conservative: it does not validate biological
activity. It checks whether generated peptides are near-neighbor copies, whether
their simple composition statistics are within the GRAMPA reference distribution,
and whether an existing QMAP HC50 head gives a rough hemolysis proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
VALID_AA = set(AA)
POSITIVE = set("KR")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AILMFWYV")
AROMATIC = set("FWY")
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


def load_generated(path: Path, max_generated: int | None) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if set(row["sequence"]).issubset(VALID_AA):
                rows.append(row)
            if max_generated is not None and len(rows) >= max_generated:
                break
    return rows


def load_reference_grampa(path: Path) -> list[str]:
    seqs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("sequence", "").strip().upper()
            if seq and set(seq).issubset(VALID_AA):
                seqs.append(seq)
    return sorted(set(seqs))


def load_reference_fasta(path: Path) -> list[str]:
    seqs: list[str] = []
    cur: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seq = "".join(cur).upper()
                    if seq and set(seq).issubset(VALID_AA):
                        seqs.append(seq)
                cur = []
            else:
                cur.append(line)
    if cur:
        seq = "".join(cur).upper()
        if seq and set(seq).issubset(VALID_AA):
            seqs.append(seq)
    return sorted(set(seqs))


def load_reference_sets(cfg: dict[str, Any]) -> dict[str, list[str]]:
    refs_cfg = cfg["inputs"].get("reference_sets")
    if not refs_cfg:
        return {"grampa": load_reference_grampa(resolve(cfg["inputs"]["reference_grampa_csv"]))}

    out: dict[str, list[str]] = {}
    for item in refs_cfg:
        name = str(item["name"])
        path = resolve(item["path"])
        kind = str(item.get("type", "fasta"))
        if kind == "grampa_csv":
            seqs = load_reference_grampa(path)
        elif kind == "fasta":
            seqs = load_reference_fasta(path)
        else:
            raise ValueError(f"Unsupported reference set type: {kind}")
        max_sequences = item.get("max_sequences")
        if max_sequences is not None:
            seqs = seqs[: int(max_sequences)]
        out[name] = seqs
    return out


def props(seq: str) -> dict[str, float]:
    n = len(seq)
    charge = sum(1 if c in POSITIVE else -1 if c in NEGATIVE else 0 for c in seq)
    return {
        "length": float(n),
        "charge": float(charge),
        "gravy": float(sum(KD_SCALE.get(c, 0.0) for c in seq) / max(n, 1)),
        "hydrophobic_fraction": float(sum(c in HYDROPHOBIC for c in seq) / max(n, 1)),
        "cationic_fraction": float(sum(c in POSITIVE for c in seq) / max(n, 1)),
        "aromatic_fraction": float(sum(c in AROMATIC for c in seq) / max(n, 1)),
        "cysteine_fraction": float(seq.count("C") / max(n, 1)),
        "pg_fraction": float((seq.count("P") + seq.count("G")) / max(n, 1)),
    }


def kmers(seq: str, k: int) -> set[str]:
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    denom = len(a | b)
    return 0.0 if denom == 0 else len(a & b) / denom


def nearest_neighbors(
    generated: list[dict[str, Any]],
    reference: list[tuple[str, str]],
    *,
    kmer_size: int,
    prefilter_top_k: int,
) -> list[dict[str, Any]]:
    ref_kmers = [kmers(seq, kmer_size) for _, seq in reference]
    out = []
    for row in generated:
        seq = row["sequence"]
        qk = kmers(seq, kmer_size)
        candidates = sorted(
            range(len(reference)),
            key=lambda i: (jaccard(qk, ref_kmers[i]), -abs(len(seq) - len(reference[i][1]))),
            reverse=True,
        )[:prefilter_top_k]
        best_idx = -1
        best_ratio = -1.0
        for idx in candidates:
            ratio = SequenceMatcher(None, seq, reference[idx][1], autojunk=False).ratio()
            if ratio > best_ratio:
                best_idx = idx
                best_ratio = ratio
        best_set, best_seq = reference[best_idx]
        out.append({
            "nearest_reference_set": best_set,
            "nearest_sequence": best_seq,
            "nearest_identity_proxy": float(best_ratio),
            "nearest_length": len(best_seq),
            "exact_match_reference": float(seq == best_seq),
        })
    return out


def aa_distribution(seqs: list[str]) -> np.ndarray:
    counts = np.ones(len(AA), dtype=float) * 1e-9
    idx = {aa: i for i, aa in enumerate(AA)}
    for seq in seqs:
        for c in seq:
            if c in idx:
                counts[idx[c]] += 1.0
    return counts / counts.sum()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def summarize_numeric(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out = {}
    for key in keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=float)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p05": float(np.quantile(vals, 0.05)),
            "p50": float(np.quantile(vals, 0.50)),
            "p95": float(np.quantile(vals, 0.95)),
        }
    return out


def load_hc50_model(split_dir: Path, cfg: dict[str, Any], device: torch.device):
    from scripts.evaluate_qmap_jepa import load_encoder
    from scripts.finetune_qmap_jepa import MeanPoolRegressor

    ckpt = torch.load(split_dir / "best_model.pt", map_location=device, weights_only=False)
    encoder, max_seq_len = load_encoder(resolve(cfg["hc50_proxy"]["pretrain_checkpoint"]), device)
    d_model = int(getattr(encoder, "d_model"))
    model = MeanPoolRegressor(
        encoder=encoder,
        d_model=d_model,
        hidden=int(cfg["hc50_proxy"].get("hidden", 512)),
        dropout=float(cfg["hc50_proxy"].get("dropout", 0.25)),
        freeze_encoder=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, max_seq_len - 2


@torch.no_grad()
def score_hc50(seqs: list[str], cfg: dict[str, Any], device: torch.device) -> tuple[np.ndarray, list[int]]:
    from scripts.finetune_qmap_jepa import predict_dataset

    root = resolve(cfg["hc50_proxy"]["checkpoint_root"])
    batch_size = int(cfg["hc50_proxy"].get("batch_size", 256))
    split_preds = []
    trunc_counts = []
    for split in cfg["hc50_proxy"]["splits"]:
        split_dir = root / f"split_{int(split)}"
        model, max_aa_len = load_hc50_model(split_dir, cfg, device)
        pred, n_truncated = predict_dataset(model, seqs, max_aa_len, batch_size, device)
        split_preds.append(pred)
        trunc_counts.append(int(n_truncated))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.stack(split_preds, axis=1), trunc_counts


def build_candidate_table(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sel = cfg.get("candidate_selection", {})
    scenario = sel.get("scenario_key", "broad_spectrum_potent")
    max_ident = float(sel.get("max_identity", 0.80))
    min_hc50 = float(sel.get("min_hc50_log10", 2.0))
    min_len = float(sel.get("min_length", 1))
    max_len = float(sel.get("max_length", 10**9))
    max_abs_charge = float(sel.get("max_abs_charge", 10**9))
    max_cationic_fraction = float(sel.get("max_cationic_fraction", 1.0))
    top_k = int(sel.get("top_k", 25))
    candidates = [
        r for r in rows
        if r["scenario_key"] == scenario
        and r["nearest_identity_proxy"] <= max_ident
        and r.get("hc50_log10_mean", -1e9) >= min_hc50
        and min_len <= r["length"] <= max_len
        and abs(r["charge"]) <= max_abs_charge
        and r["cationic_fraction"] <= max_cationic_fraction
    ]
    candidates.sort(key=lambda r: (
        r.get("esm2_independent_b0", 1e9) + r.get("jepa_oracle_b0", 1e9),
        -r.get("hc50_log10_mean", -1e9),
        r["nearest_identity_proxy"],
    ))
    keep_keys = [
        "scenario_key", "sample_index", "sequence", "nearest_identity_proxy",
        "nearest_sequence", "jepa_oracle_b0", "esm2_independent_b0",
        "hc50_log10_mean", "hc50_log10_std", "charge", "gravy", "length",
        "cationic_fraction",
    ]
    return [{k: r.get(k) for k in keep_keys} for r in candidates[:top_k]]


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], metrics: dict[str, Any], candidates: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    with open(out_dir / "predictions.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "top_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)

    lines = ["# Generated Peptide Plausibility Summary", ""]
    nn = metrics["nearest_neighbor"]
    lines.append(f"- Generated peptides evaluated: {metrics['n_generated']}")
    lines.append(f"- Reference sequences: {metrics['n_reference']} across {len(metrics['reference_sets'])} sets")
    lines.append(f"- Exact-match reference fraction: {nn['exact_match_fraction']:.3f}")
    lines.append(f"- Median nearest-neighbor identity proxy: {nn['identity_proxy']['p50']:.3f}")
    lines.append(f"- 95th percentile nearest-neighbor identity proxy: {nn['identity_proxy']['p95']:.3f}")
    lines.append("")
    lines.append("## Reference Sets")
    lines.append("")
    lines.append("| Set | n | nearest p50 | nearest p95 | exact-match fraction |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, vals in metrics["reference_sets"].items():
        lines.append(
            f"| {name} | {vals['n_reference']} | {vals['nearest_identity_proxy']['p50']:.3f} | "
            f"{vals['nearest_identity_proxy']['p95']:.3f} | {vals['exact_match_fraction']:.3f} |"
        )
    if "hc50_proxy" in metrics:
        hc = metrics["hc50_proxy"]
        lines.append(f"- Mean predicted log10 HC50 proxy: {hc['all']['mean']:.3f}")
        lines.append(f"- Fraction with predicted log10 HC50 >= 2.0: {hc['fraction_log10_ge_2']:.3f}")
    lines.append(f"- Top candidates passing filters: {len(candidates)}")
    lines.append("")
    lines.append("| Scenario | n | nearest p50 | nearest p95 | charge mean | GRAVY mean | HC50 mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, vals in metrics["by_scenario"].items():
        hc_mean = vals.get("hc50_log10", {}).get("mean", float("nan"))
        lines.append(
            f"| {key} | {vals['n']} | {vals['nearest_identity_proxy']['p50']:.3f} | "
            f"{vals['nearest_identity_proxy']['p95']:.3f} | {vals['charge']['mean']:.2f} | "
            f"{vals['gravy']['mean']:.2f} | {hc_mean:.3f} |"
        )
    lines.append("")
    lines.append("Interpretation: these are computational plausibility filters. They can flag near-neighbor copying, composition shift, and predicted hemolysis risk, but they do not validate activity or safety.")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    manifest = {
        "experiment_id": cfg["experiment_id"],
        "config": str(out_dir / "config_resolved.yaml"),
        "metrics": str(out_dir / "metrics.json"),
        "predictions": str(out_dir / "predictions.jsonl"),
        "top_candidates": str(out_dir / "top_candidates.json"),
        "summary": str(out_dir / "SUMMARY.md"),
        "status": "formal_artifact" if "formal" in cfg["experiment_id"] else "smoke_artifact",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(resolve(args.config)) as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 42)))
    use_cuda = cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()
    device = torch.device(f"cuda:{int(cfg.get('gpu', 0))}" if use_cuda else "cpu")

    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve(args.config), out_dir / "config_resolved.yaml")

    max_generated = cfg["inputs"].get("max_generated")
    generated = load_generated(resolve(cfg["inputs"]["generated_predictions"]), max_generated)
    reference_sets = load_reference_sets(cfg)
    reference = sorted({seq for seqs in reference_sets.values() for seq in seqs})
    reference_with_sets = [
        (name, seq)
        for name, seqs in reference_sets.items()
        for seq in seqs
    ]

    nn_rows = nearest_neighbors(
        generated,
        reference_with_sets,
        kmer_size=int(cfg["nearest_neighbor"].get("kmer_size", 3)),
        prefilter_top_k=int(cfg["nearest_neighbor"].get("prefilter_top_k", 75)),
    )

    rows: list[dict[str, Any]] = []
    for row, nn in zip(generated, nn_rows):
        seq_props = props(row["sequence"])
        rows.append({**row, **seq_props, **nn})

    if cfg.get("hc50_proxy", {}).get("enabled", False):
        hc50_by_split, trunc_counts = score_hc50([r["sequence"] for r in rows], cfg, device)
        for i, row in enumerate(rows):
            vals = hc50_by_split[i]
            row["hc50_log10_mean"] = float(vals.mean())
            row["hc50_log10_std"] = float(vals.std())
            row["hc50_log10_by_split"] = [float(v) for v in vals]
        hc50_summary = {
            "all": summarize_numeric(rows, ["hc50_log10_mean"])["hc50_log10_mean"],
            "fraction_log10_ge_2": float(np.mean([r["hc50_log10_mean"] >= 2.0 for r in rows])),
            "truncated_by_split": trunc_counts,
            "checkpoint_root": cfg["hc50_proxy"]["checkpoint_root"],
        }
    else:
        hc50_summary = None

    prop_keys = ["length", "charge", "gravy", "hydrophobic_fraction", "cationic_fraction", "aromatic_fraction", "cysteine_fraction", "pg_fraction"]
    ref_prop_rows = [props(seq) for seq in reference]
    metrics: dict[str, Any] = {
        "experiment_id": cfg["experiment_id"],
        "research_decision": cfg["research_decision"],
        "n_generated": len(rows),
        "n_reference": len(reference),
        "reference_set_counts": {name: len(seqs) for name, seqs in reference_sets.items()},
        "nearest_neighbor": {
            "exact_match_fraction": float(np.mean([r["exact_match_reference"] for r in rows])),
            "identity_proxy": summarize_numeric(rows, ["nearest_identity_proxy"])["nearest_identity_proxy"],
        },
        "reference_sets": {},
        "composition": {
            "generated": summarize_numeric(rows, prop_keys),
            "reference": summarize_numeric(ref_prop_rows, prop_keys),
            "aa_js_divergence_vs_reference": js_divergence(
                aa_distribution([r["sequence"] for r in rows]),
                aa_distribution(reference),
            ),
            "outlier_fractions": {
                "length_gt_50": float(np.mean([r["length"] > 50 for r in rows])),
                "abs_charge_gt_15": float(np.mean([abs(r["charge"]) > 15 for r in rows])),
                "cationic_fraction_gt_0.60": float(np.mean([r["cationic_fraction"] > 0.60 for r in rows])),
                "cysteine_fraction_gt_0.25": float(np.mean([r["cysteine_fraction"] > 0.25 for r in rows])),
            },
        },
        "by_scenario": {},
    }
    if hc50_summary is not None:
        metrics["hc50_proxy"] = hc50_summary

    for name, seqs in reference_sets.items():
        per_set_nn = nearest_neighbors(
            generated,
            [(name, seq) for seq in seqs],
            kmer_size=int(cfg["nearest_neighbor"].get("kmer_size", 3)),
            prefilter_top_k=int(cfg["nearest_neighbor"].get("prefilter_top_k", 75)),
        )
        metrics["reference_sets"][name] = {
            "n_reference": len(seqs),
            "exact_match_fraction": float(np.mean([r["exact_match_reference"] for r in per_set_nn])),
            "nearest_identity_proxy": summarize_numeric(per_set_nn, ["nearest_identity_proxy"])["nearest_identity_proxy"],
        }

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario_key"]].append(row)
    for key, sub in sorted(by_scenario.items()):
        scenario_metrics = {
            "n": len(sub),
            "nearest_identity_proxy": summarize_numeric(sub, ["nearest_identity_proxy"])["nearest_identity_proxy"],
            "exact_match_fraction": float(np.mean([r["exact_match_reference"] for r in sub])),
            **summarize_numeric(sub, ["length", "charge", "gravy"]),
        }
        if hc50_summary is not None:
            scenario_metrics["hc50_log10"] = summarize_numeric(sub, ["hc50_log10_mean"])["hc50_log10_mean"]
            scenario_metrics["fraction_hc50_log10_ge_2"] = float(np.mean([r["hc50_log10_mean"] >= 2.0 for r in sub]))
        metrics["by_scenario"][key] = scenario_metrics

    candidates = build_candidate_table(rows, cfg)
    write_outputs(out_dir, rows, metrics, candidates, cfg)
    print(f"Saved generated-peptide plausibility artifacts to {out_dir}")


if __name__ == "__main__":
    main()
