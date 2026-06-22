"""
Export QMAP prediction JSONL files from saved fine-tuned checkpoints.

This is a lightweight reproducibility utility: it does not retrain models. It
loads split-local best_model.pt checkpoints and writes the prediction files that
QMAP leaderboard reviewers can inspect alongside metrics.json and summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from qmap import QMAPBenchmark

from scripts.evaluate_qmap_jepa import ECOLI, load_encoder
from scripts.finetune_qmap_conditional import ConditionalRegressor, HC50, predict_property
from scripts.finetune_qmap_jepa import MeanPoolRegressor, predict_dataset, predictions_for_key, target_key


def export_single(args: argparse.Namespace, split: int, device: torch.device) -> None:
    split_dir = args.out_dir / f"split_{split}"
    ckpt = torch.load(split_dir / "best_model.pt", map_location=device, weights_only=False)
    encoder, max_seq_len = load_encoder(args.checkpoint, device)
    max_aa_len = max_seq_len - 2
    d_model = int(getattr(encoder, "d_model"))
    model = MeanPoolRegressor(
        encoder=encoder,
        d_model=d_model,
        hidden=args.hidden,
        dropout=args.dropout,
        freeze_encoder=(args.mode == "head"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    benchmark = QMAPBenchmark(split=split)
    values, _ = predict_dataset(model, benchmark.sequences, max_aa_len, args.batch_size, device)
    key = target_key(args.target)
    predictions = predictions_for_key(key, values)
    out_path = split_dir / f"predictions_{args.target}.jsonl"
    with open(out_path, "w") as f:
        for sample, pred in zip(benchmark.samples, predictions):
            f.write(json.dumps({"id": sample.id, "sequence": sample.sequence, **pred}) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


def export_conditional(args: argparse.Namespace, split: int, device: torch.device) -> None:
    split_dir = args.out_dir / f"split_{split}"
    ckpt = torch.load(split_dir / "best_model.pt", map_location=device, weights_only=False)
    properties = ckpt["properties"]
    property_to_idx = ckpt.get("property_to_idx") or {prop: i for i, prop in enumerate(properties)}
    encoder, max_seq_len = load_encoder(args.checkpoint, device)
    max_aa_len = max_seq_len - 2
    d_model = int(getattr(encoder, "d_model"))
    model = ConditionalRegressor(
        encoder=encoder,
        d_model=d_model,
        n_properties=len(properties),
        property_dim=args.property_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        freeze_encoder=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    benchmark = QMAPBenchmark(split=split)
    ecoli_values, _ = predict_property(model, benchmark.sequences, ECOLI, property_to_idx, max_aa_len, args.batch_size, device)
    hc50_values, _ = predict_property(model, benchmark.sequences, HC50, property_to_idx, max_aa_len, args.batch_size, device)
    out_path = split_dir / "predictions_conditional.jsonl"
    with open(out_path, "w") as f:
        for sample, ecoli, hc50 in zip(benchmark.samples, ecoli_values, hc50_values):
            f.write(json.dumps({"id": sample.id, "sequence": sample.sequence, ECOLI: float(ecoli), HC50: float(hc50)}) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/jepa_pretrain_868k/last_jepa.pt"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["single", "conditional"], required=True)
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--target", choices=["ecoli", "hc50"], default="ecoli")
    parser.add_argument("--mode", choices=["head", "full"], default="head")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--property-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    for split in args.splits:
        if args.kind == "single":
            export_single(args, split, device)
        else:
            export_conditional(args, split, device)


if __name__ == "__main__":
    main()
