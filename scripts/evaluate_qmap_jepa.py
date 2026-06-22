"""
Evaluate JEPA-AMP frozen embeddings on the QMAP benchmark.

This script uses the official qmap-benchmark split masks and metric code:
for each split, it trains a Ridge regressor on QMAP-allowed DBAASP samples
with Escherichia coli MIC labels, predicts the QMAP benchmark samples, and
reports Full and high-efficiency e. coli Pearson correlations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from qmap import DBAASPDataset, QMAPBenchmark  # noqa: E402

from src.data.tokenizer import PAD_ID, encode
from src.models.jepa import JEPA


ECOLI = "Escherichia coli"


@dataclass
class SplitResult:
    split: int
    n_train_allowed: int
    n_train_ecoli: int
    n_benchmark: int
    n_full_ecoli: int
    n_high_eff_ecoli: int
    n_train_truncated: int
    n_benchmark_truncated: int
    full_ecoli_pearson: float
    full_ecoli_rmse: float
    high_eff_ecoli_pearson: float
    high_eff_ecoli_rmse: float


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str], max_aa_len: int):
        self.sequences = sequences
        self.max_aa_len = max_aa_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        truncated = len(seq) > self.max_aa_len
        ids = torch.tensor(encode(seq[: self.max_aa_len], add_special_tokens=True), dtype=torch.long)
        return {"input_ids": ids, "length": torch.tensor(len(ids)), "truncated": torch.tensor(truncated)}


def collate_sequences(batch: list[dict]) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    lengths = torch.empty(len(batch), dtype=torch.long)
    truncated = torch.empty(len(batch), dtype=torch.bool)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        input_ids[i, : len(ids)] = ids
        lengths[i] = item["length"]
        truncated[i] = item["truncated"]
    return {"input_ids": input_ids, "lengths": lengths, "truncated": truncated}


def load_encoder(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = JEPA(**cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    encoder = model.context_encoder.to(device).eval()
    max_seq_len = int(cfg["model"].get("max_seq_len", 52))
    return encoder, max_seq_len


@torch.no_grad()
def embed_sequences(
    encoder,
    sequences: list[str],
    *,
    device: torch.device,
    max_aa_len: int,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, int]:
    ds = SequenceDataset(sequences, max_aa_len=max_aa_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_sequences,
    )
    embs: list[np.ndarray] = []
    n_truncated = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].tolist()
        h = encoder(input_ids)  # (B, L, D)
        for i, length in enumerate(lengths):
            # Mean-pool amino-acid tokens only, excluding BOS/EOS.
            embs.append(h[i, 1 : length - 1].mean(0).cpu().float().numpy())
        n_truncated += int(batch["truncated"].sum().item())
    return np.stack(embs), n_truncated


def ecoli_xy(dataset: DBAASPDataset) -> tuple[list[str], np.ndarray]:
    sequences, targets = [], []
    for sample in dataset.samples:
        target = sample.targets.get(ECOLI)
        if target is None:
            continue
        if target.consensus is None or math.isnan(target.consensus) or target.consensus <= 0:
            continue
        sequences.append(sample.sequence)
        targets.append(math.log10(target.consensus))
    return sequences, np.asarray(targets, dtype=np.float32)


def predictions_for(dataset: DBAASPDataset, values: Iterable[float]) -> list[dict[str, float]]:
    return [{ECOLI: float(v)} for v in values]


def metric_dict(metrics: dict, key: str) -> dict:
    metric = metrics.get(key)
    if metric is None:
        return {}
    return metric.dict()


def run_split(
    split: int,
    *,
    encoder,
    device: torch.device,
    max_aa_len: int,
    batch_size: int,
    num_workers: int,
    alpha: float,
    out_dir: Path,
    mask_threads: int | None,
) -> SplitResult:
    dataset = DBAASPDataset()
    benchmark = QMAPBenchmark(split=split)

    train_mask = benchmark.get_train_mask(
        dataset.sequences,
        show_progress=True,
        num_threads=mask_threads,
    )
    train_dataset = dataset[train_mask]
    train_sequences, y_train = ecoli_xy(train_dataset)
    if len(train_sequences) < 10:
        raise RuntimeError(f"split {split}: too few e. coli training samples ({len(train_sequences)})")

    X_train, n_train_truncated = embed_sequences(
        encoder,
        train_sequences,
        device=device,
        max_aa_len=max_aa_len,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    reg.fit(X_train, y_train)

    X_bench, n_benchmark_truncated = embed_sequences(
        encoder,
        benchmark.sequences,
        device=device,
        max_aa_len=max_aa_len,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    y_pred = reg.predict(X_bench)
    pred_all = predictions_for(benchmark, y_pred)
    full_metrics = benchmark.compute_metrics(pred_all, log=True, mean_metrics=False)

    pred_by_id = {sample.id: pred for sample, pred in zip(benchmark.samples, pred_all)}
    high_eff = benchmark.with_bacterial_targets([ECOLI]).with_efficiency_below(10.0)
    high_pred = [pred_by_id[sample.id] for sample in high_eff.samples]
    high_metrics = high_eff.compute_metrics(high_pred, log=True, mean_metrics=False)

    split_dir = out_dir / f"split_{split}"
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "split": split,
                "full": {k: v.dict() for k, v in full_metrics.items()},
                "high_eff": {k: v.dict() for k, v in high_metrics.items()},
                "n_train_allowed": len(train_dataset),
                "n_train_ecoli": len(train_sequences),
                "n_train_truncated": n_train_truncated,
                "n_benchmark_truncated": n_benchmark_truncated,
                "alpha": alpha,
                "max_aa_len": max_aa_len,
            },
            f,
            indent=2,
        )
    with open(split_dir / "predictions_ecoli.jsonl", "w") as f:
        for sample, pred in zip(benchmark.samples, pred_all):
            f.write(json.dumps({"id": sample.id, "sequence": sample.sequence, **pred}) + "\n")

    full_ecoli = full_metrics[ECOLI]
    high_ecoli = high_metrics[ECOLI]
    return SplitResult(
        split=split,
        n_train_allowed=len(train_dataset),
        n_train_ecoli=len(train_sequences),
        n_benchmark=len(benchmark),
        n_full_ecoli=full_ecoli.n,
        n_high_eff_ecoli=high_ecoli.n,
        n_train_truncated=n_train_truncated,
        n_benchmark_truncated=n_benchmark_truncated,
        full_ecoli_pearson=float(full_ecoli.pearson),
        full_ecoli_rmse=float(full_ecoli.rmse),
        high_eff_ecoli_pearson=float(high_ecoli.pearson),
        high_eff_ecoli_rmse=float(high_ecoli.rmse),
    )


def summarize(results: list[SplitResult]) -> dict:
    full = np.array([r.full_ecoli_pearson for r in results], dtype=float)
    high = np.array([r.high_eff_ecoli_pearson for r in results], dtype=float)
    return {
        "full_ecoli": {
            "min_pcc": float(np.nanmin(full)),
            "mean_pcc": float(np.nanmean(full)),
            "max_pcc": float(np.nanmax(full)),
        },
        "high_eff_ecoli": {
            "min_pcc": float(np.nanmin(high)),
            "mean_pcc": float(np.nanmean(high)),
            "max_pcc": float(np.nanmax(high)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/jepa_pretrain_868k/last_jepa.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/qmap_jepa_frozen_ridge"))
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mask-threads", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoder, max_seq_len = load_encoder(args.checkpoint, device)
    max_aa_len = max_seq_len - 2

    results = []
    for split in args.splits:
        print(f"\n=== QMAP split {split} on {device} ===", flush=True)
        result = run_split(
            split,
            encoder=encoder,
            device=device,
            max_aa_len=max_aa_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            alpha=args.alpha,
            out_dir=args.out_dir,
            mask_threads=args.mask_threads,
        )
        print(result, flush=True)
        results.append(result)

    summary = {
        "method": "JEPA-AMP frozen mean embedding + Ridge",
        "checkpoint": str(args.checkpoint),
        "splits": [asdict(r) for r in results],
        "summary": summarize(results),
        "notes": {
            "target": "log10 consensus MIC, as expected by QMAP compute_metrics(log=True)",
            "sequence_length": f"Sequences longer than {max_aa_len} AA are truncated for JEPA positional compatibility.",
        },
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
