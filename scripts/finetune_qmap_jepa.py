"""
Head-only or full fine-tuning of JEPA-AMP on QMAP regression targets.

The script follows QMAP's official split protocol:
  - obtain the split-specific train mask with QMAPBenchmark.get_train_mask
  - train only on allowed DBAASP samples with the selected target label
  - evaluate on the benchmark with QMAPBenchmark.compute_metrics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from qmap import DBAASPDataset, QMAPBenchmark  # noqa: E402

from scripts.evaluate_qmap_jepa import ECOLI, collate_sequences, load_encoder
from src.data.tokenizer import encode


def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


@dataclass
class FineTuneResult:
    split: int
    mode: str
    target: str
    property_key: str
    n_train_allowed: int
    n_train_target: int
    n_val_target: int
    n_full_target: int
    n_high_eff_ecoli: int | None
    n_train_truncated: int
    n_benchmark_truncated: int
    best_epoch: int
    best_val_loss: float
    full_target_pearson: float
    full_target_rmse: float
    high_eff_ecoli_pearson: float | None
    high_eff_ecoli_rmse: float | None


class QMAPMICDataset(Dataset):
    def __init__(self, sequences: list[str], targets: np.ndarray, max_aa_len: int):
        self.sequences = sequences
        self.targets = targets.astype(np.float32)
        self.max_aa_len = max_aa_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        ids = torch.tensor(encode(seq[: self.max_aa_len], add_special_tokens=True), dtype=torch.long)
        return {
            "input_ids": ids,
            "length": torch.tensor(len(ids), dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "truncated": torch.tensor(len(seq) > self.max_aa_len),
        }


def collate_mic(batch: list[dict]) -> dict:
    out = collate_sequences(batch)
    out["target"] = torch.stack([item["target"] for item in batch])
    return out


class MeanPoolRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int, hidden: int, dropout: float, freeze_encoder: bool):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids)  # (B, L, D)
        pooled = []
        for i, length in enumerate(lengths.tolist()):
            pooled.append(h[i, 1 : length - 1].mean(0))
        x = torch.stack(pooled)
        return self.head(x).squeeze(-1)


def target_key(target: str) -> str:
    if target == "ecoli":
        return ECOLI
    if target == "hc50":
        return "hc50"
    raise ValueError(f"Unsupported target: {target}")


def target_xy(dataset: DBAASPDataset, target: str) -> tuple[list[str], np.ndarray]:
    sequences, targets = [], []
    for sample in dataset.samples:
        if target == "ecoli":
            value = sample.targets.get(ECOLI)
            consensus = None if value is None else value.consensus
        elif target == "hc50":
            consensus = None if sample.hc50 is None else sample.hc50.consensus
        else:
            raise ValueError(f"Unsupported target: {target}")
        if consensus is None or math.isnan(consensus) or consensus <= 0:
            continue
        sequences.append(sample.sequence)
        targets.append(math.log10(consensus))
    return sequences, np.asarray(targets, dtype=np.float32)


def predictions_for_key(key: str, values: np.ndarray) -> list[dict[str, float]]:
    return [{key: float(v)} for v in values]


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    return idx[n_val:], idx[:n_val]


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            target = batch["target"].to(device)
            pred = model(input_ids, lengths)
            loss = F.mse_loss(pred, target, reduction="sum")
            total += float(loss.item())
            count += int(target.numel())
    return total / max(1, count)


@torch.no_grad()
def predict_dataset(model: nn.Module, sequences: list[str], max_aa_len: int, batch_size: int, device: torch.device):
    from scripts.evaluate_qmap_jepa import SequenceDataset

    ds = SequenceDataset(sequences, max_aa_len=max_aa_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_sequences)
    preds = []
    n_truncated = 0
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        pred = model(input_ids, lengths)
        preds.extend(pred.cpu().float().tolist())
        n_truncated += int(batch["truncated"].sum().item())
    return np.asarray(preds, dtype=np.float32), n_truncated


def run_split(args: argparse.Namespace, split: int, device: torch.device) -> FineTuneResult:
    torch.manual_seed(args.seed + split)
    np.random.seed(args.seed + split)
    random.seed(args.seed + split)

    encoder, max_seq_len = load_encoder(args.checkpoint, device)
    max_aa_len = max_seq_len - 2
    d_model = int(getattr(encoder, "d_model"))
    model = MeanPoolRegressor(
        encoder,
        d_model=d_model,
        hidden=args.hidden,
        dropout=args.dropout,
        freeze_encoder=(args.mode == "head"),
    ).to(device)

    dataset = DBAASPDataset()
    benchmark = QMAPBenchmark(split=split)
    train_mask = benchmark.get_train_mask(dataset.sequences, show_progress=True, num_threads=args.mask_threads)
    train_dataset = dataset[train_mask]
    property_key = target_key(args.target)
    sequences, y = target_xy(train_dataset, args.target)
    if len(sequences) < 10:
        raise RuntimeError(f"split {split}: too few {args.target} training samples ({len(sequences)})")
    train_idx, val_idx = split_indices(len(sequences), args.val_ratio, args.seed + split)

    mic_dataset = QMAPMICDataset(sequences, y, max_aa_len=max_aa_len)
    train_loader = DataLoader(
        Subset(mic_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_mic,
    )
    val_loader = DataLoader(
        Subset(mic_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_mic,
    )

    encoder_ids = {id(p) for p in model.encoder.parameters()}
    if args.mode == "full":
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids], "lr": args.lr},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.lr_encoder},
        ]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.fp16))

    split_dir = args.out_dir / f"split_{split}"
    split_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch, no_improve = float("inf"), -1, 0
    best_path = split_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.fp16)):
                pred = model(input_ids, lengths)
                loss = F.huber_loss(pred, target, delta=0.5)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * int(target.numel())
            seen += int(target.numel())
        val_loss = evaluate_loss(model, val_loader, device)
        print(
            f"split={split} epoch={epoch:03d} train={running / max(1, seen):.4f} "
            f"val_mse={val_loss:.4f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "args": jsonable_args(args)}, best_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    bench_pred, n_benchmark_truncated = predict_dataset(model, benchmark.sequences, max_aa_len, args.eval_batch_size, device)
    pred_all = predictions_for_key(property_key, bench_pred)
    full_metrics = benchmark.compute_metrics(pred_all, log=True, mean_metrics=False)

    high_metrics = {}
    if args.target == "ecoli":
        pred_by_id = {sample.id: pred for sample, pred in zip(benchmark.samples, pred_all)}
        high_eff = benchmark.with_bacterial_targets([ECOLI]).with_efficiency_below(10.0)
        high_pred = [pred_by_id[sample.id] for sample in high_eff.samples]
        high_metrics = high_eff.compute_metrics(high_pred, log=True, mean_metrics=False)

    n_train_truncated = sum(len(sequences[i]) > max_aa_len for i in train_idx)
    with open(split_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "split": split,
                "mode": args.mode,
                "target": args.target,
                "property_key": property_key,
                "full": {k: v.dict() for k, v in full_metrics.items()},
                "high_eff": {k: v.dict() for k, v in high_metrics.items()},
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "n_train_allowed": len(train_dataset),
                "n_train_target": len(train_idx),
                "n_val_target": len(val_idx),
                "n_train_truncated": n_train_truncated,
                "n_benchmark_truncated": n_benchmark_truncated,
                "max_aa_len": max_aa_len,
                "args": jsonable_args(args),
            },
            f,
            indent=2,
        )
    with open(split_dir / f"predictions_{args.target}.jsonl", "w") as f:
        for sample, pred in zip(benchmark.samples, pred_all):
            f.write(json.dumps({"id": sample.id, "sequence": sample.sequence, **pred}) + "\n")

    full_target = full_metrics[property_key]
    high_ecoli = high_metrics.get(ECOLI)
    return FineTuneResult(
        split=split,
        mode=args.mode,
        target=args.target,
        property_key=property_key,
        n_train_allowed=len(train_dataset),
        n_train_target=len(train_idx),
        n_val_target=len(val_idx),
        n_full_target=full_target.n,
        n_high_eff_ecoli=None if high_ecoli is None else high_ecoli.n,
        n_train_truncated=n_train_truncated,
        n_benchmark_truncated=n_benchmark_truncated,
        best_epoch=best_epoch,
        best_val_loss=float(best_val),
        full_target_pearson=float(full_target.pearson),
        full_target_rmse=float(full_target.rmse),
        high_eff_ecoli_pearson=None if high_ecoli is None else float(high_ecoli.pearson),
        high_eff_ecoli_rmse=None if high_ecoli is None else float(high_ecoli.rmse),
    )


def summarize(results: list[FineTuneResult]) -> dict:
    full = np.asarray([r.full_target_pearson for r in results], dtype=float)
    out = {
        "full_target": {"min_pcc": float(full.min()), "mean_pcc": float(full.mean()), "max_pcc": float(full.max())},
    }
    high_values = [r.high_eff_ecoli_pearson for r in results if r.high_eff_ecoli_pearson is not None]
    if high_values:
        high = np.asarray(high_values, dtype=float)
        out["high_eff_ecoli"] = {"min_pcc": float(high.min()), "mean_pcc": float(high.mean()), "max_pcc": float(high.max())}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/jepa_pretrain_868k/last_jepa.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/qmap_jepa_head_finetune"))
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--mode", choices=["head", "full"], default="head")
    parser.add_argument("--target", choices=["ecoli", "hc50"], default="ecoli")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-threads", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for split in args.splits:
        print(f"\n=== QMAP fine-tune split {split} target={args.target} mode={args.mode} device={device} ===", flush=True)
        result = run_split(args, split, device)
        print(result, flush=True)
        results.append(result)
    out = {
        "method": f"JEPA-AMP {args.mode} fine-tune {args.target}",
        "target": args.target,
        "splits": [asdict(r) for r in results],
        "summary": summarize(results),
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
