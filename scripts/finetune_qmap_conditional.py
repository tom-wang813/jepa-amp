"""
Conditional JEPA-AMP fine-tuning on QMAP bacterial MIC and HC50 regression.

Training records are (sequence, property_key, log10 consensus value). The
property key is either a QMAP bacterial target name or "hc50", matching the keys
expected by QMAPBenchmark.compute_metrics.
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
from scripts.finetune_qmap_jepa import jsonable_args, split_indices
from src.data.tokenizer import encode


HC50 = "hc50"


@dataclass
class ConditionalResult:
    split: int
    n_train_allowed: int
    n_train_records: int
    n_val_records: int
    n_properties: int
    n_train_truncated: int
    n_benchmark_truncated: int
    best_epoch: int
    best_val_loss: float
    full_ecoli_pearson: float
    full_ecoli_rmse: float
    high_eff_ecoli_pearson: float
    high_eff_ecoli_rmse: float
    hc50_pearson: float
    hc50_rmse: float


def valid_value(value: float | None) -> bool:
    return value is not None and not math.isnan(value) and value > 0


def build_records(dataset: DBAASPDataset) -> tuple[list[tuple[str, str, float]], list[str]]:
    records: list[tuple[str, str, float]] = []
    properties: set[str] = set()
    for sample in dataset.samples:
        for key, target in sample.targets.items():
            consensus = None if target is None else target.consensus
            if valid_value(consensus):
                records.append((sample.sequence, key, math.log10(consensus)))
                properties.add(key)
        hc50 = None if sample.hc50 is None else sample.hc50.consensus
        if valid_value(hc50):
            records.append((sample.sequence, HC50, math.log10(hc50)))
            properties.add(HC50)
    return records, sorted(properties)


class ConditionalQMAPDataset(Dataset):
    def __init__(self, records: list[tuple[str, str, float]], property_to_idx: dict[str, int], max_aa_len: int):
        self.records = records
        self.property_to_idx = property_to_idx
        self.max_aa_len = max_aa_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        seq, prop, target = self.records[idx]
        ids = torch.tensor(encode(seq[: self.max_aa_len], add_special_tokens=True), dtype=torch.long)
        return {
            "input_ids": ids,
            "length": torch.tensor(len(ids), dtype=torch.long),
            "property_idx": torch.tensor(self.property_to_idx[prop], dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            "truncated": torch.tensor(len(seq) > self.max_aa_len),
        }


def collate_conditional(batch: list[dict]) -> dict:
    out = collate_sequences(batch)
    out["property_idx"] = torch.stack([item["property_idx"] for item in batch])
    out["target"] = torch.stack([item["target"] for item in batch])
    return out


class ConditionalRegressor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        d_model: int,
        n_properties: int,
        property_dim: int,
        hidden: int,
        dropout: float,
        freeze_encoder: bool,
    ):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        self.property_emb = nn.Embedding(n_properties, property_dim)
        self.film = nn.Linear(property_dim, 2 * d_model)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
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

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor, property_idx: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids)  # (B, L, D)
        pooled = []
        for i, length in enumerate(lengths.tolist()):
            pooled.append(h[i, 1 : length - 1].mean(0))
        x = torch.stack(pooled)
        cond = self.property_emb(property_idx)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        x = x * (1 + gamma) + beta
        return self.head(x).squeeze(-1)


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["input_ids"].to(device),
                batch["lengths"].to(device),
                batch["property_idx"].to(device),
            )
            target = batch["target"].to(device)
            total += float(F.mse_loss(pred, target, reduction="sum").item())
            count += int(target.numel())
    return total / max(1, count)


@torch.no_grad()
def predict_property(
    model: ConditionalRegressor,
    sequences: list[str],
    property_key: str,
    property_to_idx: dict[str, int],
    max_aa_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    from scripts.evaluate_qmap_jepa import SequenceDataset

    ds = SequenceDataset(sequences, max_aa_len=max_aa_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_sequences)
    prop_idx = property_to_idx[property_key]
    preds: list[float] = []
    n_truncated = 0
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        prop = torch.full((input_ids.shape[0],), prop_idx, dtype=torch.long, device=device)
        pred = model(input_ids, lengths, prop)
        preds.extend(pred.cpu().float().tolist())
        n_truncated += int(batch["truncated"].sum().item())
    return np.asarray(preds, dtype=np.float32), n_truncated


def run_split(args: argparse.Namespace, split: int, device: torch.device) -> ConditionalResult:
    torch.manual_seed(args.seed + split)
    np.random.seed(args.seed + split)
    random.seed(args.seed + split)

    encoder, max_seq_len = load_encoder(args.checkpoint, device)
    max_aa_len = max_seq_len - 2
    d_model = int(getattr(encoder, "d_model"))

    dataset = DBAASPDataset()
    benchmark = QMAPBenchmark(split=split)
    train_mask = benchmark.get_train_mask(dataset.sequences, show_progress=True, num_threads=args.mask_threads)
    train_dataset = dataset[train_mask]
    records, properties = build_records(train_dataset)
    for required in (ECOLI, HC50):
        if required not in properties:
            raise RuntimeError(f"split {split}: missing required property {required!r} in training data")
    property_to_idx = {prop: i for i, prop in enumerate(properties)}
    train_idx, val_idx = split_indices(len(records), args.val_ratio, args.seed + split)

    qmap_dataset = ConditionalQMAPDataset(records, property_to_idx, max_aa_len)
    train_loader = DataLoader(
        Subset(qmap_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_conditional,
    )
    val_loader = DataLoader(
        Subset(qmap_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_conditional,
    )

    model = ConditionalRegressor(
        encoder=encoder,
        d_model=d_model,
        n_properties=len(properties),
        property_dim=args.property_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        freeze_encoder=True,
    ).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
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
            prop = batch["property_idx"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.fp16)):
                pred = model(input_ids, lengths, prop)
                loss = F.huber_loss(pred, target, delta=0.5)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * int(target.numel())
            seen += int(target.numel())
        val_loss = evaluate_loss(model, val_loader, device)
        print(f"split={split} epoch={epoch:03d} train={running / max(1, seen):.4f} val_mse={val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "properties": properties,
                    "property_to_idx": property_to_idx,
                    "args": jsonable_args(args),
                },
                best_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    ecoli_pred, n_benchmark_truncated = predict_property(
        model, benchmark.sequences, ECOLI, property_to_idx, max_aa_len, args.eval_batch_size, device
    )
    hc50_pred, _ = predict_property(model, benchmark.sequences, HC50, property_to_idx, max_aa_len, args.eval_batch_size, device)
    pred_all = [{ECOLI: float(e), HC50: float(h)} for e, h in zip(ecoli_pred, hc50_pred)]
    full_metrics = benchmark.compute_metrics(pred_all, log=True, mean_metrics=False)

    pred_by_id = {sample.id: pred for sample, pred in zip(benchmark.samples, pred_all)}
    high_eff = benchmark.with_bacterial_targets([ECOLI]).with_efficiency_below(10.0)
    high_pred = [pred_by_id[sample.id] for sample in high_eff.samples]
    high_metrics = high_eff.compute_metrics(high_pred, log=True, mean_metrics=False)

    n_train_truncated = sum(len(records[i][0]) > max_aa_len for i in train_idx)
    with open(split_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "split": split,
                "full": {k: v.dict() for k, v in full_metrics.items()},
                "high_eff": {k: v.dict() for k, v in high_metrics.items()},
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "n_train_allowed": len(train_dataset),
                "n_train_records": len(train_idx),
                "n_val_records": len(val_idx),
                "n_properties": len(properties),
                "properties": properties,
                "n_train_truncated": n_train_truncated,
                "n_benchmark_truncated": n_benchmark_truncated,
                "max_aa_len": max_aa_len,
                "args": jsonable_args(args),
            },
            f,
            indent=2,
        )
    with open(split_dir / "predictions_conditional.jsonl", "w") as f:
        for sample, pred in zip(benchmark.samples, pred_all):
            f.write(json.dumps({"id": sample.id, "sequence": sample.sequence, **pred}) + "\n")

    full_ecoli = full_metrics[ECOLI]
    high_ecoli = high_metrics[ECOLI]
    hc50 = full_metrics[HC50]
    return ConditionalResult(
        split=split,
        n_train_allowed=len(train_dataset),
        n_train_records=len(train_idx),
        n_val_records=len(val_idx),
        n_properties=len(properties),
        n_train_truncated=n_train_truncated,
        n_benchmark_truncated=n_benchmark_truncated,
        best_epoch=best_epoch,
        best_val_loss=float(best_val),
        full_ecoli_pearson=float(full_ecoli.pearson),
        full_ecoli_rmse=float(full_ecoli.rmse),
        high_eff_ecoli_pearson=float(high_ecoli.pearson),
        high_eff_ecoli_rmse=float(high_ecoli.rmse),
        hc50_pearson=float(hc50.pearson),
        hc50_rmse=float(hc50.rmse),
    )


def summarize(results: list[ConditionalResult]) -> dict:
    def stats(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {"min_pcc": float(arr.min()), "mean_pcc": float(arr.mean()), "max_pcc": float(arr.max())}

    return {
        "full_ecoli": stats([r.full_ecoli_pearson for r in results]),
        "high_eff_ecoli": stats([r.high_eff_ecoli_pearson for r in results]),
        "hc50": stats([r.hc50_pearson for r in results]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/jepa_pretrain_868k/last_jepa.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/qmap_jepa_conditional_seed42"))
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-threads", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--property-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        print(f"\n=== QMAP conditional split {split} device={device} ===", flush=True)
        result = run_split(args, split, device)
        print(result, flush=True)
        results.append(result)
    out = {
        "method": "JEPA-AMP conditional property fine-tune",
        "splits": [asdict(r) for r in results],
        "summary": summarize(results),
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
