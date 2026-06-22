"""
Create train/val/test splits from the processed AMP manifest.

Creates splits by:
 - source_holdout: hold out one or more named sources (e.g. amplify, ampbenchmark)
 - random: random shuffle split with given val/test ratios

Outputs TSV lists under `data/splits/` with columns: sequence_id\tsequence\tsources

Usage examples:
  python scripts/create_splits.py --mode source_holdout --test-sources amplify --val-ratio 0.1 --outdir data/splits
  python scripts/create_splits.py --mode random --val-ratio 0.1 --test-ratio 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Set


def stream_manifest(manifest_path: Path):
    with open(manifest_path, "r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            rec = dict(zip(header, parts))
            yield rec


def build_index(manifest_path: Path):
    # returns dict: seq_id -> {sequence, sources(list)}
    idx: Dict[str, Dict] = {}
    for rec in stream_manifest(manifest_path):
        seq_id = rec.get("sequence_id") or rec.get("sequence_id")
        seq = rec.get("sequence")
        sources = rec.get("sources", "").split("|") if rec.get("sources") else []
        idx[seq_id] = {"sequence": seq, "sources": sources}
    return idx


def split_by_source(idx: Dict[str, Dict], test_sources: Set[str], val_ratio: float = 0.1, seed: int = 0):
    train, val, test = [], [], []
    rng = random.Random(seed)
    for seq_id, info in idx.items():
        srcs = set(info.get("sources", []))
        if srcs & test_sources:
            test.append((seq_id, info))
        else:
            train.append((seq_id, info))

    # split train into train+val
    rng.shuffle(train)
    n_val = int(len(train) * val_ratio)
    val = train[:n_val]
    train = train[n_val:]
    return train, val, test


def split_random(idx: Dict[str, Dict], val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 0):
    items = list(idx.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = items[:n_test]
    val = items[n_test : n_test + n_val]
    train = items[n_test + n_val :]
    # convert to (seq_id, info)
    return [(k, v) for k, v in train], [(k, v) for k, v in val], [(k, v) for k, v in test]


def write_split(split: List, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sequence_id", "length", "sources", "record_ids", "sequence"])
        for seq_id, info in split:
            length = len(info.get("sequence")) if info.get("sequence") else ""
            sources = "|".join(info.get("sources", []))
            record_ids = info.get("record_ids", "")
            seq = info.get("sequence", "")
            w.writerow([seq_id, length, sources, record_ids, seq])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["source_holdout", "random"], default="random")
    p.add_argument("--manifest", type=Path, default=Path("data/processed/amp_sequence_manifest.tsv"))
    p.add_argument("--test-sources", type=str, default="", help="comma-separated source names to hold out as test")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=Path("data/splits"))
    args = p.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    print("Building index from manifest... this may take a moment")
    idx = {}
    # build index while capturing sources and record_ids if present
    with open(args.manifest, "r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            rec = dict(zip(header, parts))
            seq_id = rec.get("sequence_id")
            seq = rec.get("sequence")
            sources = rec.get("sources", "").split("|") if rec.get("sources") else []
            record_ids = rec.get("record_ids", "")
            idx[seq_id] = {"sequence": seq, "sources": sources, "record_ids": record_ids}

    if args.mode == "source_holdout":
        if not args.test_sources:
            raise SystemExit("--test-sources required for source_holdout mode")
        test_sources = set(s.strip() for s in args.test_sources.split(",") if s.strip())
        train, val, test = split_by_source(idx, test_sources=test_sources, val_ratio=args.val_ratio, seed=args.seed)
    else:
        train, val, test = split_random(idx, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)

    outdir = args.outdir
    write_split(train, outdir / "train.tsv")
    write_split(val, outdir / "val.tsv")
    write_split(test, outdir / "test.tsv")

    meta = {
        "mode": args.mode,
        "seed": args.seed,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    with open(outdir / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote splits to {outdir} (train={len(train)}, val={len(val)}, test={len(test)})")


if __name__ == "__main__":
    main()
