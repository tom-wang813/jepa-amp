"""
Evaluate MIC top-k enrichment for generated sequences.

Calculates Precision@k and Enrichment Factor EF@k compared to random sampling
from the pseudolabeled MIC corpus. Uses bootstrap to estimate 95% CI.

Usage example:
  python scripts/evaluate_mic_enrichment.py \
    --gen-json eval_results/conditional_gen_test.json \
    --mic-seqs data/processed/mic_pseudolabels_seqs.txt \
    --mic-npy data/processed/mic_pseudolabels.npy \
    --out eval_results/mic_enrichment_conditional_test.json
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Tuple


def load_mic(mic_seqs_path: Path, mic_npy_path: Path) -> Dict[str, float]:
    seqs = [x.strip() for x in open(mic_seqs_path) if x.strip()]
    vals = np.load(mic_npy_path)
    if len(seqs) != len(vals):
        raise ValueError("mic seqs and values length mismatch")
    return {s: float(np.mean(v)) for s, v in zip(seqs, vals)}


def load_generated(gen_json_path: Path) -> List[str]:
    data = json.load(open(gen_json_path))
    seqs = []
    # JSON structure: top-level keys -> categories -> 'samples' list
    def extract_from_obj(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "samples" and isinstance(v, list):
                    seqs.extend(v)
                else:
                    extract_from_obj(v)
        elif isinstance(obj, list):
            for item in obj:
                extract_from_obj(item)

    extract_from_obj(data)
    # normalize sequences (uppercase, remove whitespace)
    return [s.upper().replace(" ", "") for s in seqs]


def precision_at_k(sorted_seqs: List[str], mic_map: Dict[str, float], k: int, threshold: float) -> float:
    topk = sorted_seqs[:k]
    if not topk:
        return 0.0
    hits = 0
    for s in topk:
        if s in mic_map and mic_map[s] <= threshold:
            hits += 1
    return hits / len(topk)


def enrichment_factor(precision: float, k: int, mic_map: Dict[str, float], threshold: float) -> float:
    # EF@k = (positives_in_topk / k) / (N_pos / N_total)
    N_total = len(mic_map)
    N_pos = sum(1 for v in mic_map.values() if v <= threshold)
    if N_pos == 0:
        return float('nan')
    baseline = N_pos / N_total
    return (precision) / baseline


def bootstrap_baseline(mic_map: Dict[str, float], k: int, threshold: float, n_boot=1000, seed=0) -> Tuple[float, float]:
    rng = random.Random(seed)
    seqs = list(mic_map.keys())
    stats = []
    N_total = len(seqs)
    for _ in range(n_boot):
        sample = rng.sample(seqs, k)
        hits = sum(1 for s in sample if mic_map[s] <= threshold)
        stats.append(hits / k)
    arr = np.array(stats)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-json", type=Path, required=True)
    p.add_argument("--mic-seqs", type=Path, required=True)
    p.add_argument("--mic-npy", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=10.0, help="MIC cutoff to call active (lower is better)")
    p.add_argument("--ks", type=int, nargs="+", default=[1,5,10,50,100])
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    mic_map = load_mic(args.mic_seqs, args.mic_npy)
    gen_seqs = load_generated(args.gen_json)

    # restrict to unique generated sequences preserving order
    seen = set()
    uniq_gen = [s for s in gen_seqs if not (s in seen or seen.add(s))]

    # among generated, find those that are present in mic_map and sort by mic ascending
    present = [s for s in uniq_gen if s in mic_map]
    present_sorted = sorted(present, key=lambda x: mic_map[x])

    results = {
        "n_generated_total": len(gen_seqs),
        "n_generated_unique": len(uniq_gen),
        "n_generated_present_in_mic_map": len(present),
        "threshold": args.threshold,
        "ks": {},
    }

    for k in args.ks:
        prec = precision_at_k(present_sorted, mic_map, k, args.threshold)
        ef = enrichment_factor(prec, k, mic_map, args.threshold)
        # baseline bootstrap CI
        lo, hi = bootstrap_baseline(mic_map, min(k, len(mic_map)), args.threshold, n_boot=args.n_bootstrap, seed=args.seed)
        results["ks"][k] = {
            "precision@k": prec,
            "EF@k": ef,
            "baseline_precision_bootstrap_95ci": [lo, hi],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
