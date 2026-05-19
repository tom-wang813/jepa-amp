"""
Generate non-AMP negatives by shuffling AMP sequences.
AA composition preserved, sequence structure/function destroyed.
This is the standard approach used by AMPlify, iAMPpred, etc.

Usage:
  uv run python scripts/make_shuffled_neg.py [--n 50000]
"""
import argparse
import random
from pathlib import Path

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
AMP_FASTA = Path("data/processed/amp_corpus.fasta")
OUT_FASTA  = Path("data/benchmarks/shuffled_neg.fasta")


def load_fasta(path: Path, max_len: int = 50) -> list[str]:
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    s = "".join(cur).upper()
                    if 5 <= len(s) <= max_len and all(c in VALID_AA for c in s):
                        seqs.append(s)
                cur = []
            else:
                cur.append(line)
    if cur:
        s = "".join(cur).upper()
        if 5 <= len(s) <= max_len and all(c in VALID_AA for c in s):
            seqs.append(s)
    return seqs


def shuffle_seq(seq: str, rng: random.Random) -> str:
    aa = list(seq)
    rng.shuffle(aa)
    return "".join(aa)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"Loading AMP sequences from {AMP_FASTA} …")
    amp_seqs = load_fasta(AMP_FASTA)
    amp_set  = set(amp_seqs)
    print(f"  Loaded {len(amp_seqs)} AMP sequences")

    shuffled = set()
    attempts = 0
    max_attempts = args.n * 20
    while len(shuffled) < args.n and attempts < max_attempts:
        src = rng.choice(amp_seqs)
        s = shuffle_seq(src, rng)
        if s not in amp_set and s not in shuffled:
            shuffled.add(s)
        attempts += 1

    shuffled = list(shuffled)[:args.n]
    print(f"  Generated {len(shuffled)} unique shuffled negatives")

    OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FASTA, "w") as f:
        for i, s in enumerate(shuffled):
            f.write(f">shuffled_neg_{i}\n{s}\n")
    print(f"  Saved → {OUT_FASTA}")


if __name__ == "__main__":
    main()
