"""
Sequence quality metrics for evaluating generated AMP sequences.

Provides: validity, uniqueness, novelty, diversity, aa_frequency,
and physicochemical_stats. All functions take plain amino-acid strings
(no special tokens).
"""

import random
import math
from collections import Counter

from src.data.tokenizer import AMINO_ACIDS

# ---------------------------------------------------------------------------
# Kyte-Doolittle hydrophobicity scale
# ---------------------------------------------------------------------------
KD_SCALE: dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# Charged residues at pH 7
POSITIVE_AA = set("KR")
NEGATIVE_AA = set("DE")
HYDROPHOBIC_AA = set("AVILMFWP")


# ---------------------------------------------------------------------------
# Edit distance (Wagner-Fischer DP, no external dependency)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance via DP."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Use two-row rolling array for O(min(la,lb)) memory
    if la < lb:
        a, b, la, lb = b, a, lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def validity(sequences: list[str]) -> float:
    """
    Fraction of sequences where every character is a standard amino acid
    and length >= 3.
    """
    if not sequences:
        return 0.0
    aa_set = set(AMINO_ACIDS)
    valid = sum(
        1 for s in sequences
        if len(s) >= 3 and all(c in aa_set for c in s)
    )
    return valid / len(sequences)


def uniqueness(sequences: list[str]) -> float:
    """Fraction of sequences that are unique (unique / total)."""
    if not sequences:
        return 0.0
    return len(set(sequences)) / len(sequences)


def novelty(sequences: list[str], train_sequences: set[str]) -> float:
    """
    Fraction of sequences that do not appear in the training set.
    Comparison is case-insensitive (both sides uppercased at call site by
    convention, but we upper() defensively here).
    """
    if not sequences:
        return 0.0
    train_upper = {s.upper() for s in train_sequences}
    novel = sum(1 for s in sequences if s.upper() not in train_upper)
    return novel / len(sequences)


def diversity(sequences: list[str], n_sample: int = 1000) -> float:
    """
    Average pairwise normalised edit distance over a random sample of pairs.
    Normalised edit distance = edit_distance(a, b) / max(len(a), len(b)).
    Returns 0.0 if fewer than 2 sequences are provided.
    """
    seqs = [s for s in sequences if len(s) >= 1]
    n = len(seqs)
    if n < 2:
        return 0.0

    # Build a list of candidate index pairs
    # For large n, sample random pairs; for small n, enumerate all
    max_pairs = n * (n - 1) // 2
    k = min(n_sample, max_pairs)

    if max_pairs <= n_sample:
        # enumerate all pairs
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        seen: set[tuple[int, int]] = set()
        pairs = []
        attempts = 0
        while len(pairs) < k and attempts < k * 10:
            i = random.randrange(n)
            j = random.randrange(n)
            if i != j:
                p = (min(i, j), max(i, j))
                if p not in seen:
                    seen.add(p)
                    pairs.append(p)
            attempts += 1

    total = 0.0
    for i, j in pairs:
        a, b = seqs[i], seqs[j]
        denom = max(len(a), len(b))
        total += _edit_distance(a, b) / denom if denom > 0 else 0.0

    return total / len(pairs) if pairs else 0.0


def aa_frequency(sequences: list[str]) -> dict[str, float]:
    """
    Compute per-amino-acid frequency across all sequences.
    Returns a dict mapping each of the 20 standard AAs to its fraction
    of all characters. Only standard AA characters are counted.
    """
    aa_set = set(AMINO_ACIDS)
    counts: Counter = Counter()
    total = 0
    for s in sequences:
        for c in s:
            if c in aa_set:
                counts[c] += 1
                total += 1

    if total == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}

    return {aa: counts.get(aa, 0) / total for aa in AMINO_ACIDS}


def physicochemical_stats(sequences: list[str]) -> dict:
    """
    Compute a set of physicochemical statistics over a list of sequences.

    Returns a dict with:
      - mean_length         : mean sequence length
      - std_length          : std of sequence lengths
      - mean_charge         : mean net charge at pH 7
                              (K/R: +1, D/E: -1, H: +0.1)
      - mean_hydrophobicity : mean Kyte-Doolittle hydrophobicity,
                              summed and divided by sequence length
      - fraction_charged    : fraction of all residues that are K/R/D/E
      - fraction_hydrophobic: fraction of all residues that are A/V/I/L/M/F/W/P
    """
    if not sequences:
        return {
            "mean_length": 0.0,
            "std_length": 0.0,
            "mean_charge": 0.0,
            "mean_hydrophobicity": 0.0,
            "fraction_charged": 0.0,
            "fraction_hydrophobic": 0.0,
        }

    lengths = [len(s) for s in sequences]
    n = len(lengths)
    mean_len = sum(lengths) / n
    variance = sum((l - mean_len) ** 2 for l in lengths) / n
    std_len = math.sqrt(variance)

    charges = []
    hydros = []
    total_res = 0
    charged_res = 0
    hydrophobic_res = 0

    for s in sequences:
        charge = 0.0
        hydro_sum = 0.0
        for c in s:
            if c in POSITIVE_AA:
                charge += 1.0
            elif c in NEGATIVE_AA:
                charge -= 1.0
            elif c == "H":
                charge += 0.1
            kd = KD_SCALE.get(c, 0.0)
            hydro_sum += kd
            total_res += 1
            if c in {"K", "R", "D", "E"}:
                charged_res += 1
            if c in HYDROPHOBIC_AA:
                hydrophobic_res += 1
        charges.append(charge)
        hydros.append(hydro_sum / len(s) if len(s) > 0 else 0.0)

    mean_charge = sum(charges) / n
    mean_hydrophobicity = sum(hydros) / n
    fraction_charged = charged_res / total_res if total_res > 0 else 0.0
    fraction_hydrophobic = hydrophobic_res / total_res if total_res > 0 else 0.0

    return {
        "mean_length": mean_len,
        "std_length": std_len,
        "mean_charge": mean_charge,
        "mean_hydrophobicity": mean_hydrophobicity,
        "fraction_charged": fraction_charged,
        "fraction_hydrophobic": fraction_hydrophobic,
    }
