"""
Download standard AMP benchmark datasets.

Sources:
  1. AMPlify benchmark (BirolLab/AMPlify) — AMP vs non-AMP balanced test set
  2. APD3 — Antimicrobial Peptide Database FASTA
  3. HemoPI — Hemolytic peptide benchmark (toxicity proxy)
  4. CAMP4 — Collection of Anti-Microbial Peptides

Usage:
  uv run python scripts/download_benchmarks.py
"""

import json
import ssl
import urllib.request
from pathlib import Path

OUT_DIR = Path("data/benchmarks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# SSL context that bypasses certificate verification (for self-signed/expired certs)
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _download(url: str, dest: Path, description: str) -> bool:
    print(f"Downloading {description} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "jepa-amp/1.0"})
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as r:
            dest.write_bytes(r.read())
        print(f"  -> Saved to {dest} ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return False


def _parse_fasta(path: Path, label: int, max_len: int = 200) -> list[dict]:
    records = []
    cur = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    s = "".join(cur).upper()
                    if 3 <= len(s) <= max_len and all(c in VALID_AA for c in s):
                        records.append({"sequence": s, "label": label})
                cur = []
            else:
                cur.append(line)
    if cur:
        s = "".join(cur).upper()
        if 3 <= len(s) <= max_len and all(c in VALID_AA for c in s):
            records.append({"sequence": s, "label": label})
    return records


def _save_json(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    pos = sum(r["label"] == 1 for r in records)
    neg = sum(r["label"] == 0 for r in records)
    print(f"  Saved {len(records)} sequences ({pos} pos, {neg} neg) → {path}")


# ---------------------------------------------------------------------------
# 1. AMPlify benchmark (BirolLab/AMPlify on GitHub)
# ---------------------------------------------------------------------------

def download_amplify_benchmark():
    base = "https://raw.githubusercontent.com/BirolLab/AMPlify/master/data"
    files = {
        "AMPlify_AMP_test_common.fa":         (OUT_DIR / "amplify_test_pos.fasta", 1),
        "AMPlify_non_AMP_test_balanced.fa":   (OUT_DIR / "amplify_test_neg.fasta", 0),
    }
    records = []
    for fname, (dest, lbl) in files.items():
        if _download(f"{base}/{fname}", dest, f"AMPlify {'AMP' if lbl else 'non-AMP'} test"):
            records.extend(_parse_fasta(dest, lbl))

    if records:
        _save_json(records, OUT_DIR / "amplify_benchmark.json")
    return records


# ---------------------------------------------------------------------------
# 2. APD3 (Antimicrobial Peptide Database)
# ---------------------------------------------------------------------------

def download_apd3():
    # Try multiple known URLs
    urls = [
        "https://aps.unmc.edu/downloads/APD_sequence_release_09142020.fasta",
        "https://aps.unmc.edu/AP/database/APD_sequence_release_09142020.fasta",
    ]
    dest = OUT_DIR / "apd3_all.fasta"
    ok = False
    for url in urls:
        if _download(url, dest, "APD3 full database"):
            ok = True
            break

    if ok:
        records = _parse_fasta(dest, label=1, max_len=200)
        if records:
            _save_json(records, OUT_DIR / "apd3_benchmark.json")
        return records
    print("  APD3 unavailable — try manual download from https://aps.unmc.edu/downloads")
    return []


# ---------------------------------------------------------------------------
# 3. HemoPI hemolytic benchmark (toxicity proxy)
# ---------------------------------------------------------------------------

def download_hemopi():
    """
    HemoPI benchmark (Gautam et al., 2014) — hemolytic vs non-hemolytic peptides.
    Dataset used as standard toxicity benchmark in AMP literature.
    """
    base = "https://raw.githubusercontent.com/lanl/hermes/main/data/hemolytic"
    files = {
        "hemolytic_positive.fasta":    (OUT_DIR / "hemopi_pos.fasta", 1),
        "hemolytic_negative.fasta":    (OUT_DIR / "hemopi_neg.fasta", 0),
    }
    records = []
    for fname, (dest, lbl) in files.items():
        if _download(f"{base}/{fname}", dest, f"HemoPI {'hemo' if lbl else 'non-hemo'}"):
            records.extend(_parse_fasta(dest, lbl))

    # Fallback: use UniProt-based non-hemolytic negatives if download fails
    if not records:
        print("  HemoPI not available at this URL — toxicity data needs manual sourcing.")
        print("  Sources: http://crdd.osdd.net/raghava/hemopi/ or DBAASP download")
    else:
        _save_json(records, OUT_DIR / "hemopi_benchmark.json")
    return records


# ---------------------------------------------------------------------------
# 4. CAMP4
# ---------------------------------------------------------------------------

def download_camp4():
    urls = [
        "http://www.camp3.bicnirrh.res.in/camp_database/AMP_FASTAs.fasta",
        "https://www.camp3.bicnirrh.res.in/camp_database/AMP_FASTAs.fasta",
    ]
    dest = OUT_DIR / "camp4_all.fasta"
    for url in urls:
        if _download(url, dest, "CAMP4 database"):
            records = _parse_fasta(dest, label=1, max_len=200)
            if records:
                _save_json(records, OUT_DIR / "camp4_benchmark.json")
            return records
    print("  CAMP4 unavailable — SSL issue with camp3.bicnirrh.res.in")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Downloading AMP benchmark datasets")
    print("=" * 60)

    print("\n[1/4] AMPlify benchmark")
    download_amplify_benchmark()

    print("\n[2/4] APD3")
    download_apd3()

    print("\n[3/4] HemoPI (toxicity)")
    download_hemopi()

    print("\n[4/4] CAMP4")
    download_camp4()

    print("\n" + "=" * 60)
    print("Done. Results in:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        pos = sum(d["label"] == 1 for d in data)
        neg = sum(d["label"] == 0 for d in data)
        print(f"  {f.name:40s}  {pos:5d} pos  {neg:5d} neg")
