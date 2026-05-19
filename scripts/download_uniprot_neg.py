"""
Download reviewed UniProt non-AMP sequences and save as FASTA.
Target: 50,000 sequences (len 5-50 AA, no antimicrobial keyword KW-0929).

Usage:
  uv run python scripts/download_uniprot_neg.py
"""
import time
import re
import urllib.request
import urllib.parse
from pathlib import Path

OUT = Path("data/benchmarks/uniprot_neg_50k.fasta")
OUT.parent.mkdir(parents=True, exist_ok=True)

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
TARGET = 50_000
PAGE = 500


def fetch_page(cursor=None):
    query = "reviewed:true AND length:[5 TO 50] NOT keyword:KW-0929"
    url = (UNIPROT + "?query=" + urllib.parse.quote(query)
           + f"&format=fasta&size={PAGE}")
    if cursor:
        url += f"&cursor={cursor}"
    req = urllib.request.Request(url, headers={"User-Agent": "jepa-amp/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        link = r.headers.get("Link", "")
        body = r.read().decode()
    next_cursor = None
    if 'rel="next"' in link:
        m = re.search(r'cursor=([^&>]+)', link)
        if m:
            next_cursor = m.group(1)
    return body, next_cursor


def parse_fasta(text: str) -> list[str]:
    seqs, cur = [], []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur:
                s = "".join(cur).upper()
                if 5 <= len(s) <= 50 and all(c in VALID_AA for c in s):
                    seqs.append(s)
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        s = "".join(cur).upper()
        if 5 <= len(s) <= 50 and all(c in VALID_AA for c in s):
            seqs.append(s)
    return seqs


def main():
    all_seqs = []
    cursor = None
    page = 0
    while len(all_seqs) < TARGET:
        try:
            body, cursor = fetch_page(cursor)
        except Exception as e:
            print(f"  fetch error: {e}, stopping")
            break
        seqs = parse_fasta(body)
        all_seqs.extend(seqs)
        page += 1
        print(f"  page {page:3d}: +{len(seqs):4d} → total {len(all_seqs):6d}")
        if not cursor or not seqs:
            break
        time.sleep(0.3)

    all_seqs = list(dict.fromkeys(all_seqs))[:TARGET]  # dedup, cap
    with open(OUT, "w") as f:
        for i, s in enumerate(all_seqs):
            f.write(f">neg_{i}\n{s}\n")
    print(f"\nSaved {len(all_seqs)} sequences → {OUT}")


if __name__ == "__main__":
    main()
