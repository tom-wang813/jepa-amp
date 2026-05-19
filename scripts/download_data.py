"""
Download AMP sequences from public sources:
  1. UniProt reviewed AMPs (Swiss-Prot), length ≤ 50  (no cap)
  2. UniProt unreviewed AMPs (TrEMBL), length ≤ 50    (no cap)
  3. APD6 natural AMP FASTA export
  4. DBAASP public mirror subset
  5. DRAMP natural AMPs
  6. AMPSphere v.2022-03  (Zenodo 4606582, CC BY 4.0, ~863k)
  7. CAMPR4 all-sequences FASTA

Usage:
    uv run python scripts/download_data.py
"""

import gzip
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_if_fasta_like(data: str, out_path: Path, min_records: int = 10) -> int:
    stripped = data.lstrip()
    if stripped.startswith(">") and data.count(">") >= min_records:
        out_path.write_text(data)
        return data.count(">")
    return 0


def fetch_uniprot_amps(
    out_path: Path,
    max_len: int = 50,
    reviewed: bool | None = True,
):
    """Query UniProt REST API for all AMPs with length ≤ max_len (no record cap)."""
    if reviewed is True:
        reviewed_clause = "reviewed:true AND "
        label = "Swiss-Prot"
    elif reviewed is False:
        reviewed_clause = "reviewed:false AND "
        label = "TrEMBL"
    else:
        reviewed_clause = ""
        label = "all UniProt"

    print(f"Fetching AMPs from UniProt ({label}, KW-0929, len≤{max_len})...")
    query = f"keyword:KW-0929 AND {reviewed_clause}length:[1 TO {max_len}]"
    encoded = urllib.parse.quote(query)
    page_size = 500
    all_fasta: list[str] = []
    total = 0
    cursor = None

    try:
        while True:
            if cursor:
                url = (
                    f"https://rest.uniprot.org/uniprotkb/search"
                    f"?query={encoded}&format=fasta&size={page_size}&cursor={cursor}"
                )
            else:
                url = (
                    f"https://rest.uniprot.org/uniprotkb/search"
                    f"?query={encoded}&format=fasta&size={page_size}"
                )
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/plain",
            })
            with urllib.request.urlopen(req, timeout=60) as r:
                link_header = r.headers.get("Link", "")
                data = r.read().decode("utf-8")

            n_page = data.count(">")
            if n_page == 0:
                break
            all_fasta.append(data)
            total += n_page
            print(f"    page: {n_page} seqs (total so far: {total})")

            cursor = None
            if 'rel="next"' in link_header:
                m = re.search(r'cursor=([^&>]+)', link_header)
                if m:
                    cursor = m.group(1)
            if cursor is None:
                break
            time.sleep(0.3)

        combined = "\n".join(all_fasta)
        out_path.write_text(combined)
        print(f"  Saved {total} sequences -> {out_path}")
        return total
    except Exception as e:
        print(f"  UniProt fetch failed: {e}")
        return 0


def fetch_github_amp_fasta(out_path: Path):
    """AMPlify training data (public, BSD-licensed)."""
    urls = [
        "https://raw.githubusercontent.com/bcgsc/AMPlify/main/data/AMPlify_AMP_train_common.fa",
        "https://raw.githubusercontent.com/tlawrence3/Tasmanian/master/data/AMP.fasta",
    ]
    for url in urls:
        print(f"Trying {url} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read().decode("utf-8")
            n = _save_if_fasta_like(data, out_path, min_records=10)
            if n > 0:
                print(f"  Saved {n} sequences -> {out_path}")
                return n
        except Exception as e:
            print(f"  Failed: {e}")
    return 0


def fetch_apd3_fasta(out_path: Path):
    url = "https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta"
    print("Trying APD6 natural AMP FASTA export ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read().decode("utf-8", errors="ignore")
        n = _save_if_fasta_like(data, out_path, min_records=100)
        if n > 0:
            print(f"  Saved {n} sequences -> {out_path}")
            return n
    except Exception as e:
        print(f"  APD6 failed: {e}")
    return 0


def fetch_dbaasp_mirror(out_path: Path):
    url = "https://raw.githubusercontent.com/AliYoussef96/BCPNN-AMP/master/data/AMPs.fasta"
    print("Trying DBAASP public mirror subset ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read().decode("utf-8", errors="ignore")
        n = _save_if_fasta_like(data, out_path, min_records=100)
        if n > 0:
            print(f"  Saved {n} sequences -> {out_path}")
            return n
    except Exception as e:
        print(f"  DBAASP mirror failed: {e}")
    return 0


def fetch_dramp_fasta(out_path: Path):
    url = "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/natural_amps.fasta"
    print("Trying DRAMP natural AMPs ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read().decode("utf-8", errors="ignore")
        n = _save_if_fasta_like(data, out_path, min_records=10)
        if n > 0:
            print(f"  Saved {n} sequences -> {out_path}")
            return n
    except Exception as e:
        print(f"  DRAMP failed: {e}")
    return 0


def fetch_ampsphere_zenodo(out_path: Path) -> int:
    """
    AMPSphere v.2022-03: ~863k non-redundant metagenome-derived AMPs.
    License: CC BY 4.0.  Source: Zenodo record 4606582.
    After our ≤50 AA filter we expect ~200-400k usable sequences.
    """
    record_id = "4606582"
    print(f"Querying Zenodo record {record_id} (AMPSphere)...")
    try:
        api_url = f"https://zenodo.org/api/records/{record_id}"
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            meta = json.loads(r.read())

        dl_url = None
        is_gz = False
        for f in meta.get("files", []):
            fname = f.get("key", f.get("filename", ""))
            lower = fname.lower()
            if "ampsphere" in lower and lower.endswith(".fasta.gz"):
                dl_url = f["links"]["self"]
                is_gz = True
                print(f"  Found gzip FASTA: {fname}")
                break
            if "ampsphere" in lower and lower.endswith(".fasta"):
                dl_url = f["links"]["self"]
                print(f"  Found FASTA: {fname}")
                break

        if not dl_url:
            # fallback: any fasta file in the record
            for f in meta.get("files", []):
                fname = f.get("key", f.get("filename", ""))
                lower = fname.lower()
                if lower.endswith(".fasta.gz"):
                    dl_url = f["links"]["self"]
                    is_gz = True
                    print(f"  Fallback gzip FASTA: {fname}")
                    break
                if lower.endswith(".fasta"):
                    dl_url = f["links"]["self"]
                    print(f"  Fallback FASTA: {fname}")
                    break

        if not dl_url:
            print("  No FASTA file found in Zenodo record. Files available:")
            for f in meta.get("files", []):
                print(f"    {f.get('key', f.get('filename', '?'))}")
            return 0

        print(f"  Downloading {dl_url} (this may take a while)...")
        req2 = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=600) as r:
            raw = r.read()

        data = gzip.decompress(raw).decode("utf-8", errors="ignore") if is_gz else raw.decode("utf-8", errors="ignore")
        n = _save_if_fasta_like(data, out_path, min_records=1000)
        if n > 0:
            print(f"  Saved {n} sequences -> {out_path}")
            return n
        print("  Downloaded data does not look like a valid FASTA.")
        return 0
    except Exception as e:
        print(f"  AMPSphere download failed: {e}")
        return 0


def fetch_satpdb_antimicrobial(out_path: Path) -> int:
    """SATPdb antimicrobial peptides subset (~10k sequences, experimentally validated)."""
    urls = [
        "http://crdd.osdd.net/raghava/satpdb/antimicrobial.fasta",
        "https://webs.iiitd.edu.in/raghava/satpdb/antimicrobial.fasta",
    ]
    print("Trying SATPdb antimicrobial FASTA ...")
    for url in urls:
        print(f"  Trying {url} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read().decode("utf-8", errors="ignore")
            n = _save_if_fasta_like(data, out_path, min_records=100)
            if n > 0:
                print(f"  Saved {n} sequences -> {out_path}")
                return n
        except Exception as e:
            print(f"    Failed: {e}")
    print("  SATPdb not reachable; skip.")
    return 0


def fetch_ampbenchmark(out_path: Path) -> int:
    """AMPBenchmark public dataset (BioGenies, GitHub)."""
    url = "https://raw.githubusercontent.com/BioGenies/AMPBenchmark/main/data/AMPBenchmark_public.fasta"
    print("Trying AMPBenchmark public FASTA ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read().decode("utf-8", errors="ignore")
        n = _save_if_fasta_like(data, out_path, min_records=10)
        if n > 0:
            print(f"  Saved {n} sequences -> {out_path}")
            return n
    except Exception as e:
        print(f"  AMPBenchmark failed: {e}")
    return 0


def _fetch_zenodo_fasta(record_id: str, label: str, out_path: Path) -> int:
    """Generic Zenodo FASTA fetcher using public file URL (no auth required)."""
    print(f"Querying Zenodo record {record_id} ({label})...")
    try:
        api_url = f"https://zenodo.org/api/records/{record_id}"
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            meta = json.loads(r.read())

        target_fname = None
        is_gz = False
        for f in meta.get("files", []):
            fname = f.get("key", f.get("filename", ""))
            lower = fname.lower()
            if lower.endswith(".fasta.gz") or lower.endswith(".fa.gz") or lower.endswith(".faa.gz"):
                target_fname = fname
                is_gz = True
                break
            if lower.endswith(".fasta") or lower.endswith(".fa") or lower.endswith(".faa"):
                target_fname = fname

        if not target_fname:
            print(f"  No FASTA found in record {record_id}.")
            return 0

        dl_url = f"https://zenodo.org/records/{record_id}/files/{urllib.parse.quote(target_fname)}?download=1"
        print(f"  Downloading {target_fname} ...")
        req2 = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=600) as r:
            raw = r.read()

        data = gzip.decompress(raw).decode("utf-8", errors="ignore") if is_gz else raw.decode("utf-8", errors="ignore")
        n = _save_if_fasta_like(data, out_path, min_records=10)
        if n > 0:
            print(f"  Saved {n} sequences -> {out_path}")
            return n
        print("  Downloaded data is not a valid FASTA.")
        return 0
    except Exception as e:
        print(f"  Zenodo {record_id} ({label}) failed: {e}")
        return 0


def fetch_ampsphere_zenodo(out_path: Path) -> int:
    """AMPSphere v.2022-03: ~863k metagenome AMPs (CC BY 4.0, Zenodo 4606582)."""
    return _fetch_zenodo_fasta("4606582", "AMPSphere", out_path)


def fetch_hydramp_zenodo(out_path: Path) -> int:
    """HydrAMP training peptides <25 AA (Zenodo 7420278)."""
    return _fetch_zenodo_fasta("7420278", "HydrAMP", out_path)


def fetch_amplify_zenodo(out_path: Path) -> int:
    """AMPlify training/test FASTA (Zenodo 7320306)."""
    return _fetch_zenodo_fasta("7320306", "AMPlify-Zenodo", out_path)


def fetch_campr4(out_path: Path) -> int:
    """
    CAMPR4: ~24k natural + synthetic AMPs.
    Tries the known bulk-export endpoint; falls back to the older CAMPR3 mirror.
    """
    print("Trying CAMPR4 all-sequences FASTA ...")
    urls = [
        # CAMPR4 bulk export (natural + synthetic combined)
        "http://camp.bicnirrh.res.in/camp_sequences.fasta",
        "http://camp.bicnirrh.res.in/export/all_camp_sequences.fasta",
        # CAMPR3 GitHub mirror (older but publicly archived)
        "https://raw.githubusercontent.com/jspv/camp3_data/master/camp3_sequences.fasta",
    ]
    for url in urls:
        print(f"  Trying {url} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read().decode("utf-8", errors="ignore")
            n = _save_if_fasta_like(data, out_path, min_records=100)
            if n > 0:
                print(f"  Saved {n} sequences -> {out_path}")
                return n
        except Exception as e:
            print(f"    Failed: {e}")
    print("  CAMPR4 not reachable; place a FASTA export at data/raw/campr4.fasta manually.")
    return 0


if __name__ == "__main__":
    total = 0

    n = fetch_uniprot_amps(OUT_DIR / "uniprot_amps.fasta", reviewed=True)
    total += n

    n = fetch_uniprot_amps(OUT_DIR / "uniprot_amps_trembl.fasta", reviewed=False)
    total += n

    n = fetch_apd3_fasta(OUT_DIR / "apd3.fasta")
    total += n

    n = fetch_dbaasp_mirror(OUT_DIR / "dbaasp_amps.fasta")
    total += n

    n = fetch_github_amp_fasta(OUT_DIR / "amplify_amps.fasta")
    total += n

    n = fetch_dramp_fasta(OUT_DIR / "dramp_amps.fasta")
    total += n

    n = fetch_ampsphere_zenodo(OUT_DIR / "ampsphere.fasta")
    total += n

    n = fetch_campr4(OUT_DIR / "campr4.fasta")
    total += n

    n = fetch_satpdb_antimicrobial(OUT_DIR / "satpdb_amps.fasta")
    total += n

    n = fetch_ampbenchmark(OUT_DIR / "ampbenchmark.fasta")
    total += n

    n = fetch_hydramp_zenodo(OUT_DIR / "hydramp_amps.fasta")
    total += n

    n = fetch_amplify_zenodo(OUT_DIR / "amplify_zenodo_amps.fasta")
    total += n

    print(f"\nTotal sequences downloaded: {total}")
    print(f"Files in {OUT_DIR}:")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_dir():
            print(f"  {f.name}/")
            continue
        size = f.stat().st_size
        if size > 0:
            print(f"  {f.name}: {size/1024:.1f} KB")
        else:
            print(f"  {f.name}: EMPTY (skip)")

    manual_dir = OUT_DIR / "manual"
    manual_dir.mkdir(exist_ok=True)
    print(f"\nManual-drop directory ready: {manual_dir}")
