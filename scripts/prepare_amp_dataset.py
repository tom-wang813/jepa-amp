"""
Prepare a canonical AMP corpus from multiple raw FASTA sources.

Outputs:
  - data/processed/amp_corpus.fasta
  - data/processed/amp_source_stats.json
  - data/processed/amp_sequence_manifest.tsv

Usage:
    uv run python scripts/prepare_amp_dataset.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    category: str
    required: bool = False


SOURCE_SPECS = [
    SourceSpec("uniprot_reviewed", RAW_DIR / "uniprot_amps.fasta", "uniprot", required=True),
    SourceSpec("uniprot_trembl", RAW_DIR / "uniprot_amps_trembl.fasta", "uniprot", required=True),
    SourceSpec("apd3", RAW_DIR / "apd3.fasta", "database"),
    SourceSpec("dbaasp", RAW_DIR / "dbaasp_amps.fasta", "database"),
    SourceSpec("amplify", RAW_DIR / "amplify_amps.fasta", "benchmark"),
    SourceSpec("dramp", RAW_DIR / "dramp_amps.fasta", "database"),
    SourceSpec("dramp_general", RAW_DIR / "general_amps.fasta", "dramp"),
    SourceSpec("dramp_natural", RAW_DIR / "natural_amps.fasta", "dramp"),
    SourceSpec("dramp_specific", RAW_DIR / "specific_amps.fasta", "dramp"),
    SourceSpec("dramp_plant", RAW_DIR / "plant_amps.fasta", "dramp"),
    SourceSpec("dramp_candidate", RAW_DIR / "candidate_amps.fasta", "dramp"),
    SourceSpec("dramp_expanded", RAW_DIR / "expanded_amps.fasta", "dramp"),
    SourceSpec("dramp_stapled", RAW_DIR / "stapled_amps.fasta", "dramp"),
    SourceSpec("dramp_synthetic", RAW_DIR / "synthetic_amps.fasta", "dramp"),
    SourceSpec("dramp_patent", RAW_DIR / "patent_amps.fasta", "dramp"),
    SourceSpec("dramp_antimicrobial", RAW_DIR / "Antimicrobial_amps.fasta", "dramp_activity"),
    SourceSpec("dramp_antibacterial", RAW_DIR / "Antibacterial_amps.fasta", "dramp_activity"),
    SourceSpec("dramp_antifungal", RAW_DIR / "Antifungal_amps.fasta", "dramp_activity"),
    SourceSpec("dramp_antiviral", RAW_DIR / "Antiviral_amps.fasta", "dramp_activity"),
    SourceSpec("dramp_anticancer", RAW_DIR / "Anticancer_amps.fasta", "dramp_activity"),
    SourceSpec("dramp_antiparasitic", RAW_DIR / "Antiparasitic_amps.fasta", "dramp_activity"),
    SourceSpec("dramp_insecticidal", RAW_DIR / "Insecticidal_amps.fasta", "dramp_activity"),
    SourceSpec("dramp_anti_sars_cov_2", RAW_DIR / "Anti-SARS-CoV-2_amps.fasta", "dramp_activity"),
    SourceSpec("dramp_anti_gram_positive", RAW_DIR / "Anti-Gram-positive_amps.fasta", "dramp_activity"),
    SourceSpec("dramp_anti_gram_misc", RAW_DIR / "Anti-Gram-_amps.fasta", "dramp_activity"),
    # New large sources
    SourceSpec("ampsphere", RAW_DIR / "ampsphere.fasta", "metagenome"),
    SourceSpec("campr4", RAW_DIR / "campr4.fasta", "database"),
    SourceSpec("satpdb", RAW_DIR / "satpdb_amps.fasta", "database"),
    SourceSpec("ampbenchmark", RAW_DIR / "ampbenchmark.fasta", "benchmark"),
    SourceSpec("hydramp", RAW_DIR / "hydramp_amps.fasta", "benchmark"),
    SourceSpec("amplify_zenodo", RAW_DIR / "amplify_zenodo_amps.fasta", "benchmark"),
]

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def _discover_manual_sources() -> list[SourceSpec]:
    manual_dir = RAW_DIR / "manual"
    if not manual_dir.exists():
        return []

    specs = []
    for path in sorted(manual_dir.glob("*.fasta")):
        specs.append(
            SourceSpec(
                name=f"manual_{path.stem}",
                path=path,
                category="manual",
                required=False,
            )
        )
    return specs


def _normalize_sequence(seq: str) -> str:
    return "".join(seq.upper().split())


def _iter_fasta_records(path: Path):
    """
    Iterate FASTA records while tolerating non-ASCII bytes in some source files.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        yield from SeqIO.parse(handle, "fasta")


def _sniff_status(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "missing", "file does not exist"
    if path.stat().st_size == 0:
        return "empty", "file size is 0 bytes"
    head = path.read_text(errors="ignore")[:512]
    stripped = head.lstrip()
    if not stripped:
        return "empty", "file has no readable content"
    if stripped.startswith(">"):
        return "ok", ""
    if "<html" in stripped.lower() or "<br" in stripped.lower() or "warning" in stripped.lower():
        return "invalid_html", stripped.splitlines()[0][:160]
    return "invalid_format", stripped.splitlines()[0][:160]


def _load_source(spec: SourceSpec, min_len: int, max_len: int) -> tuple[list[dict], dict]:
    status, detail = _sniff_status(spec.path)
    stats = {
        "source": spec.name,
        "path": str(spec.path.relative_to(PROJECT_ROOT)),
        "category": spec.category,
        "required": spec.required,
        "status": status,
        "detail": detail,
        "records_total": 0,
        "records_valid": 0,
        "filtered_non_standard": 0,
        "filtered_too_short": 0,
        "filtered_too_long": 0,
        "duplicates_within_source": 0,
        "filtered_parse_error": 0,
    }
    if status != "ok":
        return [], stats

    seen = set()
    rows = []
    for record in _iter_fasta_records(spec.path):
        stats["records_total"] += 1
        try:
            seq = _normalize_sequence(str(record.seq))
        except Exception:
            stats["filtered_parse_error"] += 1
            continue
        if not seq:
            stats["filtered_too_short"] += 1
            continue
        if any(c not in VALID_AAS for c in seq):
            stats["filtered_non_standard"] += 1
            continue
        if len(seq) < min_len:
            stats["filtered_too_short"] += 1
            continue
        if len(seq) > max_len:
            stats["filtered_too_long"] += 1
            continue
        if seq in seen:
            stats["duplicates_within_source"] += 1
            continue
        seen.add(seq)
        rows.append(
            {
                "sequence": seq,
                "source": spec.name,
                "record_id": record.id,
                "description": record.description,
                "length": len(seq),
            }
        )
    stats["records_valid"] = len(rows)
    return rows, stats


def prepare_dataset(min_len: int = 5, max_len: int = 50) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    source_stats = []
    merged: dict[str, dict] = {}
    all_sources = SOURCE_SPECS + _discover_manual_sources()
    for spec in all_sources:
        rows, stats = _load_source(spec, min_len=min_len, max_len=max_len)
        source_stats.append(stats)
        for row in rows:
            seq = row["sequence"]
            if seq not in merged:
                merged[seq] = {
                    "sequence": seq,
                    "length": row["length"],
                    "sources": [row["source"]],
                    "record_ids": [row["record_id"]],
                    "descriptions": [row["description"]],
                }
            else:
                merged[seq]["sources"].append(row["source"])
                merged[seq]["record_ids"].append(row["record_id"])
                merged[seq]["descriptions"].append(row["description"])

    merged_rows = sorted(merged.values(), key=lambda x: (x["length"], x["sequence"]))
    source_overlap = defaultdict(int)
    for row in merged_rows:
        key = "|".join(sorted(set(row["sources"])))
        source_overlap[key] += 1

    fasta_path = PROCESSED_DIR / "amp_corpus.fasta"
    with open(fasta_path, "w") as f:
        for idx, row in enumerate(merged_rows, start=1):
            source_str = "|".join(sorted(set(row["sources"])))
            f.write(
                f">amp_{idx:06d} len={row['length']} sources={source_str}\n"
                f"{row['sequence']}\n"
            )

    manifest_path = PROCESSED_DIR / "amp_sequence_manifest.tsv"
    with open(manifest_path, "w") as f:
        f.write("sequence_id\tlength\tsources\trecord_ids\tsequence\n")
        for idx, row in enumerate(merged_rows, start=1):
            f.write(
                f"amp_{idx:06d}\t{row['length']}\t{'|'.join(sorted(set(row['sources'])))}\t"
                f"{'|'.join(row['record_ids'])}\t{row['sequence']}\n"
            )

    total_valid_before_global_dedup = sum(s["records_valid"] for s in source_stats)
    report = {
        "filters": {"min_len": min_len, "max_len": max_len, "valid_amino_acids": "".join(sorted(VALID_AAS))},
        "sources": source_stats,
        "summary": {
            "sources_configured": len(all_sources),
            "sources_usable": sum(1 for s in source_stats if s["status"] == "ok"),
            "total_valid_before_global_dedup": total_valid_before_global_dedup,
            "unique_sequences": len(merged_rows),
            "cross_source_duplicates_removed": total_valid_before_global_dedup - len(merged_rows),
            "output_fasta": str(fasta_path.relative_to(PROJECT_ROOT)),
            "output_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        },
        "source_overlap": dict(sorted(source_overlap.items())),
    }

    stats_path = PROCESSED_DIR / "amp_source_stats.json"
    with open(stats_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    report = prepare_dataset()

    print("\n=== AMP Corpus Summary ===")
    print(f"usable sources: {report['summary']['sources_usable']} / {report['summary']['sources_configured']}")
    print(f"valid before global dedup: {report['summary']['total_valid_before_global_dedup']}")
    print(f"unique sequences: {report['summary']['unique_sequences']}")
    print(f"cross-source duplicates removed: {report['summary']['cross_source_duplicates_removed']}")

    print("\n=== Source Status ===")
    for src in report["sources"]:
        print(
            f"{src['source']}: status={src['status']}, total={src['records_total']}, "
            f"valid={src['records_valid']}, short={src['filtered_too_short']}, "
            f"long={src['filtered_too_long']}, non_standard={src['filtered_non_standard']}, "
            f"dup_in_source={src['duplicates_within_source']}"
        )
        if src["detail"]:
            print(f"  detail: {src['detail']}")

    print("\nWrote:")
    print(f"  {report['summary']['output_fasta']}")
    print(f"  {report['summary']['output_manifest']}")
    print("  data/processed/amp_source_stats.json")


if __name__ == "__main__":
    main()
