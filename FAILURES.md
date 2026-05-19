# Failures

## 2026-04-17 - Raw multi-source AMP corpus not actually populated

Context
- Goal: assemble a multi-database AMP corpus before JEPA pre-training.
- Expected sources included UniProt, APD3, DBAASP, and DRAMP-style raw files.

Execution
- Added `scripts/prepare_amp_dataset.py` and ran it on the current `data/raw/` directory.
- Generated `data/processed/amp_source_stats.json` as the canonical source report.

Insight
- `[Logic_Error]` The training pipeline previously looked multi-source in config, but only two UniProt files were actually usable.
- `apd3.fasta` and `dbaasp_amps.fasta` were empty, `amplify_amps.fasta` was missing, and `dramp_amps.fasta` contained HTML warnings instead of FASTA.
- Result: the processed corpus had only `1627` unique AMP sequences, which is too small to justify claiming meaningful JEPA pre-training.
