## JEPA-AMP

JEPA-AMP is a small research codebase for antimicrobial peptide representation learning and generation.

The current pipeline has three stages:

1. Raw multi-source AMP collection into `data/raw/`.
2. Canonical corpus preparation into `data/processed/amp_corpus.fasta` with filtering, deduplication, and source reports.
3. `JEPA` pre-training followed by conditional generation fine-tuning with a `prefix -> suffix` seq2seq objective so the decoder learns realistic stopping behaviour with `EOS`.

## Project Layout

- `src/data/dataset.py`: JEPA pre-train dataset and seq2seq fine-tune dataset
- `scripts/prepare_amp_dataset.py`: merge raw sources into a canonical processed AMP corpus
- `DATA_SOURCES.md`: source inventory, current status, and manual-ingestion notes
- `src/train/pretrain.py`: JEPA pre-training loop
- `src/train/finetune.py`: conditional generator fine-tuning loop
- `src/eval/run_eval.py`: generation metrics + AMP scoring + representation probe
- `src/eval/rep_eval.py`: standalone JEPA-vs-random representation evaluation
- `scripts/download_data.py`: reviewed/unreviewed UniProt AMP download helpers

## Recommended Workflow

```bash
uv run python main.py download-data
uv run python main.py prepare-data
uv run python main.py pretrain --config configs/jepa_pretrain.yaml --gpu 0
uv run python main.py finetune --config configs/finetune.yaml --gpu 0
uv run python main.py eval --gpu 0
uv run python main.py rep-eval --gpu 0
```

For the current `~28k` cleaned public AMP corpus, prefer the smaller research configs:

```bash
uv run python -m src.train.pretrain --config configs/jepa_pretrain_28k.yaml --gpu 0
uv run python -m src.train.finetune --config configs/finetune_28k.yaml --gpu 0
```

You can also run the modules directly:

```bash
uv run python scripts/prepare_amp_dataset.py
uv run python -m src.train.pretrain --config configs/jepa_pretrain.yaml --gpu 0
uv run python -m src.train.finetune --config configs/finetune.yaml --gpu 0
uv run python -m src.eval.run_eval --gpu 0
uv run python -m src.eval.rep_eval --gpu 0
```

## Notes

- Pre-training still uses JEPA-style block masking.
- Fine-tuning now uses a separate seq2seq dataset instead of reusing random masked target blocks.
- The canonical training corpus now comes from `data/processed/amp_corpus.fasta`, not directly from raw FASTA files.
- `data/processed/amp_source_stats.json` is the source-of-truth report for how many usable sequences each raw database contributed.
- Any extra manual FASTA exports can be placed under `data/raw/manual/` and will be merged automatically on the next `prepare-data` run.
- `configs/jepa_pretrain_28k.yaml` and `configs/finetune_28k.yaml` are the recommended moderate-size configs for the current public corpus scale.
- `eval_results/` should be treated as cached artifacts. Re-run evaluation after changing code or checkpoints.
