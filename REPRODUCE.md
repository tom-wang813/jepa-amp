REPRODUCE.md

This repository contains scripts and data to reproduce MIC pseudolabel generation,
regressor training/evaluation, and baseline comparisons.

Key files produced in this work:
- data/processed/mic_pseudolabels.npy
- data/processed/mic_pseudolabels_seqs.txt
- eval_results/mic_regressor.joblib
- eval_results/mic_regressor_results.json
- eval_results/mic_regressor_grampa_test_thr2.5.json
- eval_results/mic_regressor_grampa_test_thr3.0.json
- eval_results/mic_regressor_grampa_test_thr3.5.json
- eval_results/mic_enrichment_conditional_test.json
- eval_results/conditional_gen_test.json

Important commands

1) Generate pseudolabels (already produced):

```bash
uv run python scripts/generate_mic_pseudolabels.py --gpu 1
```

2) Train + evaluate MIC regressor on pseudolabels (already produced):

```bash
uv run python scripts/train_and_evaluate_mic_regressor.py \
  --mic-seqs data/processed/mic_pseudolabels_seqs.txt \
  --mic-npy data/processed/mic_pseudolabels.npy \
  --gen-json eval_results/conditional_gen_test.json \
  --pretrain-ckpt checkpoints/jepa_pretrain_868k/last_jepa.pt \
  --out eval_results/mic_regressor_results.json \
  --reg-out eval_results/mic_regressor.joblib \
  --gpu 1
```

3) Evaluate regressor on GRAMPA held-out test (bootstrap CI):

```bash
uv run python scripts/evaluate_regressor_on_grampa_test.py \
  --reg-joblib eval_results/mic_regressor.joblib \
  --pretrain-ckpt checkpoints/jepa_pretrain_868k/last_jepa.pt \
  --gpu 1 --n-bootstrap 1000 --threshold 3.0 \
  --out eval_results/mic_regressor_grampa_test_thr3.0.json
```

4) Baseline comparison (ESM2 / AMPlify / JEPA probe) on GRAMPA test:

```bash
python scripts/evaluate_baselines.py --gpu 0 --n-bootstrap 1000 --out eval_results/baseline_eval_grampa.json
```

Notes and recommendations
- Use `--gpu` to select an available CUDA device (we used `cuda:1` in prior runs). Avoid `cuda:0` if shared.
- If `AMPlify` or `transformers` are not installed, `scripts/evaluate_baselines.py` will fall back to any saved probe at `eval_results/amp_classifier.pkl`.
- For publication-quality results, generate 500–1000 candidate sequences, run all scorers, and compute EF@k curves with 1000+ bootstrap replicates.
