# Research Question → Data Map

> Focus: **MIC prediction** (generation experiments parked).
> All paths relative to repo root.

---

## RQ1 — Baseline recognition: does JEPA hold up on AMP classification?

**Test sets**
| File | Description |
|------|-------------|
| `eval_results/classifier_benchmark.json` | All variants on `amplify_test` + `apd3_independent` |

**Key numbers (amplify_test — apple-to-apple vs AMPlify)**

| Model | ROC-AUC | F1 | MCC |
|-------|---------|----|-----|
| JEPA-AMPlify-identical (3 k) | **0.9584** | **0.8989** | **0.8016** |
| JEPA-v6 (868 k balanced, no-leak) | 0.8877 | 0.8003 | 0.6351 |
| JEPA-v3 | 0.8855 | 0.7097 | 0.5651 |

**Key numbers (apd3_independent)**

| Model | ROC-AUC | MCC |
|-------|---------|-----|
| JEPA-AMPlify-identical | **0.9640** | **0.8130** |
| JEPA-v6 | 0.8798 | 0.6478 |

**TeX section**: `sections/results/classification.tex` (already exists)

---

## RQ2 — In-domain MIC regression: JEPA ≥ existing methods?

### 2a. GRAMPA benchmark

| File | Description |
|------|-------------|
| `eval_results/mic_regressor_results_v3.json` | JEPA val RMSE/Spearman |
| `eval_results/baseline_eval_grampa.json` | All scorers on GRAMPA test (n=394) |
| `eval_results/mic_regressor_grampa_preds.npz` | Per-sample predictions |

**Key numbers (GRAMPA test, n=394, threshold=3.0)**

| Model | Pearson | RMSE | Spearman |
|-------|---------|------|----------|
| AMPlify | — | — | — |
| JEPA probe | — | — | — |
| JEPA FiLM-MLP | 0.622 | **0.619** | 0.553 |
| JEPA Transformer | **0.640** | 0.627 | **0.561** |
| ESM-2 FiLM-MLP | 0.554 | 0.635 | 0.530 |

*(scorers: AMPlify, JEPA_probe, LR, RF, SVM, GBM — detailed breakdown in `baseline_eval_grampa.json`)*

### 2b. QMAP homology-aware benchmark

| File | Description |
|------|-------------|
| `eval_results/qmap_jepa_head_finetune_seed*/` | 3 seeds × 5 splits |
| `eval_results/supplementary_abcd/A_jepa_vs_mlm/A_summary.json` | JEPA vs MLM aggregate |
| `eval_results/qmap_benchmark_comparison.md` | Leaderboard comparison |

**Key numbers (mean Pearson, 5 splits × 3 seeds)**

| Method | Full E. coli | High-eff E. coli | HC50 |
|--------|-------------|-----------------|------|
| QMAP ESM2 linear | 0.360 | 0.160 | 0.070 |
| Cai et al. 2025 | **0.520** | 0.290 | — |
| JEPA head-only | 0.501 ± 0.002 | 0.372 ± 0.005 | **0.327** |
| JEPA conditional | 0.512 ± 0.009 | **0.388 ± 0.013** | 0.307 |
| MLM head-only | 0.493 ± 0.003 | 0.381 ± 0.008 | — |

**Statistical test (JEPA vs MLM, n=15 paired):** Δ=+0.030, 95% CI [0.006, 0.054], p=0.034
**Statistical test (JEPA vs ESM2, n=15 paired):** Δ=+0.099, 95% CI [0.073, 0.126], p<0.001

**TeX section**: `sections/results/mic.tex` + `sections/results/qmap.tex` (both exist)

---

## RQ3 — Cross-species transfer: zero-shot / few-shot data-efficiency

| File | Description |
|------|-------------|
| `eval_results/cross_species_transfer/metrics.json` | All routes, 3 seeds, zero-shot + in-domain |
| `eval_results/cross_species_transfer/SUMMARY.md` | Human-readable summary |
| `eval_results/cross_species_transfer/transfer_heatmap.png` | Heatmap figure |
| `eval_results/supplementary_abcd/C_statistical_tests/C_statistical_tests.json` | per-route Δ & p-values |

**Zero-shot Pearson (mean over 3 seeds)**

| Route | JEPA | ESM-2 | Δ |
|-------|------|-------|---|
| E. coli → S. aureus | **0.553** | 0.451 | +0.102 |
| E. coli → P. aeruginosa | **0.657** | 0.547 | +0.110 |
| S. aureus → E. coli | **0.384** | 0.300 | +0.084 |
| S. aureus → P. aeruginosa | **0.484** | 0.403 | +0.081 |
| P. aeruginosa → E. coli | **0.536** | 0.417 | +0.119 |

Mean JEPA zero-shot: **0.523** vs ESM-2: **0.423**; permutation p < 0.001 (n=15 pairs)

**TeX section**: `sections/results/cross_species.tex` ← **new**

---

## RQ4 — Homology leakage audit

| File | Description |
|------|-------------|
| `eval_results/supplementary_abcd/D_homology_leakage/D1_cross_species_identity.json` | Train–test pairwise identity |
| `eval_results/supplementary_abcd/D_homology_leakage/D2_mic_train_test_identity.json` | GRAMPA train–test identity |
| `eval_results/supplementary_abcd/D_homology_leakage/D3_identity_cutoff_robustness.json` | Performance at <70% / <80% cutoffs |
| `eval_results/supplementary_abcd/D_homology_leakage/D_SUMMARY.md` | Summary narrative |

**Performance after CD-HIT filtering (mean Pearson, JEPA zero-shot)**

| Route | Unfiltered | <80% cutoff | <70% cutoff |
|-------|-----------|-------------|-------------|
| E. coli → S. aureus | 0.553 | 0.414 (n=137) | 0.424 (n=115) |
| E. coli → P. aeruginosa | 0.657 | 0.501 (n=82) | 0.514 (n=77) |
| S. aureus → E. coli | 0.384 | 0.324 (n=211) | 0.299 (n=197) |
| S. aureus → P. aeruginosa | 0.484 | 0.156 (n=62) | 0.175 (n=54) |
| P. aeruginosa → E. coli | 0.536 | 0.395 (n=355) | 0.391 (n=312) |

**TeX section**: `sections/results/homology.tex` ← **new**

---

## RQ5 — Blind-2026 external test: generalisation to post-2024 data

| File | Description |
|------|-------------|
| `eval_results/external_elife2025_supp2_mic.json` | Primary: n=104 peptides, E. coli, per-sample preds |
| `eval_results/external_elife2025_mic.json` | Secondary: n=28, E. coli ATCC 25922 |
| `eval_results/cumulative_gain_final.png` | Cumulative gain @ k figure |
| `eval_results/error_analysis_seaborn.png` | 4-panel error analysis |

**Performance (supp2, n=104)**

| Model | Pearson | Spearman |
|-------|---------|----------|
| JEPA-AMP | **0.552** | **0.549** |
| ESM-2 | 0.321 | 0.343 |

**Cumulative gain (top-10% most potent, MIC ≤ 2.8 µM)**

| Metric | JEPA-AMP | ESM-2 | Perfect |
|--------|----------|-------|---------|
| AUCG | **0.84** | 0.62 | 0.95 |
| Hit-rate @ top-10% | 36% | 27% | — |
| Hit-rate @ top-20% | 64% | 36% | — |

ΔAUCG = +0.22; permutation p = 0.033 (n=10 000 sign-flip permutations)

**TeX section**: `sections/results/external.tex` ← **new**
