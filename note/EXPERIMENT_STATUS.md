# JEPA-AMP Experiment Status Table
_Last updated: 2026-06-12 10:50 JST_

## Legend
- ✅ LOCKED — artifact exists, number in paper
- ⚠ PARTIAL — artifacts exist but the run did not fully close cleanly
- ⚠ VERIFY — results exist but source files disagree and need reconciliation
- ❌ NEGATIVE — completed, negative result (keep as is)
- 📝 TODO — not yet started

---

## Core RQ Results

| # | Experiment | Status | Key Result | Artifact |
|---|---|---|---|---|
| 1 | AMP Classification (JEPA) | ✅ LOCKED | AUROC 0.958, MCC 0.802 | `eval_results/amp_classification_evidence/metrics.json` |
| 2 | AMP Classification (ESM-2 baseline) | ✅ LOCKED | AUROC 0.963 | `eval_results/classifier_benchmark.json` |
| 3 | AMP Classification (AMPlify published) | ✅ LOCKED | AUROC 0.984 | `eval_results/classifier_benchmark.json` |
| 4 | MIC Regression — JEPA Transformer | ✅ LOCKED | Pearson 0.640, RMSE 0.627 | `checkpoints/formal_mic_868k_transformer/test_metrics.json` |
| 5 | MIC Regression — JEPA FiLM-MLP | ✅ LOCKED | Pearson 0.622, RMSE 0.619 | `checkpoints/formal_mic_868k_mlp/test_metrics.json` |
| 6 | MIC Regression — ESM-2 baseline | ✅ LOCKED | Pearson 0.554, RMSE 0.635 | `checkpoints/formal_esm2_mic/test_metrics.json` |
| 7 | QMAP Full E.coli (conditional) | ✅ LOCKED | PCC 0.512 ± 0.009 | `eval_results/qmap_stats_pack/metrics.json` |
| 8 | QMAP High-eff E.coli (conditional) | ✅ LOCKED | PCC 0.388 ± 0.013 (+34% vs prior) | `eval_results/qmap_stats_pack/metrics.json` |
| 9 | QMAP HC50 | ✅ LOCKED | PCC 0.327 ± 0.004 | `eval_results/qmap_stats_pack/metrics.json` |
| 10 | Charge Control — AR v4 (proposed) | ✅ LOCKED | R²=0.866, MAE=2.23 | `eval_results/generation_control_formal/metrics.json` |
| 11 | Charge Control — Ablation no_aux | ✅ LOCKED | R²=−16.76 (aux loss critical) | `eval_results/generation_control_ablation/` |
| 12 | Charge Control — Ablation no_dropout | ✅ LOCKED | R²=0.805 | `eval_results/generation_control_ablation/` |
| 13 | GRAVY Control | ✅ LOCKED | R²=0.020 (NEGATIVE, report as limit) | `eval_results/generation_control_formal/metrics.json` |
| 14 | Length Control | ✅ LOCKED | R²=−1.71 (NEGATIVE) | `eval_results/generation_control_formal/metrics.json` |
| 15 | MIC-conditioned generation | ✅ LOCKED | Δ=−0.250 (broad), selectivity fails | `eval_results/mic_conditioned_generation_formal/` |
| 16 | MC-Dropout MIC | ❌ NEGATIVE | Δ=+0.0028 (no improvement) | `eval_results/mc_dropout_formal/metrics.json` |

---

## New Analyses (Synchronized)

| # | Experiment | Status | Key Result | ETA |
|---|---|---|---|---|
| 17 | NAR decoder training (80 ep) | ✅ DONE | artifacts under `checkpoints/generator_868k_nar/` | 0h |
| 18 | Masked Diffusion training | ✅ DONE | artifacts under `checkpoints/generator_868k_diffusion/` | 0h |
| 19 | AR v6 training (7-dim cond) | ✅ DONE | artifacts under `checkpoints/generator_868k_v6/` | 0h |
| 20 | NAR generation control eval | ✅ DONE | Charge R² 0.693 | 0h |
| 21 | Diffusion generation control eval | ✅ DONE | Charge R² 0.702, Length R² 1.000 | 0h |
| 22 | v6 7-dim conditioning eval | ✅ DONE | Helix R² 0.533, pI R² 0.330 | 0h |
| 23 | Charge interpolation (re-run) | ✅ DONE | Charge sweep R² 0.923; GRAVY sweep R² 0.074 | 0h |
| 24 | Embedding quality (k-NN, probing) | ✅ LOCKED | JEPA k-NN Pearson 0.598 < ESM-2 0.618 | `eval_results/embedding_quality/metrics.json` |
| 25 | **Physicochemical probing** | ✅ DONE | All props R²>0.83 in BOTH models | `eval_results/physicochemical_probe/metrics.json` |
| 26 | **Learning curve (1%–100%)** | ✅ DONE | 100% data: JEPA 0.622 vs ESM-2 0.632 | 0h |
| 27 | **Cross-species transfer** | ✅ DONE | JEPA zero-shot better on all listed routes | 0h |
| 28 | **MLM pretraining (868k)** | ✅ DONE | best val_loss 2.4314 | 0h |
| 29 | **MLM MIC fine-tuning** | ✅ DONE | Test Pearson 0.6085, RMSE 0.6364 | 0h |
| 30 | **TTT benchmark suite** | ⚠ PARTIAL | metrics written; markdown summary crashed | ~0.5h to fix summary bug |
| 31 | **V7 generation eval** | ✅ DONE | standalone summary exists | 0h |
| 32 | **Formal generator comparison** | ✅ DONE | MLM encoder + V4 decoder strongest charge control (R² 0.846) | 0h |
| 33 | **TTT transfer** | ⚠ VERIFY | summary and metrics file are out of sync | ~0.5–1h to reconcile |

---

## Physicochemical Probe Results (✅ 刚完成)

| Property | JEPA-AMP R² | ESM-2 R² | Insight |
|---|---|---|---|
| net_charge | 0.835 | 0.933 | Encoded, directly controllable |
| gravy | **0.921** | 0.989 | **Encoded but generation fails → data correlation problem** |
| helix | 0.903 | 0.988 | Encoded, future control target |
| length | 0.973 | 0.969 | Encoded, NAR has explicit length head |
| mol_weight | 0.965 | 0.963 | Correlated with length |
| aromaticity | 0.919 | 0.981 | Encoded |

**Key insight**: GRAVY failure is NOT a representation problem (R²=0.921). It's a conditioning architecture problem — GRAVY and charge are correlated in training data distribution.

---

## Paper / Integration Status

| Section | Status | Blocking on |
|---|---|---|
| Abstract | ⚠️ partial | sync latest side-analysis choice; not GPU-blocked |
| Introduction | 📝 NEEDS REWRITE | reframe around plasticity + data efficiency |
| Methods | ✅ complete | — |
| Results: RQ1 (Classification) | ✅ complete | — |
| Results: RQ1 (MIC) | ✅ complete | — |
| Results: RQ2 (QMAP) | ✅ complete | — |
| Results: RQ3 (Generation control) | ⚠️ partial | reconcile V7/comparison protocol before quoting new numbers |
| Results: Plasticity analysis | ⚠️ partial | learning-curve and comparison results need integration |
| Results: Phys-chem probing | ⚠️ partial | artifact exists; write-up not synced |
| Results: Cross-species transfer | ⚠️ partial | artifact exists; write-up not synced |
| Discussion | ⚠️ needs update | update GRAVY/TTT interpretation from verified artifacts |

---

## Remaining Time Estimate

```
GPU / experiment work:
  - 0h for currently known queued runs

Bookkeeping / reconciliation:
  - Fix or regenerate TTT benchmark summary: ~0.5h
  - Reconcile TTT transfer summary vs metrics: ~0.5–1h
  - Reconcile V7 standalone eval vs comparison_formal protocol: ~0.5–1h

Paper integration:
  - Update results/discussion from verified artifacts: ~2–4h
  - Broader rewrite / venue shaping: additional ~2–4h
```
