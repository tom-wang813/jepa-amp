# JEPA-AMP Evidence Manifest

This manifest is the first Gate 0 pass for the manuscript. It links claim-bearing numbers in the current draft to visible artifacts in the repository and marks whether each number is ready for submission use.

Status labels:

- `LOCKED`: artifact path and manuscript number agree well enough for conservative paper use.
- `PARTIAL`: artifact exists, but uncertainty, ablation, or provenance is incomplete.
- `NEEDS_EVIDENCE`: current artifact does not safely support the exact manuscript number or wording.
- `ORACLE_ONLY`: evidence is mediated by a computational scorer and should not imply wet-lab activity.

## Source Files Checked

- Draft sections: `paper/sections/abstract.tex`, `paper/sections/results.tex`, `paper/sections/discussion.tex`, `paper/sections/threats.tex`
- Classification artifact: `eval_results/classifier_benchmark.json`
- Classification prediction/CI artifact: `eval_results/amp_classification_evidence/metrics.json`, `eval_results/amp_classification_evidence/predictions.jsonl`
- MIC logs: `logs/mic_transformer.log`, `logs/mic_mlp.log`, `logs/esm2_mic.log`, `logs/mic_mc_dropout.log`
- QMAP summary: `eval_results/qmap_benchmark_comparison.md`
- QMAP statistics pack: `eval_results/qmap_stats_pack/metrics.json`, `eval_results/qmap_stats_pack/SUMMARY.md`, `paper/figures/qmap_benchmark_summary.png`
- QMAP artifacts: `eval_results/qmap_jepa_*`
- QMAP official package metadata/leaderboard: `.venv/lib/python3.11/site-packages/qmap_benchmark-0.1.1.dist-info/METADATA`
- Generation artifacts: `eval_results/generation_control_formal/metrics.json`, `eval_results/generation_control_formal/predictions.jsonl`, `eval_results/mic_conditioned_generation_formal/metrics.json`
- Split provenance: `data/splits/split_meta.json`, `data/splits/train.tsv`, `data/splits/val.tsv`, `data/splits/test.tsv`

## Manuscript Number Audit

| Manuscript claim/number | Draft location | Current artifact | Agreement | Status | Action |
|---|---|---|---|---:|---|
| AMP classification AUROC 0.958, MCC 0.802 | Abstract; Results Table 1 | `eval_results/classifier_benchmark.json`: `JEPA-AMPlify-identical` ROC-AUC 0.9584, MCC 0.8016; `eval_results/amp_classification_evidence/metrics.json`: AUROC CI [0.9488, 0.9670], MCC CI [0.7713, 0.8302] | Matches with local bootstrap CI | `LOCKED` | Keep CI scoped to locally scored JEPA predictions; published baselines remain aggregate-only. |
| ESM2 AMPlify-identical AUROC 0.963 | Results Table 1 | `eval_results/classifier_benchmark.json`: `ESM2-AMPlify-identical` ROC-AUC 0.9631 | Matches | `LOCKED` | Keep wording as "does not surpass". |
| Published AMPlify ensemble AUROC 0.984 | Results Table 1 | `eval_results/classifier_benchmark.json`: AMPlify ensemble ROC-AUC 0.9837 | Matches | `LOCKED` | Keep citation/provenance to published source. |
| APD3 independent AUROC 0.944, MCC 0.758 | Results text | `eval_results/classifier_benchmark.json`: AUROC 0.9444, MCC 0.7576; `eval_results/amp_classification_evidence/metrics.json`: AUROC CI [0.9315, 0.9567], MCC CI [0.7237, 0.7928] | Matches with local bootstrap CI | `LOCKED` | Keep as cross-dataset screen, not an isolated proof of model superiority. |
| GRAMPA MIC Transformer Pearson 0.640, RMSE 0.627 | Abstract; Results Table 2 | `checkpoints/formal_mic_868k_transformer/test_metrics.json`; predictions and manifest in the same directory | Matches formal rerun | `LOCKED` | Use as the highest-correlation JEPA MIC result. |
| GRAMPA MIC MLP Pearson 0.622, RMSE 0.619 | Results Table 2 | `checkpoints/formal_mic_868k_mlp/test_metrics.json`; predictions and manifest in the same directory | Matches formal rerun | `LOCKED` | Use as the lowest-RMSE JEPA MIC result. |
| ESM2 MIC Pearson 0.554, RMSE 0.635 | Results Table 2 | `checkpoints/formal_esm2_mic/test_metrics.json`; predictions and manifest in the same directory | Matches formal rerun | `LOCKED` | Use as the formal ESM-2 baseline. |
| MC-Dropout improves Transformer RMSE by 5.4% / delta -0.026 | **INVALIDATED** — old exploratory checkpoint | `eval_results/mc_dropout_formal/metrics.json`: formal checkpoint gives Δ=+0.0028 (+0.4%, worse not better); uncertainty-error Pearson=0.046 (p=0.039, near-zero) | Contradicts prior claim | `NEEDS_EVIDENCE` | **Do not claim MC-Dropout improves MIC prediction.** Old result was from a different (non-formal) checkpoint. Remove or demote to limitation. |
| QMAP full E. coli PCC 0.512 and high-efficiency PCC 0.388 | Abstract; Results Table 3 | `eval_results/qmap_benchmark_comparison.md`: conditional seed stability Full 0.5122 +/- 0.0086; High-eff 0.3877 +/- 0.0130; `eval_results/qmap_stats_pack/metrics.json` confirms seed means and paired deltas | Matches rounded multi-seed summary | `LOCKED` | Keep as seed-stable homology-aware evidence; avoid claiming clear full-E. coli superiority over Cai et al. 2025. |
| QMAP HC50-specific PCC 0.327 | Abstract; Results Table 3 | `eval_results/qmap_benchmark_comparison.md`: HC50 head-only 0.3273 +/- 0.0035; `eval_results/qmap_stats_pack/metrics.json`: conditional-minus-HC50-head paired delta -0.0208 | Matches | `LOCKED` | Keep as HC50-specific head claim, not shared-head claim. |
| QMAP conditional seed-42 table values 0.514/0.384/0.307 | Results Table 3 | `eval_results/qmap_benchmark_comparison.md`: conditional seed 42 Full 0.5139, High-eff 0.3840, HC50 0.3068 | Matches | `LOCKED` | Decide whether table should show seed-42 or multi-seed mean; avoid mixing without label. |
| QMAP paired split-seed deltas +0.011 Full E. coli, +0.015 high-efficiency E. coli, -0.021 HC50 shared-vs-specific | Results QMAP text | `eval_results/qmap_stats_pack/metrics.json`: conditional_minus_head_full_ecoli mean 0.0111; conditional_minus_head_high_eff_ecoli mean 0.0152; conditional_minus_hc50_head mean -0.0208 | Matches | `LOCKED` | Use as boundary evidence for shared conditioning helping bacterial MIC more than HC50. |
| QMAP prior baselines 0.360/0.160/0.070, 0.510/0.220, 0.520/0.290 | Results Table 3 | `eval_results/qmap_benchmark_comparison.md`; `.venv/lib/python3.11/site-packages/qmap_benchmark-0.1.1.dist-info/METADATA` leaderboard | Matches official package metadata/leaderboard for qmap-benchmark 0.1.1 | `LOCKED` | Keep wording as protocol-aligned leaderboard comparison; paired tests are unavailable without prior raw predictions. |
| Charge control R2 0.866 | Results text | `eval_results/generation_control_formal/metrics.json`: proposed dual-pathway decoder charge R2 0.8658, MAE 2.230 | Matches formal artifact | `LOCKED` | Use as the formal charge-control claim. |
| Weak-conditioning and AdaLN-only baselines have negative charge R2 | Results Table 4/text | `eval_results/generation_control_formal/metrics.json`: weak-conditioning R2 -11.83, AdaLN-only R2 -17.90 | Matches formal artifact | `LOCKED` | Use as evidence of condition-invariant baselines, not full component necessity. |
| Generated charge table values | Results Table 4 | `eval_results/generation_control_formal/metrics.json` and `predictions.jsonl` contain per-target generated values | Matches formal artifact | `LOCKED` | Main table uses representative rows; full artifact contains all 10 conditions. |
| GRAVY and length control remain limited | Results text and discussion | `eval_results/generation_control_formal/metrics.json`: proposed GRAVY R2 0.020, length R2 -1.71 | Matches formal artifact | `LOCKED` | Keep as limitation/boundary claim. |
| MIC inactivation shift +0.515 JEPA / +0.443 ESM2 log2 units | Abstract; Results MIC-generation section | `eval_results/mic_conditioned_generation_formal/metrics.json`: inactive E. coli vs neutral control | Matches formal artifact | `ORACLE_ONLY` | State as computational scorer shift only. |
| Broad-spectrum potency shift -0.250 JEPA / -0.291 ESM2 log2 units | Abstract; Results MIC-generation section | `eval_results/mic_conditioned_generation_formal/metrics.json`: broad E. coli vs AMP-like physicochemical control | Matches formal artifact | `ORACLE_ONLY` | Keep as model-predicted global activity steering. |
| Species-selective control fails | Results and discussion | `eval_results/mic_conditioned_generation_formal/metrics.json`: E. coli-selective and S. aureus-selective scenarios do not separate target species under JEPA or ESM2 scorer | Supported | `LOCKED` | Keep visible as negative result. |
| Generated peptides are not exact AMP-database copies | Results MIC-generation section | `eval_results/generated_peptide_plausibility_formal/metrics.json`: combined GRAMPA/APD3/DRAMP/UniProt exact-match fraction 0.000; nearest-neighbor identity proxy median 0.526, p95 0.688; per-set exact-match fractions all 0.000 | Matches formal artifact | `LOCKED` | Use as multi-database near-neighbor plausibility screen, not proof of chemical novelty. |
| Generated peptides pass a computational HC50 proxy for prioritisation | Results MIC-generation section | `eval_results/generated_peptide_plausibility_formal/metrics.json`: mean predicted log10 HC50 2.303; fraction >=2.0 is 0.847 | Matches formal artifact | `ORACLE_ONLY` | Keep as toxicity proxy only. |
| Bootstrap uncertainty for generation claims | Results MIC-generation section | `eval_results/generated_evidence_bootstrap/metrics.json`: MIC deltas exclude zero for broad/inactive global activity; selectivity intervals include zero | Matches formal artifact | `LOCKED` | Use intervals as generated-sample uncertainty only, not model-training or assay uncertainty. |

## Immediate Blockers

1. MIC-conditioned generation now has an independent ESM2 scorer, but remains `ORACLE_ONLY` until wet-lab or stronger external validation exists.
2. QMAP table should consistently report either seed-42 values or multi-seed means; the abstract currently uses multi-seed rounded values while the table includes seed-42 rows.

## Gate 0 Completion Criteria

Gate 0 is complete when every row above has:

- one selected artifact path,
- one selected config or command source,
- one selected split source,
- one machine-readable metrics file,
- one prediction file where paired or bootstrap analysis is needed,
- and a status of either `LOCKED` or explicitly removed/demoted from the manuscript.

## Gate 0 Infrastructure Added

The MIC training entrypoints now write evidence-lock artifacts after test evaluation:

- JEPA MIC: `src/train/finetune_supervised.py`
- ESM MIC: `src/train/train_esm_supervised.py`

Each formal MIC run should now produce:

- `test_metrics.json`
- `test_predictions.jsonl`
- `config_resolved.yaml`
- `run_manifest.json`

This infrastructure made the formal rerun auditable; the completed artifacts are listed below.

## Formal MIC Reproduction Completed

The formal rerun completed and replaces the conflicting cached MIC numbers:

| Model | Artifact | Pearson | RMSE | MAE | Spearman | n |
|---|---|---:|---:|---:|---:|---:|
| JEPA-AMP Transformer | `checkpoints/formal_mic_868k_transformer/test_metrics.json` | 0.640 | 0.627 | 0.488 | 0.561 | 2,057 |
| JEPA-AMP FiLM-MLP | `checkpoints/formal_mic_868k_mlp/test_metrics.json` | 0.622 | 0.619 | 0.489 | 0.553 | 2,057 |
| ESM-2 (35M) FiLM-MLP | `checkpoints/formal_esm2_mic/test_metrics.json` | 0.554 | 0.635 | 0.495 | 0.530 | 2,059 |

The manuscript MIC table has been revised to use these locked artifacts.

## Observation

The classification, QMAP, formal MIC prediction, physicochemical generation-control, MIC-conditioned generation, and generated-peptide plausibility numbers are now mostly traceable. Generation remains computational-only evidence.

## Interpretation

The paper can aim upward with the current computational evidence, but biological activity wording must remain conservative until wet-lab or stronger external validation exists.

## Next Action

Add figure/table compression for the target venue template and, if time permits, bootstrap uncertainty for the classification and generated-peptide screens.
