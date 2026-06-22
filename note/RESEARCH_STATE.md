# Research State — JEPA-AMP
_Last updated: 2026-06-12 | Session: 5_

## Central Research Question
Can a JEPA-pretrained AMP encoder serve as a unified backbone for classification,
quantitative MIC prediction, and multi-property conditional generation — and
does its representational plasticity advantage over ESM-2 survive homology-aware
evaluation and three distinct generation paradigms?

## Current Hypothesis
JEPA pre-training produces more adaptable (higher fine-tuning headroom) representations
than ESM-2 masked-token pretraining, even though frozen JEPA embeddings score lower
than frozen ESM-2. Three generation paradigms (AR, NAR, Masked Diffusion) on the same
frozen backbone will show different diversity/fidelity tradeoffs but comparable
charge-control performance given the same dual-pathway conditioning design.
Extended 7-dim conditioning (adding helix propensity, pI, hydrophobic moment, AMP score)
will enable control of additional biologically relevant properties beyond charge.

## Phase
Result synchronization and paper integration

## Workflow
```yaml
work_mode: formal_experiment
writing_target: submission_paper
evidence_gate: sufficient   # core results locked; new side analyses need sync, not reruns
engineering_stage: stable_pipeline
active_directive: >
  Synchronize note/ state with artifacts produced on 2026-06-09 to 2026-06-12,
  identify unfinished work, and estimate remaining time before the next paper update.
blocked_actions:
  - Do not claim MC-Dropout improves MIC prediction (formal result: Δ=+0.0028, wrong direction)
  - Do not run new QMAP seeds (3-seed pack already locked)
  - Do not regenerate data splits
  - Do not overstate weak v6 controls (GRAVY, hydrophobic moment, length)
  - Do not mix `generation_control_v7/` numbers with `comparison_formal/` numbers without checking protocol mismatch
  - Do not quote `ttt_transfer/SUMMARY.md` as final until it is reconciled with `ttt_transfer/metrics.json`
next_action: >
  Reconcile post-2026-06-09 artifact inconsistencies (TTT transfer summary vs metrics,
  V7 standalone eval vs comparison_formal), then update paper-facing tables/text from
  the verified source only.
```

## Current Focus
All queued experiment runs have finished. The current focus is to synchronize
state files, separate finished runs from unfinished write-up work, and avoid
using stale summaries that no longer match the latest machine-readable artifacts.

## Writing Constraints
- Keep the paper readable for non-native English readers.
- Prefer common words over rare academic words when meaning is unchanged.
- Do not exceed typical TOEFL-level vocabulary unless a technical term is necessary.
- Prefer short sentences and direct claims over dense, abstract phrasing.
- When a result is weak or negative, say so plainly.
- Keep each paragraph concrete: state the result, what it means, and stop.
- Avoid filler, scene-setting, and repeated motivation inside Results or Discussion.
- Each section should contain only section-appropriate content.
- Results: numbers, comparisons, and direct findings only.
- Discussion: meaning, limits, and implications only.
- Methods: setup, data, and procedure only.
- Do not mix speculation into Results.

## Model Size Summary
| Model | Total | Trainable | Notes |
|---|---|---|---|
| JEPA Encoder (shared) | 14.23M | frozen | d=384, 8L, nhead=8 |
| AR v4 (3-dim cond) | 25.07M | 10.84M | proposed, charge R²=0.866 |
| AR v6 (7-dim cond) | 25.07M | 10.84M | +helix/pI/HM/AMP_score |
| NAR (3-dim cond) | 25.16M | 10.94M | bidirectional, parallel decode |
| Masked Diffusion (3-dim) | 30.44M | 16.21M | 6-layer denoiser + timestep emb |

## RQ Status
| RQ | Claim | Status | Evidence |
|----|-------|--------|----------|
| RQ1a: AMP classification | JEPA approaches ESM2/AMPlify | ✅ LOCKED | `eval_results/amp_classification_evidence/metrics.json` — AUROC 0.958, MCC 0.802 |
| RQ1b: MIC regression | JEPA > ESM2 Pearson +0.086 | ✅ LOCKED | `checkpoints/formal_mic_868k_transformer/test_metrics.json` — Pearson 0.640 |
| RQ2: QMAP homology-aware | High-eff E.coli +34% over SOTA | ✅ LOCKED | `eval_results/qmap_stats_pack/metrics.json` — 0.388 ± 0.013 |
| RQ3a: Charge control | Dual-pathway R²=0.866 | ✅ LOCKED | `eval_results/generation_control_formal/metrics.json` |
| RQ3b: Ablation | Aux loss critical; dropout aids diversity | ✅ LOCKED | `eval_results/generation_control_ablation/SUMMARY.md` — no_aux R²=-16.76, no_dropout R²=0.805 |
| RQ3c: Representation plasticity | Frozen ESM2>JEPA; fine-tuned JEPA>ESM2 | ✅ LOCKED | `eval_results/embedding_quality/SUMMARY.md` |
| RQ4: NAR paradigm | Charge control weaker than AR; weak length control | ✅ DONE | `eval_results/generation_control_nar/SUMMARY.md` — Charge R² 0.693 |
| RQ4: Diffusion paradigm | Best new paradigm; perfect length control in this eval | ✅ DONE | `eval_results/generation_control_diffusion/SUMMARY.md` — Charge R² 0.702, Length R² 1.000 |
| RQ4: v6 7-dim cond | Partial extra control only | ✅ DONE | `eval_results/generation_control_v6/SUMMARY.md` — Helix R² 0.533, pI R² 0.330, AMP R² 0.237 |
| MC-Dropout | **NEGATIVE**: no improvement from formal ckpt | ✅ LOCKED (negative) | `eval_results/mc_dropout_formal/metrics.json` — Δ=+0.0028 |
| TTT benchmark suite | Metrics computed; summary write crashed | ⚠ PARTIAL | `eval_results/ttt_benchmarks/metrics.json` exists; `logs/ttt_benchmarks.log` ends with formatting `ValueError` |
| TTT transfer | Results exist but summary and metrics are out of sync | ⚠ VERIFY | `eval_results/ttt_transfer/SUMMARY.md` (2026-06-11) vs `metrics.json` (2026-06-12) |
| V7 / MLM comparison | New comparison pack finished | ✅ DONE | `eval_results/comparison_formal/metrics.json` + `SUMMARY.md` |

## Locked Numbers (paper-ready)
| Metric | Value | Artifact |
|---|---|---|
| AMP AUROC | 0.958 [0.949, 0.967] | amp_classification_evidence/metrics.json |
| MIC Pearson (Transformer) | 0.640 | formal_mic_868k_transformer/test_metrics.json |
| MIC RMSE (FiLM-MLP) | 0.619 | formal_mic_868k_mlp/test_metrics.json |
| QMAP Full E.coli | 0.512 ± 0.009 | qmap_stats_pack/metrics.json |
| QMAP High-eff E.coli | 0.388 ± 0.013 | qmap_stats_pack/metrics.json |
| QMAP HC50 | 0.327 ± 0.004 | qmap_stats_pack/metrics.json |
| Charge R² (v4 proposed) | 0.866 | generation_control_formal/metrics.json |
| Charge R² (NAR) | 0.693 | generation_control_nar/metrics.json |
| Charge R² (Diffusion) | 0.702 | generation_control_diffusion/metrics.json |
| Charge R² (v6) | 0.659 | generation_control_v6/metrics.json |
| Helix R² (v6) | 0.533 | generation_control_v6/metrics.json |
| pI R² (v6) | 0.330 | generation_control_v6/metrics.json |
| AMP score R² (v6) | 0.237 | generation_control_v6/metrics.json |
| Charge R² (no_aux ablation) | -16.76 | generation_control_ablation/metrics.json |
| Charge R² (no_dropout ablation) | 0.805 | generation_control_ablation/metrics.json |
| Frozen JEPA k-NN MIC (k=5) | Pearson 0.598 | embedding_quality/metrics.json |
| Frozen ESM2 k-NN MIC (k=5) | Pearson 0.618 | embedding_quality/metrics.json |

## Agreed Next Step
1. Treat all queued experiments as finished unless a specific missing artifact is identified.
2. Reconcile `ttt_transfer` and `comparison_formal` against their machine-readable metrics.
3. Update paper text from verified artifacts only; stale summaries are not source of truth.
4. Keep wording simple and direct in all rewritten paragraphs.

## Open Questions
- Which `ttt_transfer` setting is the intended paper result: 10-step, 50-step, or 100-step TTT?
- Why does `generation_control_v7/SUMMARY.md` disagree with `comparison_formal/metrics.json` for GRAVY and HC50?
- Should TTT results be presented as a negative side study or omitted from the main paper?
- Wet lab: when will MIC results for the 20 generated candidates be available?
- Journal target: Bioinformatics (ready now) vs Cell Systems (needs wet lab)?

## State Sources
- Experiment index: `eval_results/` directories + `EVIDENCE_MANIFEST.md`
- Active config: `configs/generation_control_v7.yaml` or `configs/comparison_formal.yaml` equivalent via resolved artifacts
- Active entrypoint: `scripts/compare_all_generators_formal.py` / `scripts/eval_ttt_transfer.py` / `scripts/eval_ttt_benchmarks.py`
- Output dir: `eval_results/comparison_formal/` and `eval_results/ttt_transfer/`
- Split manifest: `data/splits/split_meta.json`
- Draft: `paper/main.tex` + `paper/sections/*.tex`
- Draft status: `advisor_draft` — core sections exist; note/docs status files are now more stale than the paper placeholders
