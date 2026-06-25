# Session Log — JEPA-AMP
_Append-only. Most recent entry first._

---
## Session 7 — 2026-06-25
**Phase**: Paper data compilation

### Decisions made
- Compiled all MIC-related results (in-domain, cross-species, blind-2026, MIC-conditioned generation, interpretability) into `paper/MIC_DATA_SUMMARY.md` as a single paper-facing reference.
- Confirmed overnight runs from 2026-06-24 are complete: interpretability probing, fewshot warmstart cross-species, and fewshot bacteria embedding eval.

### What changed
- `paper/MIC_DATA_SUMMARY.md` — NEW: single-file reference for all MIC numbers, source paths, and status. Sections 1–9 cover regression, MC Dropout, cross-species zero-shot and few-shot, blind-2026, MIC-conditioned generation, and new interpretability results.

### Key numbers (verified against artifacts)
| Result | Value | Source |
|---|---|---|
| JEPA Transformer MIC Pearson | 0.640 | `checkpoints/formal_mic_868k_transformer/test_metrics.json` |
| ESM-2 35M MIC Pearson | 0.554 | `checkpoints/formal_esm2_mic/test_metrics.json` |
| Cross-species zero-shot mean | JEPA 0.523 vs ESM-2 35M 0.423 | `cross_species_transfer/metrics.json` |
| Blind-2026 temporal | JEPA 0.552 vs ESM-2 650M 0.596±0.017 | `external_elife2025_supp2_mic.json` |
| MIC broad-spectrum Δ | JEPA −0.250, ESM-2 −0.291 (both agree) | `mic_conditioned_generation_formal/metrics.json` |
| MIC inactive Δ | JEPA +0.515, ESM-2 +0.443 (both agree) | same |
| MIC species selectivity | No separation (negative result) | same |
| JEPA MIC linear probe R² (E.coli) | 0.162 ± 0.079 | `interpretability/jepa/mic_linear_r2.json` |
| ESM-2 35M MIC linear probe R² (E.coli) | 0.037 ± 0.122 | `interpretability/esm2/mic_linear_r2.json` |
| Fewshot warmstart 100-shot mean (JEPA) | 0.505 | `fewshot_cross_species_warmstart/metrics.json` |

### Overnight results (2026-06-24) confirmed
- `eval_results/interpretability/` — MIC linear probing R² + species decodability (3 models)
- `eval_results/fewshot_cross_species_warmstart/` — few-shot adaptation with warm-start (JEPA, ESM2, ESM2-650M, MLM)
- `eval_results/fewshot_bact_emb_jepa/` and `fewshot_bact_emb_esm2_650m/` — bacteria-specific embedding evaluation
- `eval_results/fewshot_v2/` — updated fewshot v2 for all 4 models

### Next session starts with
Identify which sections of `paper/main.tex` or `paper/sections/*.tex` still need the new numbers from this session, and fill them in. Priority: interpretability results (Section 9 of MIC_DATA_SUMMARY) have not yet been integrated into any paper section.

### Agent handoff
Current tool: Claude (FleetView)
Source of truth: `paper/MIC_DATA_SUMMARY.md` (new), `note/RESEARCH_STATE.md`
Safe next command: `ls paper/sections/`
Do not do: rerun any MIC training or evaluation experiments; all MIC evidence is locked
Open decision: how to present ESM-2 650M blind-2026 result given protocol mismatch with JEPA (frozen+head vs fine-tuned)

---
## Session 6 — 2026-06-22
**Phase**: Baseline extension

### Decisions made
- ESM-2 650M added as additional comparison baseline (instead of RAMPMLM, which has no existing checkpoint)
- Cross-species and blind-2026 evaluations run with frozen 650M + MLP head (same protocol as cross_species_transfer.py)

### What changed
- `scripts/cross_species_transfer.py` — `ESM2Embedder` made configurable by model_key; 650M added to evaluation loop as `esm2_650m`
- `scripts/eval_esm2_650m_blind2026.py` — new script: frozen 650M + MLP head trained on GRAMPA E.coli, evaluated on 104-sequence supp2 blind set
- `eval_results/cross_species_transfer/metrics.json` — new key `esm2_650m` (5 species pairs × 3 seeds)
- `eval_results/external_elife2025_supp2_mic.json` — new metric key `ESM-2 650M (frozen+head)`

### Evidence used
| Model | Cross-species zero-shot | Blind-2026 |
|---|---|---|
| JEPA-AMP | mean 0.523 | 0.552 |
| ESM-2 35M | mean 0.423 | 0.321 |
| ESM-2 650M | mean 0.442 | 0.596 ± 0.017 |

### Interpretation
- JEPA still leads ESM-2 35M on both benchmarks
- ESM-2 650M surpasses JEPA on blind-2026 (0.596 vs 0.552) but not on cross-species (0.442 vs 0.523)
- ⚠️ This is a paper risk: a larger frozen ESM-2 can beat JEPA on the temporal held-out test. Needs careful framing — JEPA is 14M vs 650M (46×), and the blind-2026 advantage of 650M disappears on cross-species generalization.
- Protocol note: blind-2026 ESM-2 650M uses frozen embeddings + MLP head (consistent with cross-species protocol), whereas the existing JEPA/35M blind-2026 numbers use fine-tuned checkpoints. This protocol difference should be disclosed if cited together.

### Next session starts with
Decide how to present the 650M comparison in the paper — whether to add it as a supplementary comparison or use it to strengthen JEPA's efficiency argument (14M JEPA ≈ 650M ESM-2 on generalization, better on cross-species).

### Agent handoff
Current tool: Claude (FleetView)
Source of truth: `note/RESEARCH_STATE.md`, `eval_results/cross_species_transfer/metrics.json`, `eval_results/external_elife2025_supp2_mic.json`
Safe next command: `python3 -c "import json; d=json.load(open('eval_results/external_elife2025_supp2_mic.json')); print(d['metrics'])"`
Do not do: re-run blind-2026 with fine-tuned 650M (no checkpoint exists); claim JEPA beats 650M on blind-2026
Open decision: paper framing of 650M comparison (efficiency argument vs. honest failure mode)

---
## Session 5 — 2026-06-12
**Phase**: State synchronization + unfinished-work audit

### Decisions made
- Confirmed there are **no currently running training or evaluation jobs** on this workspace.
- Confirmed that previously queued jobs from the 2026-06-06 status sheet are all finished:
  NAR, Diffusion, v6, charge interpolation, learning curve, and cross-species transfer all have artifacts.
- Confirmed a second experiment line completed on 2026-06-09 to 2026-06-11:
  MLM pretraining, MLM MIC fine-tuning, V7 generation eval, and `comparison_formal`.
- Reclassified unfinished work as **artifact reconciliation and paper integration**, not GPU compute.
- Identified two source-of-truth conflicts that must be resolved before paper use:
  - `eval_results/ttt_transfer/SUMMARY.md` vs `eval_results/ttt_transfer/metrics.json`
  - `eval_results/generation_control_v7/SUMMARY.md` vs `eval_results/comparison_formal/metrics.json`

### What changed
- `note/RESEARCH_STATE.md` — updated to 2026-06-12 state and verified-next-step framing
- `note/EXPERIMENT_STATUS.md` — synchronized old RUNNING/QUEUED items to their actual finished state

### Evidence used
- `logs/pipeline.log`
- `logs/ttt_benchmarks.log`
- `logs/mlm_pretrain_868k.log`
- `logs/mlm_finetune_mic.log`
- `eval_results/comparison_formal/metrics.json`
- `eval_results/ttt_transfer/metrics.json`
- `eval_results/learning_curve/SUMMARY.md`
- `eval_results/cross_species_transfer/SUMMARY.md`
- `eval_results/charge_interpolation/metrics.json`

### Current status snapshot
- All planned experiments visible in old status docs have run to completion.
- `ttt_benchmarks` is only partially closed: metrics were written, but summary generation crashed with a formatting bug.
- `ttt_transfer` has updated metrics on 2026-06-12 after the summary was already written on 2026-06-11.
- The workspace now needs bookkeeping and paper-facing result selection more than additional compute.

### Next session starts with
1. Reconcile `ttt_transfer` summary/metrics and pick the canonical TTT setting for reporting.
2. Reconcile V7 standalone evaluation with `comparison_formal` before quoting any V7 control number.
3. Update paper-facing tables/text or a fresh status memo from verified artifacts only.

### Agent handoff
Source of truth for new side analyses:
`eval_results/comparison_formal/metrics.json`,
`eval_results/ttt_transfer/metrics.json`,
`eval_results/ttt_benchmarks/metrics.json`
Do not trust timestamp-older markdown summaries when a newer machine-readable metrics file exists.

---
## Session 4 — 2026-06-08
**Phase**: Final result integration + writing simplification

### Decisions made
- Confirmed that NAR, Diffusion, and v6 runs all finished and their artifacts exist.
- Added a dedicated v6 evaluation script and config to score 7-dim control targets.
- Set writing preference for the paper: simple wording, easy sentence structure,
  and vocabulary no harder than typical TOEFL level unless a technical term is required.
- Added a stricter writing rule: every section should say only what belongs in that
  section, and every paragraph should stay concrete and low on filler.
- Chosen writing stance for new paradigm results:
  - Diffusion = strongest new paradigm
  - NAR = weaker than AR on charge and weak on length
  - v6 = partial success only (helix/pI somewhat controllable; other added controls weak)

### What changed
- `scripts/evaluate_generation_control.py` — added support for NAR and Diffusion checkpoints
- `scripts/evaluate_generation_control_v6.py` — NEW: formal evaluation for 7-dim control
- `configs/generation_control_nar.yaml` — NEW
- `configs/generation_control_diffusion.yaml` — NEW
- `configs/generation_control_v6.yaml` — NEW
- `configs/finetune_868k_v6.yaml` — corrected to `generator_version: v6`; reduced startup risk
- `src/data/dataset.py` — removed redundant 3-dim precompute when building V6 conditions
- `src/train/finetune.py` — route `v6` through the correct generator/data path
- `eval_results/generation_control_nar/` — NEW formal results
- `eval_results/generation_control_diffusion/` — NEW formal results
- `eval_results/generation_control_v6/` — NEW formal results
- `note/RESEARCH_STATE.md` — updated for final write-up phase and writing constraints

### Evidence used
- `eval_results/generation_control_nar/SUMMARY.md`
- `eval_results/generation_control_diffusion/SUMMARY.md`
- `eval_results/generation_control_v6/SUMMARY.md`
- `checkpoints/generator_868k_v6/best_generator.pt`

### Final numbers added to working memory
- NAR: Charge R²=0.693, GRAVY R²=-0.386, Length R²=-1.155, Unique=0.727
- Diffusion: Charge R²=0.702, GRAVY R²=-0.259, Length R²=1.000, Unique=0.929
- v6: Charge R²=0.659, Helix R²=0.533, pI R²=0.330, AMP score R²=0.237, Unique=0.925

### Next session starts with
Update `paper/sections/results.tex` and nearby discussion text using the final
generation paradigm numbers, while keeping wording simple and easy to read.

### Agent handoff
Source of truth: `note/RESEARCH_STATE.md` + `eval_results/generation_control_{nar,diffusion,v6}/`
Writing rule: prefer plain English; avoid hard vocabulary unless technically necessary
Do not claim: v6 solves GRAVY, hydrophobic moment, or length control

---
## Session 3 — 2026-06-06
**Phase**: Formal experiment + parallel paper writing

### Decisions made
- Added 3 new generation paradigms: NAR decoder, Masked Diffusion decoder, AR v6 (7-dim conditions)
- Extended condition vector from 3→7 dims: added helix propensity (Chou-Fasman), pI, hydrophobic moment, AMP classifier score
- Precomputed AMP scores for all 868k training sequences → `data/processed/amp_scores_cache.json`
- Dropped MC-Dropout claim: formal checkpoint shows Δ=+0.0028 (wrong direction vs old exploratory result)
- Paper sections rewritten: Methods (correct encoder size d=384/8L), Results (new Tables 3-5 with placeholders), Discussion (plasticity insight added)
- Ablation training completed: v4_no_aux (val=2.129) and v4_no_dropout (val=2.470)
- Ablation evaluation completed: no_aux R²=-16.76 (aux loss is critical), no_dropout R²=0.805

### What changed
- `src/models/generator_nar.py` — NEW: NAR decoder (25.16M, bidirectional + length head + iterative refinement)
- `src/models/generator_diffusion.py` — NEW: MDLM masked diffusion decoder (30.44M, 6-layer denoiser + timestep emb)
- `src/train/finetune_nar_diffusion.py` — NEW: training script for NAR and Diffusion
- `src/data/dataset.py` — added `compute_conditions_v6()`, `AMPSeq2SeqDatasetV6`, `build_seq2seq_datasets_v6()`
- `scripts/precompute_amp_scores.py` — NEW: batch-scores 868k seqs with JEPA classifier
- `scripts/evaluate_generation_control.py` — existing, used for ablation eval
- `scripts/umap_embedding_comparison.py` — NEW: 5 UMAP figures (JEPA vs ESM2 vs generated)
- `scripts/embedding_quality_analysis.py` — NEW: k-NN MIC, linear probe, silhouette
- `scripts/mc_dropout_formal.py` — NEW: formal MC-Dropout evaluation
- `scripts/charge_interpolation.py` — NEW: continuous charge sweep (not yet run cleanly)
- `paper/sections/abstract.tex` — rewritten with plasticity insight + 3 paradigms
- `paper/sections/methods.tex` — rewritten: correct arch, NAR/Diffusion/v6 sections
- `paper/sections/results.tex` — rewritten: Tables 3-5 added, `\todo{---}` placeholders
- `paper/sections/discussion.tex` — rewritten: plasticity para, 3-paradigm para, MC-Dropout removed
- `configs/finetune_868k_nar.yaml` — NEW
- `configs/finetune_868k_diffusion.yaml` — NEW
- `configs/finetune_868k_v6.yaml` — NEW
- `configs/generation_control_ablation.yaml` — NEW (includes no_aux + no_dropout variants)
- `eval_results/generation_control_ablation/` — NEW: ablation eval results
- `eval_results/embedding_quality/` — NEW: JEPA vs ESM2 frozen embedding quality
- `eval_results/umap/` — NEW: 5 UMAP figures
- `eval_results/mc_dropout_formal/` — NEW: MC-Dropout negative result
- `note/RESEARCH_STATE.md` — NEW (this session)
- `note/SESSION_LOG.md` — NEW (this session)
- `note/PITFALLS.md` — NEW (this session)
- `docs/PAPER_STATUS.md` — NEW: comprehensive status doc

### Evidence used
- `eval_results/generation_control_ablation/SUMMARY.md` for ablation results
- `eval_results/embedding_quality/metrics.json` for plasticity insight
- `eval_results/mc_dropout_formal/metrics.json` for negative MC-Dropout result
- `paper/EVIDENCE_MANIFEST.md` and `paper/ISMB_EVIDENCE_GAP.md` for paper readiness

### Currently running
- **Job b71d3k5g2**: NAR (epoch ~17/80) → Diffusion → v6, sequential on GPU 0 (RTX 3090)
- GPU status: 72°C, 99% util, 10.6GB/24GB VRAM, 346W/350W

### Next session starts with
Check if job b71d3k5g2 completed. If yes, run generation control evaluation for NAR,
Diffusion, and v6, then fill `paper/sections/results.tex` `\todo{---}` placeholders.
Command: `tail -5 logs/finetune_nar.log && tail -5 logs/finetune_diffusion.log`

### Agent handoff
Current tool: Claude (FleetView)
Source of truth: `note/RESEARCH_STATE.md`, `paper/EVIDENCE_MANIFEST.md`
Safe next command: `tail -5 logs/finetune_nar.log` to check training progress
Do not do: start additional GPU jobs; kill job b71d3k5g2; modify locked artifacts
Do not resume: MC-Dropout improvement claim (negative result, locked)
Open decision: journal target (Bioinformatics now vs Cell Systems after wet lab)

---
## Session 2 — 2026-06-05 (reconstructed from artifacts)
**Phase**: Pre-experiment → Formal experiment

### Decisions made
- Pre-training completed on 868k sequences (JEPA encoder, d=384, 8L)
- AR decoder v4 (dual-pathway AdaLN + cross-attn condition token) trained and evaluated
- QMAP benchmark run: 3 seeds × 3 JEPA variants across 5 homology-aware splits
- Formal MIC reproduction locked: Transformer Pearson 0.640, FiLM-MLP RMSE 0.619
- AMP classification locked: AUROC 0.958, MCC 0.802
- Generation control locked: charge R²=0.866, GRAVY/length control failed
- MIC-conditioned generation locked: global shifts reproduced by ESM2 scorer;
  species selectivity failed (sparse GRAMPA supervision)
- Evidence manifest created: Gate 0 complete, most claims LOCKED

### What changed
- All formal evaluation artifacts in `eval_results/`
- `paper/EVIDENCE_MANIFEST.md` and `paper/ISMB_EVIDENCE_GAP.md` created
- `paper/sections/` — initial draft written

### Next session starts with
Ablation experiments (v4_no_aux, v4_no_dropout) + new generation paradigms

---
## Session 1 — 2026-04-17 (from NOTES.md)
**Phase**: Exploration → Pipeline stabilization

### Decisions made
- Changed finetune task from random-block to prefix→suffix seq2seq
- Expanded corpus from 1627 → 868,724 sequences (AMPSphere added)
- Stable data split protocol (sorted → shuffle → split)
- JEPA encoder upgraded: d=256/6L → d=384/8L

### What changed
- `src/data/dataset.py` — AMPSeq2SeqDataset (prefix→suffix)
- `scripts/prepare_amp_dataset.py` — multi-source corpus builder
- `configs/jepa_pretrain_868k.yaml`, `configs/finetune_868k_v*.yaml`
