# Session Log — JEPA-AMP
_Append-only. Most recent entry first._

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
