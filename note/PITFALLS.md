# Pitfalls & Closed Directions — JEPA-AMP

---

## [Negative_Result] MC-Dropout-MIC-improvement
**Date closed**: 2026-06-06
**Category**: [Logic_Error]
**What was tried**: MC-Dropout inference (T=50 passes) on MIC Transformer model to improve RMSE
**Evidence**: `eval_results/mc_dropout_formal/metrics.json`
  - Standard: RMSE=0.6266, Pearson=0.6405
  - MC-Dropout: RMSE=0.6294, Pearson=0.6369 (worse, not better)
  - Uncertainty-error Pearson: 0.046 (near-zero calibration)
**Why closed**: Old exploratory claim (Δ=-0.026) came from a different non-formal checkpoint.
  Formal checkpoint rerun shows MC-Dropout gives no benefit.
**Conclusion**: The Transformer MIC model's dropout rate (~0.4) is too low/architecture too
  deterministic for MC-Dropout to provide meaningful uncertainty.
**Re-entry condition**: Only if a new model is trained with higher dropout (≥0.5) AND
  explicitly designed for Bayesian uncertainty. Do not revisit for current formal checkpoint.

---

## [Negative_Result] GRAVY-control-generation
**Date closed**: 2026-06-05
**Category**: [Hypothesis_Failed]
**What was tried**: Conditioning the AR decoder on GRAVY (hydrophobicity) target value
**Evidence**: `eval_results/generation_control_formal/metrics.json`
  - Proposed dual-pathway v4: GRAVY R²=0.020, MAE=1.201
  - All baselines: negative GRAVY R²
**Why closed**: GRAVY is entangled with charge residue composition (K/R are both
  cationic and hydrophilic). The model cannot independently control GRAVY while
  also satisfying a charge constraint.
**Conclusion**: Report as limitation; GRAVY control requires either orthogonal
  conditioning basis or decoupled single-property training.
**Re-entry condition**: If v6 7-dim conditioning or diffusion decoder shows GRAVY R²>0.3.

---

## [Negative_Result] Length-control-generation
**Date closed**: 2026-06-05
**Category**: [Hypothesis_Failed]
**What was tried**: Conditioning the AR decoder on target sequence length
**Evidence**: `eval_results/generation_control_formal/metrics.json`
  - Proposed dual-pathway v4: Length R²=-1.71, MAE=5.11
**Why closed**: AR decoder with EOS-based stopping cannot reliably control length via
  soft conditioning alone. Length emerges from residue-level decisions, not global signal.
**Conclusion**: Report as limitation. NAR decoder predicts length explicitly (separate head),
  which may overcome this. Masked Diffusion operates on fixed-length sequences.
**Re-entry condition**: Check NAR length R² after training completes.

---

## [Negative_Result] Species-selective-MIC-conditioning
**Date closed**: 2026-06-05
**Category**: [Hypothesis_Failed]
**What was tried**: MIC-conditioned generation targeting E.coli vs S.aureus selectively
**Evidence**: `eval_results/mic_conditioned_generation_formal/metrics.json`
  - E.coli selective: JEPA Δ=-0.088, ESM2 Δ=-0.136 (both species shift together)
  - S.aureus selective: JEPA Δ=-0.049, ESM2 Δ=-0.051 (near-zero separation)
  - Bootstrap CIs include zero for species separation
**Why closed**: GRAMPA supervision is sparse (~3.5/20 bacteria per peptide) and
  broad-spectrum activity dominates. The model cannot learn species-selective patterns
  from this supervision regime.
**Conclusion**: Report as a data limitation, not an architecture failure. A dataset
  with dense cross-species paired measurements would be needed.
**Re-entry condition**: Never (data problem), unless a new multi-species dataset
  with >10 measurements per sequence is available.

---

## [Scope_Decision] AMPSphere-only-pretraining
**Date closed**: 2026-04-17
**Category**: [Scope_Decision]
**What was tried**: Pre-training only on UniProt reviewed AMPs (1,143 sequences)
**Evidence**: NOTES.md session 1 — fine-tune converged at val=2.34 (only slightly better than random ln(25)=3.22)
**Why closed**: 1,627 sequences was insufficient for JEPA pre-training;
  AMPSphere expanded corpus to 868,724 sequences, enabling real pre-training.
**Conclusion**: Minimum effective corpus for JEPA pre-training on AMPs is ~100k+.
**Re-entry condition**: Never — 868k corpus is the current standard.

---

## [Env_Bug] charge-interpolation-GPU-contention
**Date closed**: 2026-06-06
**Category**: [Env_Bug]
**What was tried**: Running charge_interpolation.py concurrently with ablation training
**Evidence**: Output stuck at 4 lines; bio-repair processes also running on GPU 0
**Why closed**: Multiple GPU jobs competing for 24GB VRAM caused OOM/stall.
  charge_interpolation.py was killed; needs to be rerun after training queue clears.
**Conclusion**: Only one GPU-intensive job at a time on this machine.
**Re-entry condition**: Run after job b71d3k5g2 (NAR→Diffusion→v6) completes.
  Command: `uv run python scripts/charge_interpolation.py 2>&1 | tee logs/charge_interpolation.log`

---

## [Env_Bug] nvidia-smi-GPU1-failure
**Date closed**: 2026-06-06
**Category**: [Env_Bug]
**What was tried**: `nvidia-smi` to monitor GPU temperature and memory
**Evidence**: "Unable to determine the device handle for GPU0000:47:00.0: Unknown Error"
  GPU at bus 0000:47:00.0 shows Model="Unknown", Video BIOS="??.??.??.??.??"
**Why closed**: Server has 2 GPUs. GPU 1 (bus 47:00.0) has hardware fault.
  GPU 0 (bus 01:00.0) is the working RTX 3090 used for training.
**Conclusion**: Always use `nvidia-smi -i 0` or `nvidia-smi -i 1` specifically.
  GPU 0 working normally: 72°C, 99% util, 346W/350W.
**Re-entry condition**: Report to server admin; not our problem to fix.
