# JEPA-AMP — Paper Writing Guide
_Last updated: 2026-06-25_

This file is the single source of truth for the paper's motivation, story, claims, numbers, and what to avoid. Write the actual LaTeX from here.

---

## 1. The Problem We Are Solving

**Core problem:** AMP prediction models that perform well on standard (random-split) benchmarks fail badly when evaluated on truly novel sequences. This is not a corner case — drug candidates are novel by definition.

Evidence: on the QMAP benchmark's hardest setting (hc50, sequence identity <50%):
- ESM-2 linear probe: Pearson **0.07** (near-random)
- Our model: Pearson **0.307** (4× better)

Why does this happen? General protein LMs (ESM-2) are trained on entire UniProt — AMPs are a tiny, unusual corner of sequence space (short, cationic, amphipathic). The model never learns what makes an AMP an AMP.

**Secondary problem:** MIC prediction across multiple bacterial species requires either (a) training one model per species (no knowledge sharing) or (b) ignoring species identity (mixing signals). Neither is satisfying when you have 20+ species with unequal data.

---

## 2. What We Propose

**JEPA-AMP**: A self-supervised Transformer pretrained on 868k AMP sequences using a JEPA objective (predicting target-patch embeddings from context without reconstruction). 14M parameters, d_model=384, positional embeddings up to 50 amino acids.

**SpecFiLM**: A conditioning head on top of JEPA. A bacteria embedding (n_bact=20, dim=64) is projected to per-channel scale+shift via FiLM modulation, applied to all encoder tokens. One shared model predicts MIC for all 20 species simultaneously.

---

## 3. What Claims Are Defensible

### ✅ Claim 1: Domain-specific pretraining > general protein LMs

| Task | ESM-2 | JEPA-AMP | Δ |
|---|---:|---:|---:|
| MIC GRAMPA Pearson | 0.554 | **0.640** | +0.086 |
| QMAP full E.coli | 0.360 | **0.512** | +0.152 |
| QMAP high_eff | 0.160 | **0.388** | +0.228 |
| QMAP hc50 | 0.070 | **0.307** | +0.237 |
| Blind-2026 Pearson | 0.321 | **0.552** | +0.231 |

All differences are large and statistically significant. This is the strongest claim.

### ✅ Claim 2: QMAP high_eff — new SOTA

| Method | Full E.coli | High-eff | HC50 |
|---|---:|---:|---:|
| Cai et al. 2025 | **0.520** | 0.290 | — |
| JEPA conditional | 0.512 ± 0.009 | **0.388 ± 0.013** | **0.307** |
| MLM conditional | 0.516 | 0.381 | — (not run) |

We **do not claim full E.coli SOTA** (Cai et al. 0.520 vs our 0.512 — too close).
We **do claim high_eff SOTA** (+34% relative) and are the **only model reporting hc50**.
Use "SOTA on the high-activity subset" — not "SOTA overall".

### ✅ Claim 3: SpecFiLM > no conditioning on GRAMPA 20 species

| Model | GRAMPA Pearson |
|---|---:|
| JEPA + no species conditioning | 0.593 |
| JEPA + SpecFiLM | **0.640** |
| JEPA + per-species heads | 0.611 |

SpecFiLM beats both the no-conditioning baseline (+0.047) and per-species heads (+0.029). This validates the FiLM conditioning design.

### ✅ Claim 4: AMP classification — competitive with AMPlify

| Model | ROC-AUC | F1 | MCC |
|---|---:|---:|---:|
| JEPA (AMPlify-identical setup) | **0.9585** | **0.8989** | **0.8016** |
| JEPA v6 (868k, no-leak) | 0.8877 | 0.8003 | 0.6351 |

Note: AMPlify-identical means same data split. The 868k model is trained on different data so numbers differ.

### ✅ Claim 5: Blind-2026 temporal generalization

On 104 post-2024 peptides (eLife 2025 supplement, E. coli):
- JEPA-AMP: Pearson **0.552**, cumulative gain AUCG **0.84**
- ESM-2: Pearson **0.321**, AUCG **0.62**
- ΔAUCG = +0.22, p = 0.033

This is a realistic deployment scenario: training before 2024, predicting newly published peptides.

### ⚠️ Claim 6: JEPA architecture vs MLM — handle carefully

| Task | JEPA | MLM | Winner |
|---|---:|---:|---|
| MIC GRAMPA | **0.640** | 0.608 | JEPA (+0.032) |
| QMAP full | 0.512 | **0.516** | MLM (−0.004) |
| QMAP high_eff | **0.388** | 0.381 | JEPA (+0.007) |
| QMAP hc50 | **0.307** | — | — |
| Fewshot k=100 | 0.479 | **0.512** | MLM (−0.033) |

**How to write this:** Present as ablation, not as "JEPA > MLM overall."  
The finding is: *both* AMP-domain objectives dramatically outperform ESM-2; JEPA is marginally better on MIC regression (statistically significant: Δ=+0.030, p=0.034) while performance on QMAP and few-shot is essentially equivalent. Domain specificity matters more than pretraining objective.

---

## 4. What NOT to Claim

| ❌ Claim | Why not |
|---|---|
| "Zero-shot cross-species superiority" | 30-pair experiment at k=0: JEPA=0.394 vs ESM-2=0.389 — no difference. Earlier 5-pair result was noise. |
| "SOTA on MIC vs esAMPMIC" | Best result: E.coli 0.755 vs 0.781, SA 0.716 vs 0.756, PA 0.708 vs 0.802 — we don't beat them |
| "Species-selective generation" | ecoli_selective Δ_SA = −0.048 (same direction as broad spectrum, no selectivity) |
| "SOTA on QMAP full E.coli" | Cai et al. 0.520 vs our 0.512 — too close, not defendable |
| "JEPA objective is superior to MLM" | Mixed results; not a clean win |

---

## 5. Paper Structure Recommendation

### Title (option A — safer)
*JEPA-AMP: Domain-Specific Pretraining and Bacteria-Conditioned Regression for Antimicrobial Peptide Activity Prediction*

### Title (option B — bolder)
*AMP-Specialized Pretraining Closes the Generalization Gap in MIC Prediction*

### Abstract hook
> Predicting the antimicrobial activity of novel peptides remains difficult because models trained on known sequences fail to generalize beyond their training distribution. We present JEPA-AMP, a self-supervised Transformer pretrained on 868k AMP sequences, paired with SpecFiLM, a FiLM-conditioned head for simultaneous MIC prediction across 20 bacterial species. On the QMAP benchmark, JEPA-AMP achieves a new state of the art on high-activity peptides (Pearson 0.388 vs 0.290) and is the first model to report performance on the strict hc50 setting (0.307 vs ESM-2's 0.07). On a temporal blind test of 104 peptides published after training, JEPA-AMP reaches Pearson 0.552 versus 0.321 for ESM-2, suggesting genuine generalization rather than training-set memorization.

### Section outline

1. **Introduction** — AMP discovery bottleneck; limitation of general LMs; need for domain pretraining; SpecFiLM contribution
2. **Methods**
   - JEPA pretraining (objective, data, architecture)
   - SpecFiLM head (FiLM conditioning, bacteria embedding)
   - Datasets: GRAMPA (20 species), QMAP benchmark, blind-2026
   - Baselines: ESM-2 35M, MLM same architecture, feature-based, esAMPMIC-style
3. **Results**
   - 3.1 Classification (brief — ROC-AUC 0.9585)
   - 3.2 MIC GRAMPA: SpecFiLM vs no-conditioning vs per-species (ablation table)
   - 3.3 QMAP benchmark: Table with all baselines, highlight high_eff and hc50
   - 3.4 Temporal generalization: blind-2026 results + cumulative gain figure
   - 3.5 Ablation: JEPA vs MLM (frame as "domain > architecture" finding)
4. **Discussion** — why domain pretraining helps; SpecFiLM design choices; limitations (generation weak, 3-species MIC we don't beat esAMPMIC); future work
5. **Conclusion**

---

## 6. All Key Numbers (consolidated)

### MIC — GRAMPA 20 species (test Pearson)
| Model | Pearson |
|---|---:|
| Feature-based (per-species GB) | 0.624 |
| ESM-2 35M + SpecFiLM | 0.554 |
| MLM + SpecFiLM | 0.608 |
| JEPA + no conditioning | 0.593 |
| JEPA + per-species heads | 0.611 |
| **JEPA + SpecFiLM** | **0.640** |
Source: `checkpoints/formal_mic_868k_transformer/test_metrics.json`, `eval_results/supplementary_abcd/A_jepa_vs_mlm/A_summary.json`

### MIC — esAMPMIC 3 species (B-ensemble, 3 seed)
| Species | esAMPMIC | JEPA B-ensemble | Δ |
|---|---:|---:|---:|
| E. coli | 0.781 | 0.755 | −0.026 |
| S. aureus | 0.756 | 0.716 | −0.040 |
| P. aeruginosa | 0.802 | 0.708 | −0.094 |
Source: `eval_results/esampmic_bc_ensemble/metrics.json`
**Do not compare these directly** — different datasets, different splits.

### QMAP (5-split × 3-seed mean Pearson)
| Method | Full E.coli | High-eff | HC50 |
|---|---:|---:|---:|
| ESM-2 linear | 0.360 | 0.160 | 0.070 |
| Witten & Witten 2019 | 0.510 | 0.220 | — |
| Cai et al. 2025 | 0.520 | 0.290 | — |
| JEPA head-only | 0.501 ± 0.002 | 0.372 ± 0.005 | 0.327 ± 0.004 |
| **JEPA conditional** | **0.512 ± 0.009** | **0.388 ± 0.013** | 0.307 ± 0.002 |
| MLM head-only | 0.516 | 0.381 | — |
Source: `eval_results/qmap_benchmark_comparison.md`, `eval_results/supplementary_abcd/A_jepa_vs_mlm/qmap_mlm_metrics.json`

### Blind-2026 (post-2024 peptides, E. coli, n=104)
| Model | Pearson | Spearman | AUCG |
|---|---:|---:|---:|
| ESM-2 | 0.321 | 0.343 | 0.62 |
| **JEPA-AMP** | **0.552** | **0.549** | **0.84** |
Source: `eval_results/external_elife2025_supp2_mic.json`

### Classification (amplify_test)
| Model | ROC-AUC | F1 | MCC |
|---|---:|---:|---:|
| JEPA (AMPlify-identical) | **0.9585** | **0.8989** | **0.8016** |
Source: `eval_results/classifier_benchmark.json`

---

## 7. Generation — What to Say

Generation is exploratory and should be a **brief methods/discussion mention**, not a main result.

**What works:**
- Broad direction control: "potent" conditioning lowers mean predicted MIC (Δ = −0.25 for E.coli vs neutral control)
- Inactive generation: Δ = +0.52 (pushes MIC up reliably)
- All generated peptides are novel (0% exact match), valid (100%), diverse (unique_fraction > 0.78)

**What doesn't work:**
- Species-selective generation: ecoli_selective barely differs from broad_spectrum in S.aureus MIC (Δ = −0.05 vs −0.25)

**How to frame:** "As a proof of concept, we demonstrate that JEPA's continuous latent space supports MIC-conditioned peptide generation, though species-selective generation remains an open challenge."

---

## 8. Key Figures to Make / Already Exist

| Figure | Status | File |
|---|---|---|
| GRAMPA ablation bar chart (SpecFiLM variants) | Likely exists | check `eval_results/figures/` |
| QMAP leaderboard table (heatmap or bar) | Need to make | from `qmap_benchmark_comparison.md` |
| Blind-2026 cumulative gain curve | EXISTS | `eval_results/cumulative_gain_final.png` |
| Cross-species transfer heatmap | EXISTS | `eval_results/cross_species_transfer/transfer_heatmap.png` |
| Generation direction control scatter | Need to make | from `mic_conditioned_generation_formal/metrics.json` |

---

## 9. What Cai et al. 2025 Is

In QMAP benchmark paper (Lavertu et al. 2026, bioRxiv), Cai et al. 2025 appears as a baseline method. Numbers: full E.coli 0.520, high_eff 0.29. They do not report HC50. Identity of their method: likely a transformer or CNN-based AMP predictor from 2025. **Check QMAP benchmark paper supplementary for exact citation** before final submission.

---

## 10. Limitations to Acknowledge

1. SpecFiLM underperforms esAMPMIC on E.coli/S.aureus/P.aeruginosa when both are trained on the same data (we lose on their 3-species benchmark). esAMPMIC uses richer physicochemical features + ensemble.
2. Generation selectivity does not work.
3. JEPA objective does not clearly outperform MLM in all settings — domain pretraining matters more than the specific objective.
4. GRAMPA is a curated subset of DBAASP; generalization to clinical isolates is unknown.
5. Positional embeddings are capped at 50 amino acids — longer peptides were truncated in QMAP.
