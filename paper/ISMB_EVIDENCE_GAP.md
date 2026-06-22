# ISMB Evidence Gap Audit for JEPA-AMP

## Goal

Target the next version of JEPA-AMP at an ISMB/Bioinformatics-level computational biology submission. This audit does not change experiments or paper numbers. It maps each current paper claim to available evidence, the remaining weakness, and the minimum additional experiment needed before the claim should be treated as submission-ready.

## Current Source of Truth

- Draft: `paper/main.tex` and `paper/sections/*.tex`
- Classification evidence: `eval_results/classifier_benchmark.json`
- MIC evidence candidates: `logs/mic_transformer.log`, `logs/mic_mlp.log`, `logs/esm2_mic.log`, `logs/mic_mc_dropout.log`, `configs/mic_868k*.yaml`, `configs/esm2_mic.yaml`
- QMAP evidence: `eval_results/qmap_*` metric summaries and prediction artifacts
- Generation evidence: `eval_results/generation_control_formal/metrics.json`, `eval_results/generation_control_formal/predictions.jsonl`, `eval_results/mic_conditioned_generation_formal/metrics.json`
- Data split provenance: `data/splits/split_meta.json`, `data/splits/train.tsv`, `data/splits/val.tsv`, `data/splits/test.tsv`

## Readiness Verdict

The current draft is no longer just a technical report, but the evidence package is not yet ISMB-ready. The strongest current evidence is the homology-aware QMAP result plus the conservative classification result. MIC prediction and physicochemical generation now have locked formal artifacts, and MIC-conditioned generation has a matched-control plus ESM2 scorer check. The remaining weakness is that generated-peptide activity is still computational rather than biological.

## Claim-Evidence-Gap Map

| Paper claim | Current evidence | Current status | Main gap | Minimum experiment before ISMB-level claim |
|---|---|---:|---|---|
| AMP-domain JEPA gives a compact representation useful for AMP classification. | `classifier_benchmark.json`: JEPA approaches ESM2/AMPlify but does not surpass AMPlify; APD3 transfer is useful. | Paper-eligible with conservative wording | This supports utility, not superiority. Need statistical uncertainty and exact split artifact lock. | Add bootstrap CIs or repeated-seed uncertainty for ROC-AUC/F1/MCC; archive config, split, predictions, and metrics together. |
| JEPA representation transfers to MIC prediction. | Formal artifacts in `checkpoints/formal_mic_868k_transformer`, `checkpoints/formal_mic_868k_mlp`, and `checkpoints/formal_esm2_mic`. | Paper-eligible with conservative wording | Single-split computational benchmark; MC-Dropout remains exploratory. | Add confidence intervals/error-by-range analysis if space permits. |
| MC-Dropout improves MIC inference. | `mic_mc_dropout.log` suggests inference-time RMSE improvement. | Promising but not locked | Needs exact linkage to the paper table and same test set. | Evaluate standard vs MC-Dropout using one formal MIC run artifact; report paired error deltas and CI. |
| JEPA is competitive on homology-aware QMAP E. coli MIC. | QMAP summaries across seeds: full E. coli mean roughly 0.50-0.52; high-efficiency improves; HC50 is weaker without a task-specific head. | Strongest current evidence | Needs statistical presentation and raw-prediction audit; comparison against prior methods should be carefully aligned to available raw artifacts. | Build a QMAP stats pack: per-seed table, mean/SD/CI, paired tests where raw paired predictions exist, and a figure-ready summary. |
| HC50 needs task-specific heads rather than generic MIC representation. | QMAP HC50 head-only summaries improve over generic conditional representation. | Useful boundary claim | Needs wording as a boundary, not a failure or universal conclusion. | Lock HC50 head configs/outputs and report task-specific vs generic head comparison with uncertainty. |
| Physicochemical-conditioned generation controls charge. | `generation_control_formal`: proposed decoder charge R2 0.866, charge MAE 2.23; weaker baselines have negative charge R2. | Locked for bounded charge-control claim | This compares available decoder checkpoints, but does not isolate every training mechanism such as auxiliary loss and context dropout. | Keep wording as "design reduces conditioning collapse"; do not claim every component is necessary without a finer ablation. |
| The generator controls hydrophobicity and length. | Same file shows GRAVY and length control are limited or coupled. | Negative/mixed result only | The paper must not claim reliable hydrophobicity or length control. | Keep as limitation unless ablations show a fix. If tested, report GRAVY MAE and length MAE separately from charge. |
| MIC-conditioned generation controls antimicrobial activity. | `mic_conditioned_generation_formal` shows matched-control shifts under JEPA and ESM2 scorers; selectivity fails. | Computational-only, paper-eligible if bounded | No wet-lab validation; ESM2 scorer reduces but does not eliminate in silico circularity. | Keep wording as model-predicted global activity steering; add nearest-neighbor and toxicity/HC50 screens before stronger candidate claims. |
| Generated peptides are biologically plausible candidates. | `generated_peptide_plausibility_formal` reports exact-match, nearest-neighbor, composition, and QMAP HC50 proxy screens. | Computational-only, paper-eligible if bounded | The screen can flag obvious artifacts but cannot validate synthesis, stability, toxicity, or activity. | Keep as prioritisation evidence; wet-lab or stronger external screens are needed for candidate claims. |
| Method design reduces conditioning collapse. | Current draft explains the design; no full component ablation yet. | Hypothesis only | "Design is consistent with" is safe; "all parts are necessary" is not. | Run decoder ablation suite and report collapse metrics: target-property slope/R2, output diversity, validity/AMP score, and mode collapse indicators. |
| Paper is formatted as a computational biology submission. | Draft now has standard sections but current PDF has overfull boxes and is about 14 pages. | Layout blocker | ISMB proceedings formatting/page constraints are not met yet. | After evidence pass, compress tables, resize architecture/QMAP/generation tables, and convert to the target official template. |

## Minimum ISMB Evidence Plan

### Gate 0: Evidence Lock

Decision changed by this gate: whether current numbers can be cited at all.

Required artifacts:

- A run manifest that maps every paper table entry to config, command, split, seed, output directory, metrics file, prediction file, and log.
- A short `NEEDS_EVIDENCE` list for any number that cannot be traced.
- No new scientific claim should be added until this gate is complete.

### Gate 1: Formal MIC Reproduction

Decision changed by this gate: whether MIC prediction remains a main result or becomes supporting evidence.

Minimum run set:

- JEPA-Transformer MIC model on the locked split.
- JEPA-MLP MIC model on the same split.
- ESM2 MIC baseline on the same split.
- Standard inference and MC-Dropout inference evaluated from the same checkpoint where applicable.

Required metrics:

- RMSE, Pearson, Spearman, MAE, calibration/error by MIC range, and paired error deltas.
- `metrics.json`, predictions, copied config, and run log per model.

### Gate 2: QMAP Statistics Pack

Decision changed by this gate: whether homology-aware generalization can be the central evidence point.

Minimum analysis:

- Aggregate all available seeds for full E. coli, high-efficiency, and HC50.
- Report mean, SD, CI, and paired tests only where raw paired predictions are available.
- Separate claims for E. coli MIC and HC50; do not imply one representation solves both equally.

### Gate 3: Conditional Decoder Ablation

Decision changed by this gate: whether conditional generation is a validated method contribution or a bounded exploratory result.

Minimum variants:

- Weak-conditioning baseline.
- AdaLN-only baseline.
- Cross-attention-only baseline.
- No conditioning dropout.
- No auxiliary physicochemical loss.
- Proposed dual-pathway decoder.

Required metrics:

- Charge MAE/R2, GRAVY MAE/R2, length MAE/R2.
- Novelty, diversity, AMP classifier score, invalid/degenerate rate.
- Per-condition target-vs-generated plots.

### Gate 4: Independent Generated-Peptide Validation

Decision changed by this gate: whether MIC-conditioned generation can be discussed as activity control rather than model-score steering.

Minimum analysis:

- Evaluate generated peptides with an independent MIC/activity scorer not used as the conditioning oracle.
- Compare against unconditional generation, shuffled-condition controls, and nearest-neighbor retrieval.
- Report species-selectivity failure explicitly if it persists.
- Add top-candidate table only with computational-screening language.

### Gate 5: Submission Formatting and Compression

Decision changed by this gate: whether the manuscript is physically submittable.

Minimum work:

- Fix overfull boxes in architecture, QMAP, MIC generation, and methods tables/figures.
- Convert to the selected official template.
- Reduce the main text to the target page budget by moving implementation detail to supplement.

## Claims to Avoid Until Evidence Improves

- Do not claim superiority over AMPlify/ESM2 for AMP classification.
- Do not claim wet-lab potency, biological activity, or assay-ready candidates from computational MIC oracle scores.
- Do not claim reliable species-selective MIC control.
- Do not claim reliable hydrophobicity or length control unless new ablations change the result.
- Do not claim all decoder components are necessary before a full ablation.
- Do not cite exact MIC numbers as final until provenance is locked.

## Recommended Next Action

Start with Gate 0. It is the cheapest and highest-impact step because it decides which current results are usable. After that, run a small smoke test for the formal MIC pipeline before launching the full MIC reproduction. No long run should start until the smoke test verifies the split, output schema, metrics, and resource behavior.

## Observation

JEPA-AMP has a plausible ISMB/Bioinformatics story after the prose revision, but the current evidence is uneven. QMAP and conservative classification claims are the strongest; MIC provenance and conditional generation evidence are the main blockers.

## Interpretation

The paper should be positioned around bounded representation utility and honest conditional-control limits, not around broad superiority or validated peptide discovery.

## Next Action

Build the Gate 0 evidence lock: create a manifest linking every claim-bearing number in the manuscript to a config, split, command, metrics artifact, predictions artifact, and log.
