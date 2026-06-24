Results

Regression performance
- We trained a Ridge regressor on JEPA mean-pooled embeddings using 868,724 pseudolabeled sequences (mean log2 MIC across 20 bacteria as target).
- Validation on pseudolabel split: RMSE = 0.2487, Spearman = 0.8981.
- Generalization to independent experimental GRAMPA test set (n=394): RMSE = 0.6736, Spearman = 0.5063.

Enrichment of generated candidates
- We generated 1,000 conditional candidates using the v3 generator (10 conditions ×100 each); the produced set is `eval_results/conditional_gen_v3_test.json`.
- Exact overlap between generated sequences and the pseudolabel corpus is zero.
- Using the trained regressor to rank generated candidates yields measurable enrichment of sequences below log2 MIC thresholds; EF@k curves with 95% bootstrap CIs are shown in `eval_results/plots/ef_curves.png` (also include kNN-proxy estimates to mitigate model circularity).

Baselines
- We evaluated available baselines (AMPlify, ESM-2 fine-tuned AMP classifier, JEPA logistic probe) on the GRAMPA test set; results are in `eval_results/baseline_eval_grampa.json` and baseline scores for generated candidates in `eval_results/baseline_scores_cond_gen_v3.json`.
- Some external dependencies were unavailable in the runtime environment; unavailable scorers report NaN and should be rerun in an environment with `transformers`/`AMPlify` installed for full comparison.

Figures
- `eval_results/plots/pred_vs_true.png`: regressor predictions vs GRAMPA true mean log2 MIC (shows moderate generalization).
- `eval_results/plots/ef_curves.png`: EF@k curves comparing regressor ranking, kNN-proxy, and baselines (with bootstrap CI).

Interpretation
- JEPA embeddings encode MIC-relevant signal, enabling a simple linear probe to recover MIC (high internal Spearman) and to generalize moderately to experimental data (Spearman ≈ 0.51).
- Prioritizing generator outputs with this probe enriches for low predicted MIC candidates; however, experimental validation is necessary because generated sequences are novel and our enrichment relies on model predictions. Bootstrap CIs and kNN proxy reduce but do not eliminate circularity concerns.

Next steps (planned)
1. Re-run baseline scorers in a full environment with `transformers` and `AMPlify` installed and update comparisons.
2. Generate larger candidate sets (n=5k–10k) and repeat EF@k to stabilize estimates.
3. Run negative controls and ablations (shuffled sequences, length/composition matches, pretrain ablations).
4. Select top 20 candidates for wet-lab MIC testing.
