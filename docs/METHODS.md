METHODS

Data sources
- AMP corpus: consolidated from multiple public AMP databases; sequences filtered to length 3–50 and canonical amino acids. See `data/processed/amp_corpus.fasta` and `data/splits/test.tsv` for held-out test sequences (source-held-out: `amplify`).
- GRAMPA: experimental MIC measurements compiled in `data/grampa.csv`. We use per-unique-sequence mean log2 MIC in evaluation.

Pseudolabel generation
- A JEPA-based MIC predictor (`JEPAMICPredictor`) was fine-tuned on available experimental MIC data (`configs/mic_868k_transformer.yaml`). We then used the trained predictor to infer log2 MIC for all sequences in the AMP corpus producing `data/processed/mic_pseudolabels.npy` (shape N x 20 for 20 bacteria). Per-sequence scalar labels used in some analyses are the mean across bacteria.

Regression probe
- We embed sequences with JEPA context encoder mean-pooled representations (see `src/eval/amp_classifier.py::JEPAAMPClassifier._embed`). A Ridge regressor (alpha=1.0) was trained on the pseudolabel embeddings (80/20 train/val split) to predict mean log2 MIC.
- Internal validation metrics: RMSE and Spearman correlation are reported. The trained regressor is saved as `eval_results/mic_regressor.joblib`.

Enrichment evaluation (EF@k)
- Enrichment Factor (EF@k) measures the fold-improvement in finding actives in the top k ranked candidates relative to random selection.
- Given ranked candidates with values v_i (lower = better), and an active threshold T (e.g., log2 MIC <= 3.0), define:

  - hits = number of candidates among top k with v_i <= T
  - N_pos = total number of actives in the full set
  - baseline = N_pos / N_total
  - EF@k = (hits / k) / baseline

- We compute EF@k both by ranking generated candidates by regressor predictions and by using a kNN proxy: assigning each generated sequence the measured MIC of its nearest neighbor in the pseudolabel embedding set.
- For GRAMPA test evaluations and baselines, we compute 95% bootstrap CIs by resampling test sequences with replacement (default 1000 replicates).

Baselines
- AMPlify (if installed) and an ESM-2 fine-tuned AMP classifier are used as baseline scorers to rank sequences by AMP probability; JEPA logistic probe (saved in `eval_results/amp_classifier.pkl` if present) is also supported.

Reproducibility
- See `REPRODUCE.md` for exact commands used to regenerate pseudolabels, train/evaluate regressors, and run baseline comparisons.

Limitations
- Pseudolabels are model-inferred and may carry biases; enrichment on generated novel sequences relies on predictors, hence circularity risk. We report EF curves with bootstrap CIs and compare regressor-based and kNN-proxy estimators to mitigate this.
