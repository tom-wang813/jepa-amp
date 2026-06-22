"""
Baseline scorer evaluation on GRAMPA test sequences.
- Attempts to score with AMPlify and ESM-2 AMP classifier.
- Falls back to a saved JEPA AMP probe model if available at `eval_results/amp_classifier.pkl`.
- Computes EF@k and bootstrap 95% CI for each scorer.

Usage:
  python scripts/evaluate_baselines.py --out eval_results/baseline_eval_grampa.json --gpu 0 --n-bootstrap 500

Notes:
- This script does NOT train models. It uses availability of external packages or the saved probe.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

from src.data.supervised_dataset import load_grampa


def ef_at_k(ranked_vals, k, threshold):
    topk = ranked_vals[:k]
    hits = sum(1 for v in topk if v <= threshold)
    N_total = len(ranked_vals)
    N_pos = sum(1 for v in ranked_vals if v <= threshold)
    if N_pos == 0:
        return float('nan'), float('nan')
    baseline = N_pos / N_total
    ef = (hits / k) / baseline
    return ef, hits / k


def bootstrap_ci(values, n_boot=1000, seed=42, alpha=0.05):
    rng = random.Random(seed)
    arr = []
    n = len(values)
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(n)]
        arr.append(np.nanmean(sample))
    lo = float(np.percentile(arr, 100 * (alpha/2)))
    hi = float(np.percentile(arr, 100 * (1 - alpha/2)))
    return [lo, hi]


def load_probe(path: Path):
    # try to load a sklearn pipeline / pickle
    try:
        import joblib
        obj = joblib.load(path)
        # Handle dict structure (e.g., {'pipeline': ..., 'fitted': True, ...})
        if isinstance(obj, dict):
            if 'pipeline' in obj:
                pipeline = obj['pipeline']
            elif 'clf' in obj:
                pipeline = obj['clf']
            else:
                return None
        else:
            pipeline = obj

        def predict(seqs):
            # Try direct predict_proba; if fails, use descriptor fallback
            try:
                return pipeline.predict_proba(seqs)[:, 1]
            except (TypeError, ValueError):
                # Fallback: compute 25-d descriptors
                from src.eval.metrics import aa_frequency, physicochemical_stats
                from src.data.tokenizer import AMINO_ACIDS
                import numpy as _np

                aa_order = list(AMINO_ACIDS)
                feats = []
                for s in seqs:
                    freqs = aa_frequency([s])
                    pc = physicochemical_stats([s])
                    vec = [_np.float32(freqs[a]) for a in aa_order]
                    vec.extend([
                        _np.float32(pc['mean_length']),
                        _np.float32(pc['mean_charge']),
                        _np.float32(pc['mean_hydrophobicity']),
                        _np.float32(pc['fraction_charged']),
                        _np.float32(pc['fraction_hydrophobic']),
                    ])
                    feats.append(vec)
                X = _np.stack(feats)
                return pipeline.predict_proba(X)[:, 1]

        return predict
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--grampa-csv', type=Path, default=Path('data/grampa.csv'))
    p.add_argument('--out', type=Path, default=Path('eval_results/baseline_eval_grampa.json'))
    p.add_argument('--ks', type=int, nargs='+', default=[1,5,10,50,100])
    p.add_argument('--threshold', type=float, default=3.0)
    p.add_argument('--n-bootstrap', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    # load GRAMPA test unique sequences
    _, _, test_ds = load_grampa(args.grampa_csv, seed=args.seed)
    seq_records = {}
    for rec in test_ds._records:
        seq_records.setdefault(rec['seq'], []).append(rec['log2_mic'])
    unique_seqs = list(seq_records.keys())
    true_vals = np.array([np.mean(seq_records[s]) for s in unique_seqs])

    # prepare scorers
    scorers = {}

    # AMPlify
    try:
        from src.eval.amp_classifier import AMPlifyClassifier
        amp = AMPlifyClassifier()
        scorers['AMPlify'] = amp.predict_proba
    except Exception:
        pass

    # ESM-2 (disabled due to CUDA driver compatibility issue)
    # try:
    #     from src.eval.amp_classifier import ESMAMPClassifier
    #     esm = ESMAMPClassifier(device=f'cuda:{args.gpu}' if __import__('torch').cuda.is_available() else 'cpu')
    #     scorers['ESM2'] = esm.predict_proba
    # except Exception:
    #     pass

    # JEPA probe from eval_results
    probe = load_probe(Path('eval_results/amp_classifier.pkl'))
    if probe is not None:
        scorers['JEPA_probe'] = probe

    # Additional saved sklearn classifiers (LR, RF, SVM, GBM)
    try:
        import joblib
        from src.eval.metrics import aa_frequency, physicochemical_stats
        from src.data.tokenizer import AMINO_ACIDS
        import numpy as _np

        saved_models = {
            'LR': Path('eval_results/amp_classifier_lr.pkl'),
            'RF': Path('eval_results/amp_classifier_rf.pkl'),
            'SVM': Path('eval_results/amp_classifier_svm.pkl'),
            'GBM': Path('eval_results/amp_classifier_gbm.pkl'),
        }

        def make_feat_matrix(seqs: list[str]) -> _np.ndarray:
            aa_order = list(AMINO_ACIDS)
            feats = []
            for s in seqs:
                freqs = aa_frequency([s])
                pc = physicochemical_stats([s])
                vec = [_np.float32(freqs[a]) for a in aa_order]
                vec.extend([
                    _np.float32(pc['mean_length']),
                    _np.float32(pc['mean_charge']),
                    _np.float32(pc['mean_hydrophobicity']),
                    _np.float32(pc['fraction_charged']),
                    _np.float32(pc['fraction_hydrophobic']),
                ])
                feats.append(vec)
            return _np.stack(feats)

        X_test_feats = make_feat_matrix(unique_seqs)
        for name, p in saved_models.items():
            if p.exists():
                try:
                    obj = joblib.load(p)
                    pipeline = obj.get('pipeline') if isinstance(obj, dict) else obj
                    def scorer_fn(seqs: list[str], pipeline=pipeline, X_full=X_test_feats):
                        # pipeline expects numeric features; use precomputed X_test_feats
                        return pipeline.predict_proba(X_full)[:, 1]
                    scorers[name] = scorer_fn
                except Exception:
                    pass
    except Exception:
        pass

    if not scorers:
        print('No scorers available (AMPlify/ESM2/probe missing). Place a probe at eval_results/amp_classifier.pkl or install dependencies.')

    results = {'n_test': len(unique_seqs), 'ks': {}, 'threshold': args.threshold, 'scorers': list(scorers.keys())}

    for name, fn in scorers.items():
        print('Scoring with', name)
        try:
            probs = fn(unique_seqs)
        except Exception as exc:
            print('Scorer', name, 'failed:', exc)
            continue
        # higher probability => more AMP. We expect low MIC linked to AMP but this is a proxy.
        # For EF we treat lower MIC as positive, so we rank by decreasing AMP probability and then map
        order = np.argsort(-probs)
        ranked_vals = list(true_vals[order])

        ks = args.ks
        scorer_res = {}
        for k in ks:
            ef, prec = ef_at_k(ranked_vals, k, args.threshold)
            scorer_res[k] = {'EF@k': ef, 'Precision@k': prec}
        # bootstrap CI for EF@k: resample test sequences and recompute EF@k
        boot = {k: [] for k in ks}
        rng = random.Random(args.seed)
        idxs = list(range(len(unique_seqs)))
        for _ in range(args.n_bootstrap):
            sample = [rng.choice(idxs) for _ in idxs]
            vals_b = true_vals[sample]
            probs_b = probs[sample]
            order_b = np.argsort(-probs_b)
            ranked_b = vals_b[order_b]
            for k in ks:
                ef_b, _ = ef_at_k(ranked_b, k, args.threshold)
                boot[k].append(ef_b)
        for k in ks:
            arr = np.array([v for v in boot[k] if not np.isnan(v)])
            if arr.size == 0:
                ci = [float('nan'), float('nan')]
            else:
                ci = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
            scorer_res[k]['EF@k_CI95'] = ci
        results['ks'][name] = scorer_res

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('Wrote baseline evaluation to', args.out)


if __name__ == '__main__':
    main()
