"""
Evaluate saved MIC regressor on GRAMPA held-out test sequences.
- loads regressor (`joblib`) trained on pseudolabels
- embeds GRAMPA test unique sequences with JEPA encoder
- predicts, computes RMSE / Spearman against mean per-sequence log2 MIC
- computes EF@k and bootstrap 95% CI
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import numpy as np
import joblib
import torch
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


def embed_sequences(encoder, sequences, device='cpu', batch_size=256):
    from src.eval.amp_classifier import JEPAAMPClassifier
    clf = JEPAAMPClassifier(encoder, device=device)
    return clf._embed(sequences, batch_size=batch_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrain-ckpt', type=Path, default=Path('checkpoints/jepa_pretrain_868k/last_jepa.pt'))
    p.add_argument('--reg-joblib', type=Path, default=Path('eval_results/mic_regressor.joblib'))
    p.add_argument('--grampa-csv', type=Path, default=Path('data/grampa.csv'))
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--threshold', type=float, default=3.0, help='log2 MIC threshold for actives')
    p.add_argument('--ks', type=int, nargs='+', default=[1,5,10,50,100])
    p.add_argument('--n-bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path, default=Path('eval_results/mic_regressor_grampa_test.json'))
    p.add_argument('--preds-out', type=Path, default=Path('eval_results/mic_regressor_grampa_preds.npz'),
                   help='Optional NPZ file to save arrays "preds" and "true" for plotting')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # load regressor
    reg = joblib.load(args.reg_joblib)
    print('Loaded regressor:', args.reg_joblib)

    # load GRAMPA and take test split
    _, _, test_ds = load_grampa(args.grampa_csv, seed=args.seed)
    # aggregate per-unique-sequence mean log2 mic
    seq_records = {}
    for i in range(len(test_ds)):
        rec = test_ds._records[i]
        seq = rec['seq']
        seq_records.setdefault(seq, []).append(rec['log2_mic'])
    unique_seqs = list(seq_records.keys())
    true_vals = np.array([np.mean(seq_records[s]) for s in unique_seqs])
    print(f'GRAMPA test unique sequences: {len(unique_seqs)}')

    # load encoder
    from src.train.finetune_supervised import _load_encoder
    encoder, _ = _load_encoder(str(args.pretrain_ckpt), device=device)

    # embed
    print('Embedding test sequences...')
    emb = embed_sequences(encoder, unique_seqs, device=str(device))

    # predict
    preds = reg.predict(emb)

    # metrics
    rmse = float(np.sqrt(mean_squared_error(true_vals, preds)))
    rho, _ = spearmanr(true_vals, preds)
    print(f'RMSE={rmse:.4f} Spearman={rho:.4f}')

    # EF@k
    order = np.argsort(preds)
    ranked_vals = list(true_vals[order])
    ks = args.ks
    results = {'rmse': rmse, 'spearman': rho, 'n_test': len(unique_seqs), 'ks': {}, 'threshold': args.threshold}
    for k in ks:
        ef, prec = ef_at_k(ranked_vals, k, args.threshold)
        results['ks'][k] = {'EF@k': ef, 'Precision@k': prec}

    # bootstrap CI for EF@k by resampling sequences with replacement
    rng = random.Random(args.seed)
    boot = {k: [] for k in ks}
    seq_idx = list(range(len(unique_seqs)))
    for b in range(args.n_bootstrap):
        sample_idx = [rng.choice(seq_idx) for _ in seq_idx]
        vals_b = true_vals[sample_idx]
        # predict by regressor on same indices
        preds_b = preds[sample_idx]
        order_b = np.argsort(preds_b)
        ranked_b = list(vals_b[order_b])
        for k in ks:
            ef_b, _ = ef_at_k(ranked_b, k, args.threshold)
            boot[k].append(ef_b)
    # compute 95% CI
    for k in ks:
        arr = np.array([v for v in boot[k] if not np.isnan(v)])
        if arr.size == 0:
            ci = [float('nan'), float('nan')]
        else:
            lo = float(np.percentile(arr, 2.5))
            hi = float(np.percentile(arr, 97.5))
            ci = [lo, hi]
        results['ks'][k].update({'EF@k_CI95': ci})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('Wrote results to', args.out)
    # optionally save preds/true for plotting
    if args.preds_out:
        np.savez(args.preds_out, preds=preds, true=true_vals)
        print('Saved preds/true to', args.preds_out)


if __name__ == '__main__':
    main()
