"""
Train a simple MIC regressor on pseudolabels using JEPA embeddings,
then score generated sequences and compute EF@k using kNN-proxy and regressor predictions.

Writes outputs to `eval_results/mic_regressor_results.json` and saves the regressor as a joblib file.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
import joblib
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

from src.eval.amp_classifier import JEPAAMPClassifier


def load_mic_map(seqs_path: Path, npy_path: Path):
    seqs = [x.strip() for x in open(seqs_path) if x.strip()]
    vals = np.load(npy_path)
    # handle multidim labels by averaging
    if vals.ndim > 1:
        vals = vals.mean(axis=1)
    if len(seqs) != len(vals):
        raise ValueError("mic seqs / values length mismatch")
    return seqs, vals.tolist()


def load_generated(gen_json_path: Path):
    data = json.load(open(gen_json_path))
    seqs = []
    def extract(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'samples' and isinstance(v, list):
                    seqs.extend(v)
                else:
                    extract(v)
        elif isinstance(obj, list):
            for it in obj:
                extract(it)
    extract(data)
    return [s.upper().replace(' ', '') for s in seqs]


def embed_sequences(encoder, sequences, device='cpu', batch_size=256):
    clf = JEPAAMPClassifier(encoder, device=device)
    # use internal _embed to get numpy embeddings
    return clf._embed(sequences)


def knn_proxy(gen_emb, mic_embs, mic_vals, k=1):
    # cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(gen_emb, mic_embs)
    nn_idx = sims.argmax(axis=1)
    return [mic_vals[i] for i in nn_idx]


def ef_at_k(ranked_vals, k, threshold):
    topk = ranked_vals[:k]
    hits = sum(1 for v in topk if v <= threshold)
    N_total = len(ranked_vals)
    N_pos = sum(1 for v in ranked_vals if v <= threshold)
    if N_pos == 0:
        return float('nan')
    baseline = N_pos / N_total
    ef = (hits / k) / baseline
    return ef, hits / k


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mic-seqs', type=Path, default=Path('data/processed/mic_pseudolabels_seqs.txt'))
    p.add_argument('--mic-npy', type=Path, default=Path('data/processed/mic_pseudolabels.npy'))
    p.add_argument('--gen-json', type=Path, default=Path('eval_results/conditional_gen_test.json'))
    p.add_argument('--pretrain-ckpt', type=Path, default=Path('checkpoints/jepa_pretrain_868k/last_jepa.pt'))
    p.add_argument('--gpu', type=int, default=1, help='CUDA device index')
    p.add_argument('--out', type=Path, default=Path('eval_results/mic_regressor_results.json'))
    p.add_argument('--reg-out', type=Path, default=Path('eval_results/mic_regressor.joblib'))
    p.add_argument('--threshold', type=float, default=10.0)
    args = p.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    mic_seqs, mic_vals = load_mic_map(args.mic_seqs, args.mic_npy)
    print('Loaded mic map:', len(mic_seqs))
    gen_seqs = load_generated(args.gen_json)
    print('Loaded generated sequences:', len(gen_seqs))

    # load JEPA encoder via JEPAAMPClassifier convenience (it expects an encoder object)
    # reuse JEPA checkpoint loading from src.train.finetune_supervised._load_encoder
    from src.train.finetune_supervised import _load_encoder
    encoder, _ = _load_encoder(str(args.pretrain_ckpt), device=torch.device(device))

    # embed mic sequences and generated sequences
    print('Embedding mic sequences (this may take a while)')
    mic_embs = embed_sequences(encoder, mic_seqs, device=device)
    print('Embedding generated sequences')
    gen_embs = embed_sequences(encoder, gen_seqs, device=device)

    # train a simple Ridge regressor with 80/20 split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(mic_embs, mic_vals, test_size=0.2, random_state=42)
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    preds_val = reg.predict(X_val)
    # sklearn versions differ: compute RMSE from MSE for compatibility
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    rho, _ = spearmanr(y_val, preds_val)
    print(f'Regressor val RMSE={rmse:.4f} Spearman={rho:.4f}')

    joblib.dump(reg, args.reg_out)
    print('Saved regressor to', args.reg_out)

    # predict on generated sequences
    gen_preds = reg.predict(gen_embs)

    # Combined ranking: use true mic values for mic_seqs, predicted for generated
    combined_vals = list(mic_vals) + list(gen_preds)
    combined_ids = ['mic_'+str(i) for i in range(len(mic_vals))] + ['gen_'+str(i) for i in range(len(gen_preds))]
    ranked = sorted(zip(combined_ids, combined_vals), key=lambda x: x[1])  # lower MIC = better

    ks = [1,5,10,50,100]
    results = {'reg_val_rmse': rmse, 'reg_val_spearman': rho, 'n_mic': len(mic_vals), 'n_gen': len(gen_preds), 'ks': {}}
    ranked_vals = [v for _, v in ranked]
    for k in ks:
        ef, prec = ef_at_k(ranked_vals, k, args.threshold)
        results['ks'][k] = {'EF@k': ef, 'Precision@k': prec}

    # kNN proxy on generated sequences: assign nearest mic neighbor value
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(gen_embs, mic_embs)
    nn_idx = sims.argmax(axis=1)
    proxy_vals = [mic_vals[i] for i in nn_idx]
    # evaluate EF@k when ranking generated by regressor (but measuring with proxy values)
    gen_ranked_idx = sorted(range(len(gen_preds)), key=lambda i: gen_preds[i])
    gen_ranked_proxy = [proxy_vals[i] for i in gen_ranked_idx]
    for k in ks:
        ef_g, prec_g = ef_at_k(gen_ranked_proxy, k, args.threshold)
        results['ks'][k].update({'EF@k_knn_proxy': ef_g, 'Precision@k_knn_proxy': prec_g})

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('Wrote results to', args.out)


if __name__ == '__main__':
    main()
