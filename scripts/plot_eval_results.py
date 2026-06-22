"""
Plot EF@k curves and prediction vs true scatter for regressor and baselines.
Usage examples:
  python scripts/plot_eval_results.py --out-dir eval_results/plots
  python scripts/plot_eval_results.py --out-dir eval_results/plots --preds eval_results/mic_regressor_grampa_preds.npz --do-embed

Notes:
- This script does not retrain models; if prediction embeddings are not present, it will only plot EF@k from JSON results.
- Requires matplotlib, numpy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import glob


def load_json(path: Path):
    return json.load(open(path))


def plot_ef_curves(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # find regressor combined results
    reg_f = Path('eval_results/mic_regressor_results.json')
    thr_files = sorted(glob.glob('eval_results/mic_regressor_grampa_test_thr*.json'))
    baseline_f = Path('eval_results/baseline_eval_grampa.json')

    plt.figure(figsize=(6,4))
    # plot regressor internal ks if present
    if reg_f.exists():
        data = load_json(reg_f)
        ks = sorted(int(k) for k in data['ks'].keys())
        ef = [data['ks'][str(k)]['EF@k'] for k in ks]
        plt.plot(ks, ef, marker='o', label='Regressor (combined rank)')

    # plot GRAMPA thr files with CI
    for f in thr_files:
        d = load_json(Path(f))
        ks = sorted(int(k) for k in d['ks'].keys())
        ef = [d['ks'][str(k)]['EF@k'] for k in ks]
        ci_low = [d['ks'][str(k)].get('EF@k_CI95', [None, None])[0] for k in ks]
        ci_high = [d['ks'][str(k)].get('EF@k_CI95', [None, None])[1] for k in ks]
        label = Path(f).stem.split('_')[-1].replace('thr','thr=')
        plt.plot(ks, ef, marker='o', label=label)
        # shade CI if available
        if all(ci_low) and all(ci_high):
            plt.fill_between(ks, ci_low, ci_high, alpha=0.2)

    # baselines (if available)
    if baseline_f.exists():
        b = load_json(baseline_f)
        for name, ks_dict in b['ks'].items():
            ks = sorted(int(k) for k in ks_dict.keys())
            ef = [ks_dict[str(k)]['EF@k'] for k in ks]
            plt.plot(ks, ef, marker='x', linestyle='--', label=f'Baseline: {name}')

    plt.xlabel('k')
    plt.ylabel('EF@k')
    plt.xscale('log')
    plt.legend()
    plt.title('Enrichment Factor (EF@k)')
    plt.grid(True, axis='y', linestyle=':')
    out_png = out_dir / 'ef_curves.png'
    plt.tight_layout()
    plt.savefig(out_png)
    print('Wrote', out_png)


def plot_pred_vs_true(preds_npz: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(preds_npz)
    preds = data['preds']
    true = data['true']
    plt.figure(figsize=(5,5))
    plt.scatter(true, preds, alpha=0.6, s=10)
    mn = min(true.min(), preds.min())
    mx = max(true.max(), preds.max())
    plt.plot([mn,mx],[mn,mx], color='k', linestyle='--')
    plt.xlabel('True log2 MIC')
    plt.ylabel('Predicted log2 MIC')
    plt.title('Regressor: pred vs true')
    plt.grid(True, linestyle=':')
    out_png = out_dir / 'pred_vs_true.png'
    plt.tight_layout()
    plt.savefig(out_png)
    print('Wrote', out_png)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path, default=Path('eval_results/plots'))
    p.add_argument('--preds', type=Path, default=Path('eval_results/mic_regressor_grampa_preds.npz'),
                   help='Optional NPZ with arrays "preds" and "true"')
    args = p.parse_args()

    plot_ef_curves(args.out_dir)
    if args.preds.exists():
        plot_pred_vs_true(args.preds, args.out_dir)
    else:
        print('No preds file found; skipping pred-vs-true. To enable, supply --preds with an NPZ containing arrays "preds" and "true".')

if __name__ == '__main__':
    main()
