"""Prettier plots for baseline evaluation.

Produces two figures:
- `eval_results/plots/ef_vs_k.png` : EF@k vs k (line per scorer, shaded 95% CI)
- `eval_results/plots/precision_vs_k.png` : Precision@k vs k (line per scorer)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

in_path = Path('eval_results/baseline_eval_grampa.json')
out_dir = Path('eval_results/plots')
out_dir.mkdir(parents=True, exist_ok=True)

sns.set(style='whitegrid', context='talk')
palette = sns.color_palette('tab10')

with open(in_path) as f:
    data = json.load(f)

ks = []
scorer_names = []
if data.get('ks'):
    # ks are stored per scorer; collect unique sorted ks
    any_scorer = next(iter(data['ks'].values()))
    ks = sorted([int(k) for k in any_scorer.keys()])
    scorer_names = sorted(data['ks'].keys())

if not ks:
    print('No ks found; aborting plot')
else:
    # Build matrices: scorer -> list of EF and CI, and Precision
    ef_by_scorer = {s: [] for s in scorer_names}
    ef_ci_by_scorer = {s: [] for s in scorer_names}
    prec_by_scorer = {s: [] for s in scorer_names}

    for s in scorer_names:
        res = data['ks'].get(s, {})
        for k in ks:
            entry = res.get(str(k), {})
            ef = entry.get('EF@k', np.nan)
            ci = entry.get('EF@k_CI95', [np.nan, np.nan])
            prec = entry.get('Precision@k', np.nan)
            ef_by_scorer[s].append(ef)
            ef_ci_by_scorer[s].append(ci)
            prec_by_scorer[s].append(prec)

    # EF@k vs k with shaded CI
    plt.figure(figsize=(8, 5))
    for i, s in enumerate(scorer_names):
        ef_vals = np.array(ef_by_scorer[s], dtype=np.float64)
        cis = np.array(ef_ci_by_scorer[s], dtype=np.float64)
        lo = cis[:, 0]
        hi = cis[:, 1]
        color = palette[i % len(palette)]
        plt.plot(ks, ef_vals, label=s, color=color, marker='o')
        # shaded CI where available
        if not np.all(np.isnan(lo)) and not np.all(np.isnan(hi)):
            plt.fill_between(ks, lo, hi, color=color, alpha=0.2)
    plt.xlabel('k')
    plt.xticks(ks)
    plt.ylabel('EF@k')
    plt.title('EF@k across scorers')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_dir / 'ef_vs_k.png', dpi=200)

    # Precision@k vs k
    plt.figure(figsize=(8, 5))
    for i, s in enumerate(scorer_names):
        prec_vals = np.array(prec_by_scorer[s], dtype=np.float64)
        color = palette[i % len(palette)]
        plt.plot(ks, prec_vals, label=s, color=color, marker='o')
    plt.xlabel('k')
    plt.xticks(ks)
    plt.ylabel('Precision@k')
    plt.ylim(0, 1)
    plt.title('Precision@k across scorers')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_dir / 'precision_vs_k.png', dpi=200)

    print('Wrote prettier plots to', out_dir)
