"""
Score generated candidates with available baseline scorers (AMPlify, ESM2, JEPA probe).
Writes JSON with per-scorer probabilities for each generated sequence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.amp_classifier import AMPlifyClassifier, ESMAMPClassifier, JEPAAMPClassifier


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gen-json', type=Path, default=Path('eval_results/conditional_gen_v3_test.json'))
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--out', type=Path, default=Path('eval_results/baseline_scores_cond_gen_v3.json'))
    args = p.parse_args()

    seqs = load_generated(args.gen_json)
    print('Loaded generated sequences:', len(seqs))

    results = {'n': len(seqs), 'scorers': {}}

    # JEPA probe if present
    try:
        import joblib
        probe_path = Path('eval_results/amp_classifier.pkl')
        if probe_path.exists():
            probe_obj = joblib.load(probe_path)
            # joblib artifact may be a dict with pipeline inside
            if isinstance(probe_obj, dict):
                if 'pipeline' in probe_obj:
                    pipeline = probe_obj['pipeline']
                elif 'clf' in probe_obj:
                    pipeline = probe_obj['clf']
                else:
                    pipeline = None
            else:
                pipeline = probe_obj

            if pipeline is None:
                raise RuntimeError('Unsupported amp_classifier.pkl structure')

            # First try direct predict_proba (pipeline may accept sequences)
            try:
                raw_probs = pipeline.predict_proba(seqs)
                probs = (raw_probs[:, 1] if getattr(raw_probs, 'ndim', 0) > 1 else raw_probs).tolist()
            except Exception:
                # Fallback: pipeline expects numeric handcrafted descriptors (20 AA freq + 5 physchem).
                try:
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
                    raw_probs = pipeline.predict_proba(X)
                    probs = (raw_probs[:, 1] if getattr(raw_probs, 'ndim', 0) > 1 else raw_probs).tolist()
                except Exception as e2:
                    raise RuntimeError(f'JEPA probe features fallback failed: {e2}')

            results['scorers']['JEPA_probe'] = probs
            print('JEPA probe scored')
    except Exception as e:
        print('JEPA probe unavailable or failed:', e)

    # ESM2
    try:
        esm = ESMAMPClassifier(device=f'cuda:{args.gpu}' if __import__('torch').cuda.is_available() else 'cpu')
        probs = esm.predict_proba(seqs).tolist()
        results['scorers']['ESM2'] = probs
        print('ESM2 scored')
    except Exception as e:
        print('ESM2 unavailable or failed:', e)

    # AMPlify
    try:
        amp = AMPlifyClassifier()
        probs = amp.predict_proba(seqs).tolist()
        results['scorers']['AMPlify'] = probs
        print('AMPlify scored')
    except Exception as e:
        print('AMPlify unavailable or failed:', e)

    # Additional saved sklearn classifiers in eval_results (LR, RF, SVM, GBM)
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

        for name, p in saved_models.items():
            if p.exists():
                try:
                    obj = joblib.load(p)
                    pipeline = obj.get('pipeline') if isinstance(obj, dict) else obj
                    probs = pipeline.predict_proba(X)
                    probs = (probs[:, 1] if getattr(probs, 'ndim', 0) > 1 else probs).tolist()
                    results['scorers'][name] = probs
                    print(f'{name} scored from {p}')
                except Exception as e:
                    print(f'{name} failed:', e)
    except Exception:
        pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('Wrote baseline scores to', args.out)


if __name__ == '__main__':
    main()
