from pathlib import Path
import joblib
p = Path('eval_results/amp_classifier.pkl')
if not p.exists():
    print('MISSING')
else:
    obj = joblib.load(p)
    print('TYPE:', type(obj))
    try:
        if isinstance(obj, dict):
            print('DICT KEYS:', list(obj.keys()))
        else:
            # try to inspect attributes
            attrs = [a for a in dir(obj) if not a.startswith('_')]
            print('ATTRS SAMPLE:', attrs[:30])
    except Exception as e:
        print('INSPECT ERROR', e)
