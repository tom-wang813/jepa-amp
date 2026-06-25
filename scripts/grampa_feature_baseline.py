"""
Step 1: Physicochemical feature baseline on GRAMPA 20 species.

Features per sequence (29 dims, via biopython):
  AA composition (20) + MW, charge@pH7, GRAVY, aromaticity,
  isoelectric_point, instability_index, helix, turn, sheet (9)

Models trained on GRAMPA sequence-level split (seed=42):
  A) Shared MLP  — all 20 species, features + species one-hot (49-dim)
  B) Gradient Boosting per species
  C) Ridge regression per species (fast baseline)

Outputs: eval_results/grampa_feature_baseline/metrics.json
"""
from __future__ import annotations
import csv, json, random, sys
from pathlib import Path

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAMPA_CSV   = PROJECT_ROOT / "data" / "grampa.csv"
OUT_DIR      = PROJECT_ROOT / "eval_results" / "grampa_feature_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MAX_LEN  = 50
SEED     = 42

sys.path.insert(0, str(PROJECT_ROOT))
from src.data.supervised_dataset import GRAMPA_TOP20

SPECIES_TO_IDX = {sp: i for i, sp in enumerate(GRAMPA_TOP20)}


# ── feature extraction ────────────────────────────────────────────────────────

def featurize(seq: str) -> np.ndarray | None:
    if not seq or not all(c in VALID_AA for c in seq):
        return None
    try:
        pa = ProteinAnalysis(seq)
        aa_dict = pa.count_amino_acids()
        aa_comp = [aa_dict.get(a, 0) / len(seq) for a in "ACDEFGHIKLMNPQRSTVWY"]  # 20 dims
        mw      = pa.molecular_weight() / 10000               # normalize
        charge  = pa.charge_at_pH(7.0) / 10
        gravy   = pa.gravy() / 5
        arom    = pa.aromaticity()
        pi      = pa.isoelectric_point() / 14
        instab  = pa.instability_index() / 100
        ss      = list(pa.secondary_structure_fraction())      # helix, turn, sheet
        return np.array(aa_comp + [mw, charge, gravy, arom, pi, instab] + ss,
                        dtype=np.float32)
    except Exception:
        return None


# ── data loading (same split as formal models) ────────────────────────────────

def load_grampa():
    recs = []
    with open(GRAMPA_CSV) as f:
        for r in csv.DictReader(f):
            seq = r["sequence"].strip().upper()
            sp  = r["bacterium"].strip()
            if (r.get("is_modified","").strip() == "False"
                    and sp in SPECIES_TO_IDX
                    and 3 <= len(seq) <= MAX_LEN
                    and all(c in VALID_AA for c in seq)):
                try:
                    recs.append({"seq": seq, "log2_mic": float(r["value"]),
                                 "species": sp, "bact_idx": SPECIES_TO_IDX[sp]})
                except ValueError:
                    pass
    return recs

def split_by_seq(recs, seed=SEED, val_r=0.10, test_r=0.10):
    unique_seqs = sorted({r["seq"] for r in recs})
    rng = random.Random(seed); rng.shuffle(unique_seqs)
    n = len(unique_seqs)
    n_te = max(1, int(n * test_r)); n_va = max(1, int(n * val_r))
    te_set = set(unique_seqs[:n_te]); va_set = set(unique_seqs[n_te:n_te+n_va])
    tr = [r for r in recs if r["seq"] not in te_set and r["seq"] not in va_set]
    va = [r for r in recs if r["seq"] in va_set]
    te = [r for r in recs if r["seq"] in te_set]
    return tr, va, te


# ── metrics ───────────────────────────────────────────────────────────────────

def mets(t, p):
    t, p = np.array(t), np.array(p)
    if len(t) < 3: return {"pearson": float("nan"), "n": len(t)}
    r, _ = pearsonr(t, p); rho, _ = spearmanr(t, p)
    rmse = float(np.sqrt(np.mean((t-p)**2)))
    return {"pearson": round(float(r),4), "spearman": round(float(rho),4),
            "rmse": round(rmse,4), "n": int(len(t))}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading GRAMPA...")
    recs = load_grampa()
    tr_r, va_r, te_r = split_by_seq(recs)
    print(f"  train={len(tr_r)}  val={len(va_r)}  test={len(te_r)}")

    print("Computing features...")
    def build_xy(subset, use_species_onehot=True):
        X, y, bidx = [], [], []
        for r in subset:
            f = featurize(r["seq"])
            if f is None: continue
            if use_species_onehot:
                oh = np.zeros(len(GRAMPA_TOP20), dtype=np.float32)
                oh[r["bact_idx"]] = 1.0
                f = np.concatenate([f, oh])
            X.append(f); y.append(r["log2_mic"]); bidx.append(r["bact_idx"])
        if not X: return np.empty((0,0),dtype=np.float32), np.empty(0), np.empty(0,dtype=int)
        return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32), np.array(bidx,dtype=int)

    X_tr, y_tr, bidx_tr = build_xy(tr_r, use_species_onehot=True)
    X_va, y_va, bidx_va = build_xy(va_r, use_species_onehot=True)
    X_te, y_te, bidx_te = build_xy(te_r, use_species_onehot=True)
    print(f"  train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")
    assert X_tr.shape[0] > 0, "No valid training samples"
    print(f"  Feature dim: {X_tr.shape[1]}")

    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr); Xs_va = scaler.transform(X_va); Xs_te = scaler.transform(X_te)

    results = {}

    # ── A) Shared MLP ─────────────────────────────────────────────────────────
    print("\n[A] Shared MLP (species one-hot + features)...")
    mlp = MLPRegressor(hidden_layer_sizes=(256,128), max_iter=500, random_state=SEED,
                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=15)
    mlp.fit(Xs_tr, y_tr)
    p_te = mlp.predict(Xs_te)
    overall_mlp = mets(y_te, p_te)
    print(f"  Overall Pearson={overall_mlp['pearson']:.4f}")
    per_sp_mlp = {}
    for i, sp in enumerate(GRAMPA_TOP20):
        mask = bidx_te == i
        if mask.sum() < 3: continue
        m = mets(y_te[mask], p_te[mask])
        per_sp_mlp[sp] = m
        print(f"  {sp:30s} Pearson={m['pearson']:.3f}  n={m['n']}")
    results["shared_mlp"] = {"overall": overall_mlp, "per_species": per_sp_mlp}

    # ── B) Gradient Boosting per species ─────────────────────────────────────
    print("\n[B] Gradient Boosting per species (no one-hot)...")
    X_tr_no, y_tr_no, bidx_tr_no = build_xy(tr_r, use_species_onehot=False)
    X_te_no, y_te_no, bidx_te_no = build_xy(te_r, use_species_onehot=False)
    sc2 = StandardScaler().fit(X_tr_no)
    Xs_tr_no = sc2.transform(X_tr_no); Xs_te_no = sc2.transform(X_te_no)

    all_p_gb, all_t_gb = [], []
    per_sp_gb = {}
    for i, sp in enumerate(GRAMPA_TOP20):
        tr_mask = bidx_tr_no == i; te_mask = bidx_te_no == i
        n_tr = int(tr_mask.sum()); n_te = int(te_mask.sum())
        if n_tr < 10 or n_te < 3:
            print(f"  {sp:30s} SKIP (n_train={n_tr})")
            continue
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, random_state=SEED)
        gb.fit(Xs_tr_no[tr_mask], y_tr_no[tr_mask])
        p = gb.predict(Xs_te_no[te_mask])
        m = mets(y_te_no[te_mask], p)
        per_sp_gb[sp] = m
        all_p_gb.extend(p.tolist()); all_t_gb.extend(y_te_no[te_mask].tolist())
        print(f"  {sp:30s} Pearson={m['pearson']:.3f}  n={m['n']}")
    overall_gb = mets(all_t_gb, all_p_gb)
    print(f"  Overall (concat) Pearson={overall_gb['pearson']:.4f}")
    results["per_species_gb"] = {"overall": overall_gb, "per_species": per_sp_gb}

    # ── C) Ridge per species ──────────────────────────────────────────────────
    print("\n[C] Ridge per species...")
    all_p_ridge, all_t_ridge = [], []
    per_sp_ridge = {}
    for i, sp in enumerate(GRAMPA_TOP20):
        tr_mask = bidx_tr_no == i; te_mask = bidx_te_no == i
        if int(tr_mask.sum()) < 5 or int(te_mask.sum()) < 3: continue
        ridge = Ridge(alpha=10.0)
        ridge.fit(Xs_tr_no[tr_mask], y_tr_no[tr_mask])
        p = ridge.predict(Xs_te_no[te_mask])
        m = mets(y_te_no[te_mask], p)
        per_sp_ridge[sp] = m
        all_p_ridge.extend(p.tolist()); all_t_ridge.extend(y_te_no[te_mask].tolist())
    overall_ridge = mets(all_t_ridge, all_p_ridge)
    print(f"  Overall (concat) Pearson={overall_ridge['pearson']:.4f}")
    results["per_species_ridge"] = {"overall": overall_ridge, "per_species": per_sp_ridge}

    # ── Save ─────────────────────────────────────────────────────────────────
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    print("\n=== SUMMARY ===")
    print(f"  Shared MLP:          Pearson={overall_mlp['pearson']:.4f}")
    print(f"  Per-species GB:      Pearson={overall_gb['pearson']:.4f}")
    print(f"  Per-species Ridge:   Pearson={overall_ridge['pearson']:.4f}")
    print(f"  [ref] SpecFiLM JEPA: Pearson=0.6402")
    print(f"  [ref] MLM SpecFiLM:  Pearson=0.6085")
    print(f"  [ref] ESM2 SpecFiLM: Pearson=0.5538")
    print(f"Saved: {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
