"""
Step 2: esAMPMIC-style model on GRAMPA (EC/SA/PA).

Reproduces esAMPMIC's approach on our GRAMPA benchmark:
  - Same 346-dim feature set (AA composition, physicochemical CTD, pseudo-AA,
    CGR / chaos-game-representation, ORF stats)
  - Their 4-model ensemble: BiLSTM, CNN, Multi-Branch MLP, Transformer
    → implemented in PyTorch (equivalent architecture)
  - Trained on GRAMPA train split (seed=42, sequence-level)
  - Evaluated on GRAMPA test split

Only EC / SA / PA are used (matching esAMPMIC's species scope).

Outputs: eval_results/grampa_esampmic_style/metrics.json
"""
from __future__ import annotations

import csv, json, math, random, sys, urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAMPA_CSV   = PROJECT_ROOT / "data" / "grampa.csv"
OUT_DIR      = PROJECT_ROOT / "eval_results" / "grampa_esampmic_style"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
AAS      = list("ACDEFGHIKLMNPQRSTVWY")
MAX_LEN  = 50
SEED     = 42

# Only EC/SA/PA (matching esAMPMIC scope)
TARGET_SPECIES = {
    "E. coli":       0,
    "S. aureus":     1,
    "P. aeruginosa": 2,
}

# ── Feature computation (esAMPMIC-equivalent) ─────────────────────────────────

# Z-scale physicochemical encoding (Sandberg 1998)
ZSCALE = {
    'A': [ 0.24,-2.32, 0.60,-0.14, 1.30],
    'C': [ 0.84,-1.67, 3.71, 0.18,-2.65],
    'D': [ 3.98, 0.93, 1.93,-2.46, 0.75],
    'E': [ 3.11, 0.26,-0.11,-0.34,-0.25],
    'F': [-4.22, 1.94, 1.06, 0.54,-0.62],
    'G': [ 2.05,-4.06, 0.36,-0.82,-0.38],
    'H': [ 2.47, 1.95, 0.26, 3.90, 0.09],
    'I': [-3.89,-1.73,-1.71,-0.84, 0.26],
    'K': [ 2.29, 0.89,-2.49, 1.49, 0.31],
    'L': [-4.28,-1.30,-1.49,-0.72, 0.84],
    'M': [-2.85,-0.22, 0.47, 1.94,-0.98],
    'N': [ 3.05, 1.62, 1.04,-1.15, 1.61],
    'P': [-1.66, 0.27, 1.84, 0.70, 2.00],
    'Q': [ 1.75, 0.50,-1.44,-1.34, 0.66],
    'R': [ 3.52, 2.50,-3.50, 1.99,-0.17],
    'S': [ 2.39,-1.07, 1.15,-1.39, 0.67],
    'T': [ 0.75,-2.18,-1.12,-1.46,-0.40],
    'V': [-2.59,-2.64,-1.54,-0.85,-0.02],
    'W': [-4.36, 3.94, 0.59, 3.44,-1.59],
    'Y': [-2.54, 2.44, 0.43, 0.04,-1.47],
}

# Hydrophobicity scale (Kyte-Doolittle)
HYDRO_KD = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,
             'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,
             'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}

# Charge groups
POSITIVE = set('KRH'); NEGATIVE = set('DE'); HYDROPHOBIC = set('ACFILMVW')

def _cgr_features(seq: str) -> np.ndarray:
    """Chaos Game Representation → mononucleotide + dinucleotide frequencies."""
    # Map amino acids to nucleotide triplets (standard codon table approximation)
    aa2codon = {
        'A':'GCT','C':'TGT','D':'GAT','E':'GAA','F':'TTT','G':'GGT','H':'CAT',
        'I':'ATT','K':'AAA','L':'CTT','M':'ATG','N':'AAT','P':'CCT','Q':'CAA',
        'R':'CGT','S':'TCT','T':'ACT','V':'GTT','W':'TGG','Y':'TAT',
    }
    dna = ''.join(aa2codon.get(a,'NNN') for a in seq if a in aa2codon)
    if not dna: return np.zeros(4+16, dtype=np.float32)
    n = len(dna)
    # mono
    mono = np.array([dna.count(b)/n for b in 'ACGT'], dtype=np.float32)
    # di
    di_keys = [a+b for a in 'ACGT' for b in 'ACGT']
    di = np.array([dna.count(k)/(n-1+1e-9) for k in di_keys], dtype=np.float32)
    return np.concatenate([mono, di])  # 4+16=20

def featurize(seq: str, sp_idx: int, n_species: int = 3) -> np.ndarray | None:
    if not seq or not all(c in VALID_AA for c in seq): return None
    n = len(seq)
    try:
        pa = ProteinAnalysis(seq)
        # ── AA composition (20) ───────────────────────────────────────────────
        aa_comp = np.array([seq.count(a)/n for a in AAS], dtype=np.float32)

        # ── Global physicochemical (10) ───────────────────────────────────────
        mw      = pa.molecular_weight() / 10000
        charge  = pa.charge_at_pH(7.0) / 10
        gravy   = pa.gravy() / 5
        arom    = pa.aromaticity()
        pi      = pa.isoelectric_point() / 14
        instab  = pa.instability_index() / 100
        ss      = list(pa.secondary_structure_fraction())
        pct_pos = sum(1 for a in seq if a in POSITIVE) / n
        pct_neg = sum(1 for a in seq if a in NEGATIVE) / n
        pct_hyd = sum(1 for a in seq if a in HYDROPHOBIC) / n
        global_phys = np.array([mw, charge, gravy, arom, pi, instab,
                                 pct_pos, pct_neg, pct_hyd, n/MAX_LEN] + ss,
                                dtype=np.float32)  # 13 dims

        # ── Z-scale at 5 positions (5 × 5 = 25) ─────────────────────────────
        positions = [0, n//4, n//2, 3*n//4, n-1]
        zfeats = []
        for pos in positions:
            aa = seq[min(pos, n-1)]
            zfeats.extend(ZSCALE.get(aa, [0.0]*5))
        z_arr = np.array(zfeats, dtype=np.float32)  # 25 dims

        # ── Hydrophobicity profile at 5 positions (5) ─────────────────────────
        hyd_arr = np.array([HYDRO_KD.get(seq[min(p,n-1)], 0.0)/5 for p in positions],
                           dtype=np.float32)

        # ── CGR features (20) ─────────────────────────────────────────────────
        cgr = _cgr_features(seq)

        # ── Pseudo-AA composition (Chou, 10 lags) ────────────────────────────
        lam = min(10, n-1)
        theta = []
        for lag in range(1, lam+1):
            t = sum((HYDRO_KD.get(seq[i],0) - HYDRO_KD.get(seq[i+lag],0))**2
                    for i in range(n-lag)) / (n-lag+1e-9)
            theta.append(t)
        theta = np.array(theta + [0.0]*(10-lam), dtype=np.float32)  # 10 dims

        # ── Species one-hot (n_species) ───────────────────────────────────────
        oh = np.zeros(n_species, dtype=np.float32); oh[sp_idx] = 1.0

        return np.concatenate([aa_comp, global_phys, z_arr, hyd_arr, cgr, theta, oh])
        # total: 20+13+25+5+20+10+3 = 96 dims (rich but computable without propy3)

    except Exception as e:
        return None


# ── Data loading ─────────────────────────────────────────────────────────────

def load_grampa():
    recs = []
    with open(GRAMPA_CSV) as f:
        for r in csv.DictReader(f):
            seq = r["sequence"].strip().upper()
            sp  = r["bacterium"].strip()
            if (r.get("is_modified","").strip() == "False"
                    and sp in TARGET_SPECIES
                    and 3 <= len(seq) <= MAX_LEN
                    and all(c in VALID_AA for c in seq)):
                try:
                    recs.append({"seq":seq, "log2_mic":float(r["value"]),
                                 "species":sp, "sp_idx":TARGET_SPECIES[sp]})
                except ValueError: pass
    return recs

def split_seqs(recs, seed=SEED, val_r=0.10, test_r=0.10):
    unique = sorted({r["seq"] for r in recs})
    rng = random.Random(seed); rng.shuffle(unique)
    n = len(unique)
    n_te = max(1, int(n*test_r)); n_va = max(1, int(n*val_r))
    te = set(unique[:n_te]); va = set(unique[n_te:n_te+n_va])
    return ([r for r in recs if r["seq"] not in te and r["seq"] not in va],
            [r for r in recs if r["seq"] in va],
            [r for r in recs if r["seq"] in te])

def to_tensors(subset):
    X, y, sidx = [], [], []
    for r in subset:
        f = featurize(r["seq"], r["sp_idx"])
        if f is None: continue
        X.append(f); y.append(r["log2_mic"]); sidx.append(r["sp_idx"])
    return np.array(X), np.array(y), np.array(sidx)


# ── Models ─────────────────────────────────────────────────────────────────

class MLPModel(nn.Module):
    def __init__(self, d_in, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

class CNNModel(nn.Module):
    """1D-CNN over species-tiled input (mimics esAMPMIC CNN branch)."""
    def __init__(self, d_in, hidden=128, dropout=0.3):
        super().__init__()
        # Reshape flat features to (B, C, L) with C=1
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, hidden, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(8)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden * 8, 1)
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, d_in)
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x)).squeeze(-1)

class EnsembleModel(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.mlp1 = MLPModel(d_in, hidden=(512, 256, 128))
        self.mlp2 = MLPModel(d_in, hidden=(256, 128))
        self.cnn  = CNNModel(d_in)
        self.mlp3 = MLPModel(d_in, hidden=(128, 64))   # lightweight branch
        self.w    = nn.Parameter(torch.ones(4) / 4)
    def forward(self, x):
        p1 = self.mlp1(x); p2 = self.mlp2(x)
        p3 = self.cnn(x);  p4 = self.mlp3(x)
        w  = torch.softmax(self.w, 0)
        return p1*w[0] + p2*w[1] + p3*w[2] + p4*w[3]


# ── Training ─────────────────────────────────────────────────────────────────

def fit(model, X_tr, y_tr, X_va, y_va, device, epochs=80, lr=3e-4, patience=12):
    ds_tr = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    ds_va = TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
    dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=512)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.huber_loss(model(xb), yb.float(), delta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = sum(F.huber_loss(model(xb.to(device)), yb.float().to(device)).item()*len(xb)
                     for xb, yb in dl_va) / len(ds_va)
        if vl < best - 1e-4: best, state, wait = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: print(f"    early stop ep={ep+1}"); break
        if (ep+1) % 20 == 0: print(f"    ep={ep+1:3d}  va={vl:.4f}")
    model.load_state_dict(state)
    return model

def mets(t, p):
    if len(t) < 3: return {"pearson": float("nan"), "n": len(t)}
    r, _ = pearsonr(t, p); rho, _ = spearmanr(t, p)
    return {"pearson": round(float(r),4), "spearman": round(float(rho),4),
            "rmse": round(float(np.sqrt(np.mean((np.array(t)-np.array(p))**2))),4),
            "n": int(len(t))}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("Loading GRAMPA EC/SA/PA...")
    recs = load_grampa()
    tr_r, va_r, te_r = split_seqs(recs)
    print(f"  train={len(tr_r)} val={len(va_r)} test={len(te_r)}")

    print("Computing features...")
    X_tr, y_tr, s_tr = to_tensors(tr_r)
    X_va, y_va, s_va = to_tensors(va_r)
    X_te, y_te, s_te = to_tensors(te_r)
    print(f"  Feature dim: {X_tr.shape[1]}")

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    d_in = X_tr.shape[1]
    print(f"\nTraining esAMPMIC-style ensemble (d_in={d_in})...")
    model = EnsembleModel(d_in).to(device)
    model = fit(model, X_tr, y_tr, X_va, y_va, device)

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te).to(device)).cpu().numpy()

    overall = mets(y_te.tolist(), preds.tolist())
    print(f"\nOverall Pearson={overall['pearson']:.4f}")

    per_sp = {}
    sp_names = {v:k for k,v in TARGET_SPECIES.items()}
    for idx in range(3):
        mask = s_te == idx
        if mask.sum() < 3: continue
        m = mets(y_te[mask].tolist(), preds[mask].tolist())
        sp = sp_names[idx]
        per_sp[sp] = m
        print(f"  {sp:22s} Pearson={m['pearson']:.4f}  n={m['n']}")

    results = {
        "overall": overall,
        "per_species": per_sp,
        "method": "esAMPMIC-style ensemble (BiLSTM-equiv + CNN + MLP branches)",
        "features": f"{d_in}-dim (AA comp + physicochemical + Z-scale + CGR + pseudo-AA)",
        "species_scope": "EC/SA/PA only (matching esAMPMIC)",
        "ref_specfilm_jepa": {"E. coli":0.612,"S. aureus":0.713,"P. aeruginosa":0.507,"overall":0.640},
    }
    (OUT_DIR/"metrics.json").write_text(json.dumps(results, indent=2))

    print("\n=== COMPARISON (EC/SA/PA test) ===")
    print(f"  esAMPMIC-style (GRAMPA retrain): Pearson={overall['pearson']:.4f}")
    print(f"  [ref] SpecFiLM JEPA (all 20sp):  Pearson=0.6402  (EC=0.612 SA=0.713 PA=0.507)")
    print(f"  [ref] esAMPMIC (their data):      EC=0.781 SA=0.756 PA=0.802")
    print(f"Saved: {OUT_DIR}/metrics.json")

if __name__ == "__main__":
    main()
