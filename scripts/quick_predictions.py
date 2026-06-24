"""
Quickly get per-sample predictions for ESM2-150M and MLM on the 5 original pairs.
Only runs k=0 and k=100 (warmstart). Saves to eval_results/fewshot_v2/{model}/.

JEPA is already done. ESM2-650M skipped (too slow while GPU is busy).
"""
from __future__ import annotations
import copy, csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

AA = "ACDEFGHIKLMNPQRSTVWY"
PAIRS = [
    ("E. coli",       "S. aureus"),
    ("E. coli",       "P. aeruginosa"),
    ("S. aureus",     "E. coli"),
    ("S. aureus",     "P. aeruginosa"),
    ("P. aeruginosa", "E. coli"),
]
SEEDS    = [42, 123, 7]
K_VALUES = [0, 5, 10, 20, 50, 100]

def load_species(csv_path, species, seed=42, max_len=50):
    recs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            seq = r["sequence"].strip().upper()
            if (r["is_modified"].strip() == "False"
                    and r["bacterium"].strip() == species
                    and 3 <= len(seq) <= max_len
                    and all(c in AA for c in seq)):
                try:
                    recs.append({"seq": seq, "log2_mic": float(r["value"])})
                except ValueError:
                    continue
    unique_seqs = sorted({r["seq"] for r in recs})
    rng = random.Random(seed); rng.shuffle(unique_seqs)
    n = len(unique_seqs)
    n_test = max(1, int(n * 0.15)); n_val = max(1, int(n * 0.10))
    test_s = set(unique_seqs[:n_test]); val_s = set(unique_seqs[n_test:n_test+n_val])
    train, val, test = [], [], []
    for r in recs:
        if r["seq"] in test_s: test.append(r)
        elif r["seq"] in val_s: val.append(r)
        else: train.append(r)
    return train, val, test

def _batch_encode_jepa(seqs, device):
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:50]) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out

class JEPAEmbedder(nn.Module):
    def __init__(self, device):
        super().__init__()
        from src.models.jepa import JEPA
        ckpt = torch.load(PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
                          map_location=device, weights_only=False)
        jepa = JEPA(**ckpt["cfg"]["model"])
        jepa.load_state_dict(ckpt["model_state"])
        self.enc = jepa.context_encoder
        self.d_model = ckpt["cfg"]["model"]["d_model"]
        for p in self.enc.parameters(): p.requires_grad_(False)
    def forward(self, seqs, device):
        ids = _batch_encode_jepa(seqs, device)
        h = self.enc(ids); pad = ids == 0
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        return h.sum(1) / (~pad).sum(1, keepdim=True).float().clamp(min=1)

class ESM2Embedder(nn.Module):
    def __init__(self, device, model_key="esm2_t12_35M"):
        super().__init__()
        from src.models.esm_head import load_esm2
        self.esm, self.alphabet, d = load_esm2(model_key)
        self.bc = self.alphabet.get_batch_converter()
        self.d_model = d; self.num_layers = self.esm.num_layers
        for p in self.esm.parameters(): p.requires_grad_(False)
    def forward(self, seqs, device):
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = self.bc(data); tokens = tokens.to(device)
        with torch.no_grad():
            out = self.esm(tokens, repr_layers=[self.num_layers], return_contacts=False)
        h = out["representations"][self.num_layers]; pad = tokens == self.alphabet.padding_idx
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        return h.sum(1) / (~pad).sum(1, keepdim=True).float().clamp(min=1)

class MLMEmbedder(nn.Module):
    def __init__(self, device):
        super().__init__()
        from src.models.mlm import MLMModel
        ckpt = torch.load(PROJECT_ROOT / "checkpoints/mlm_pretrain_868k/best_mlm.pt",
                          map_location=device, weights_only=False)
        model = MLMModel(**ckpt["cfg"]["model"])
        enc_state = {k[len("encoder."):]: v for k, v in ckpt["model_state"].items()
                     if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_state)
        self.enc = model.encoder; self.d_model = ckpt["cfg"]["model"]["d_model"]
        for p in self.enc.parameters(): p.requires_grad_(False)
    def forward(self, seqs, device):
        ids = _batch_encode_jepa(seqs, device)
        h = self.enc(ids); pad = ids == 0
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        return h.sum(1) / (~pad).sum(1, keepdim=True).float().clamp(min=1)

class MICHead(nn.Module):
    def __init__(self, d_model, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Dropout(0.3), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def precompute(emb, recs, device, batch=256):
    parts = []
    for i in range(0, len(recs), batch):
        parts.append(emb([r["seq"] for r in recs[i:i+batch]], device))
    return torch.cat(parts)

def train_head(e_tr, y_tr, e_val, y_val, d, device, epochs=60, bs=128, lr=3e-4, patience=12):
    head = MICHead(d).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    best_val, wait, best_st = float("inf"), 0, None
    idx = list(range(len(y_tr)))
    for _ in range(epochs):
        head.train(); random.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i+bs]
            F.huber_loss(head(e_tr[b]), y_tr[b]).backward()
            opt.step(); opt.zero_grad()
        head.eval()
        with torch.no_grad(): vl = F.huber_loss(head(e_val), y_val).item()
        if vl < best_val - 1e-4: best_val = vl; wait = 0; best_st = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_st: head.load_state_dict(best_st)
    return head

def finetune(src_head, e_sup, y_sup, device, epochs=200, lr=5e-5):
    head = copy.deepcopy(src_head).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    head.train()
    for _ in range(epochs):
        F.huber_loss(head(e_sup), y_sup).backward(); opt.step(); opt.zero_grad()
    return head.eval()

@torch.no_grad()
def evaluate(head, e_te, y_np):
    p = head(e_te).cpu().numpy()
    r, _ = pearsonr(p, y_np); rho, _ = spearmanr(p, y_np)
    return ({"pearson": float(r), "spearman": float(rho),
             "rmse": float(np.sqrt(np.mean((p-y_np)**2))), "n": len(y_np)},
            {"pred": p.tolist(), "actual": y_np.tolist()})

FACTORIES = {
    "jepa":      lambda d: JEPAEmbedder(d),
    "esm2":      lambda d: ESM2Embedder(d, "esm2_t12_35M"),
    "mlm":       lambda d: MLMEmbedder(d),
    "esm2_650m": lambda d: ESM2Embedder(d, "esm2_t33_650M"),
}

def run_model(model_name, device, grampa):
    out_dir = PROJECT_ROOT / "eval_results" / "fewshot_v2" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    mf = out_dir / "metrics.json"; pf = out_dir / "predictions.json"
    metrics = json.loads(mf.read_text()) if mf.exists() else {}
    preds   = json.loads(pf.read_text()) if pf.exists() else {}

    emb = FACTORIES[model_name](device).to(device).eval()
    d   = emb.d_model
    print(f"\n{'='*50}\n{model_name} (d={d})\n{'='*50}")

    for src, tgt in PAIRS:
        pk = f"{src}→{tgt}"
        pm = metrics.setdefault(pk, {}); pp = preds.setdefault(pk, {})
        for seed in SEEDS:
            sk = str(seed)
            sm = pm.setdefault(sk, {}); sp = pp.setdefault(sk, {})
            needed = [k for k in K_VALUES if str(k) not in sm]
            if not needed:
                print(f"  [skip] {pk} seed={seed}"); continue
            print(f"\n  {pk}  seed={seed}")
            src_tr, src_val, _ = load_species(grampa, src, seed)
            tgt_tr, _, tgt_te  = load_species(grampa, tgt, seed)
            e_src_tr  = precompute(emb, src_tr,  device)
            e_src_val = precompute(emb, src_val, device)
            e_tgt_te  = precompute(emb, tgt_te,  device)
            y_src_tr  = torch.tensor([r["log2_mic"] for r in src_tr],  dtype=torch.float32, device=device)
            y_src_val = torch.tensor([r["log2_mic"] for r in src_val], dtype=torch.float32, device=device)
            y_tgt     = np.array([r["log2_mic"] for r in tgt_te])
            src_head  = train_head(e_src_tr, y_src_tr, e_src_val, y_src_val, d, device)
            if 0 in needed:
                m, p = evaluate(src_head, e_tgt_te, y_tgt)
                sm["0"] = m; sp["0"] = p
                print(f"    k=0   r={m['pearson']:.3f}")
                mf.write_text(json.dumps(metrics, indent=2)); pf.write_text(json.dumps(preds, indent=2))
            rng = random.Random(seed); tgt_shuf = tgt_tr.copy(); rng.shuffle(tgt_shuf)
            for k in sorted(kk for kk in needed if kk > 0):
                sup = tgt_shuf[:k]
                e_sup = precompute(emb, sup, device)
                y_sup = torch.tensor([r["log2_mic"] for r in sup], dtype=torch.float32, device=device)
                hft = finetune(src_head, e_sup, y_sup, device)
                m, p = evaluate(hft, e_tgt_te, y_tgt)
                sm[str(k)] = m; sp[str(k)] = p
                print(f"    k={k:<3}  r={m['pearson']:.3f}")
                mf.write_text(json.dumps(metrics, indent=2)); pf.write_text(json.dumps(preds, indent=2))

    del emb; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"Done → {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=["esm2", "mlm"],
                        choices=list(FACTORIES))
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grampa = PROJECT_ROOT / "data" / "grampa.csv"
    for m in args.models:
        run_model(m, device, grampa)
