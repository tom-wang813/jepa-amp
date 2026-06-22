"""
Comprehensive TTT benchmark suite comparing JEPA vs MLM backbone adaptation.

Benchmarks (all use frozen task head + backbone TTT):

  B1  cross_species   — MIC regression, source→target species (geographic OOD)
  B2  amp_cls_ood     — AMP classification, AMPlify train → APD3 test (dataset OOD)
  B3  mic_lowdata     — MIC regression with only 5% training labels (label-scarce)
  B4  perplexity      — Head-free: model reconstruction loss as AMP score (zero-shot)

For B1/B2/B3: backbone is adapted K gradient steps per test sequence;
              task head is frozen throughout (ProteinTTT protocol).
For B4:       no head; pre-TTT reconstruction loss ranks sequences as AMP vs non-AMP.

Usage:
    uv run python scripts/eval_ttt_benchmarks.py --gpu 0
    uv run python scripts/eval_ttt_benchmarks.py --gpu 0 --benchmarks B1 B2 B3 B4
    uv run python scripts/eval_ttt_benchmarks.py --gpu 0 --ttt_steps 20
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "eval_results" / "ttt_benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS = [42, 123, 7]
SPECIES_PAIRS = [
    ("E. coli",       "S. aureus"),
    ("E. coli",       "P. aeruginosa"),
    ("S. aureus",     "E. coli"),
    ("S. aureus",     "P. aeruginosa"),
    ("P. aeruginosa", "E. coli"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def encode_batch(seqs: list[str], device) -> torch.Tensor:
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:50]) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out


def mean_pool(h: torch.Tensor, ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    mask = (ids != pad_id).unsqueeze(-1).float()
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1)


def random_block_mask(seq_len: int, block_size: int = 4,
                      num_target_blocks: int = 2, device="cpu"):
    n_blocks = max(1, seq_len // block_size)
    blocks = list(range(n_blocks))
    random.shuffle(blocks)
    tgt_blocks = set(blocks[:num_target_blocks])
    ctx = torch.zeros(seq_len, dtype=torch.bool, device=device)
    tgt = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for b in range(n_blocks):
        s, e = b * block_size, min((b + 1) * block_size, seq_len)
        (tgt if b in tgt_blocks else ctx)[s:e] = True
    return ctx, tgt


# ── TTT objectives ────────────────────────────────────────────────────────────

def jepa_ttt_loss(jepa, ids: torch.Tensor, n_masks: int = 3) -> torch.Tensor:
    from src.data.tokenizer import MASK_ID
    B, L = ids.shape
    loss = torch.tensor(0.0, device=ids.device)
    for _ in range(n_masks):
        ctx, tgt = random_block_mask(L, device=ids.device)
        ctx = ctx.unsqueeze(0).expand(B, -1)
        tgt = tgt.unsqueeze(0).expand(B, -1)
        masked = ids.clone(); masked[tgt] = MASK_ID
        ctx_h = jepa.context_encoder(masked)
        with torch.no_grad():
            tgt_h = jepa.target_encoder(ids)
        pred = jepa.predictor(ctx_h, ctx, tgt)
        p = F.layer_norm(pred[tgt],  pred.shape[-1:])
        t = F.layer_norm(tgt_h[tgt], tgt_h.shape[-1:])
        loss = loss + F.mse_loss(p, t)
    return loss / n_masks


def mlm_ttt_loss(mlm, ids: torch.Tensor, n_masks: int = 3) -> torch.Tensor:
    B, L = ids.shape
    loss = torch.tensor(0.0, device=ids.device)
    for _ in range(n_masks):
        _, tgt = random_block_mask(L, device=ids.device)
        tgt = tgt.unsqueeze(0).expand(B, -1)
        loss = loss + mlm(ids, tgt)["loss"]
    return loss / n_masks


def ttt_loss_fn(model, model_type: str, ids: torch.Tensor, n_masks: int = 3):
    return (jepa_ttt_loss if model_type == "jepa" else mlm_ttt_loss)(
        model, ids, n_masks)


# ── Get embedding from (possibly adapted) model ───────────────────────────────

@torch.no_grad()
def get_emb(model, model_type: str, ids: torch.Tensor) -> torch.Tensor:
    from src.data.tokenizer import PAD_ID
    enc = model.context_encoder if model_type == "jepa" else model.encoder
    h = enc(ids)
    return mean_pool(h, ids, pad_id=PAD_ID)


# ── Adapt one sequence and return embedding ───────────────────────────────────

def adapt_and_embed(model, model_type: str, seq: str, device,
                    ttt_steps: int, ttt_lr: float, n_masks: int) -> torch.Tensor:
    m = copy.deepcopy(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=ttt_lr, momentum=0.0, weight_decay=0.0)
    ids = encode_batch([seq], device)

    for _ in range(ttt_steps):
        opt.zero_grad()
        loss = ttt_loss_fn(m, model_type, ids, n_masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    m.eval()
    return get_emb(m, model_type, ids)          # (1, D)


# ── Task heads ────────────────────────────────────────────────────────────────

class RegressionHead(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class ClassificationHead(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)   # logit


# ── Generic head trainer ──────────────────────────────────────────────────────

def train_head(emb_fn, head, train_recs, val_recs, device, task="regression",
               epochs=60, bs=128, lr=3e-4, patience=12):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    best_val, wait, best_state = float("inf"), 0, None

    def run(recs, train):
        random.shuffle(recs)
        losses = []
        for i in range(0, len(recs), bs):
            batch = recs[i:i+bs]
            seqs  = [r["seq"] for r in batch]
            if task == "regression":
                y = torch.tensor([r["value"] for r in batch],
                                  dtype=torch.float32, device=device)
            else:
                y = torch.tensor([float(r["label"]) for r in batch],
                                  dtype=torch.float32, device=device)
            with torch.set_grad_enabled(train):
                emb = emb_fn(seqs)
                logit = head(emb)
                loss  = (F.huber_loss(logit, y) if task == "regression"
                         else F.binary_cross_entropy_with_logits(logit, y))
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    head.train()
    for _ in range(epochs):
        run(train_recs, True)
        head.eval()
        vl = run(val_recs, False)
        head.train()
        if vl < best_val - 1e-4:
            best_val = vl; best_state = copy.deepcopy(head.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= patience: break

    if best_state: head.load_state_dict(best_state)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_regression(emb_fn, head, recs, device):
    from scipy.stats import pearsonr, spearmanr
    head.eval(); preds, trues = [], []
    for i in range(0, len(recs), 256):
        batch = recs[i:i+256]
        with torch.no_grad():
            p = head(emb_fn([r["seq"] for r in batch])).cpu().tolist()
        preds.extend(p); trues.extend(r["value"] for r in batch)
    pr, _ = pearsonr(preds, trues); sr, _ = spearmanr(preds, trues)
    return {"pearson": float(pr), "spearman": float(sr), "n": len(trues)}


def eval_classification(emb_fn, head, recs, device):
    from sklearn.metrics import roc_auc_score, matthews_corrcoef
    head.eval(); logits, labels = [], []
    for i in range(0, len(recs), 256):
        batch = recs[i:i+256]
        with torch.no_grad():
            lg = head(emb_fn([r["seq"] for r in batch])).cpu().tolist()
        logits.extend(lg); labels.extend(r["label"] for r in batch)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    try:
        auroc = float(roc_auc_score(labels, probs))
    except Exception:
        auroc = float("nan")
    mcc = float(matthews_corrcoef(labels, preds))
    return {"auroc": auroc, "mcc": mcc, "n": len(labels)}


def eval_with_ttt(model, model_type, head, recs, device, task,
                  ttt_steps, ttt_lr, n_masks):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score, matthews_corrcoef
    head.eval()
    preds, trues = [], []
    for rec in recs:
        emb = adapt_and_embed(model, model_type, rec["seq"], device,
                              ttt_steps, ttt_lr, n_masks)
        with torch.no_grad():
            pred = head(emb).item()
        preds.append(pred)
        trues.append(rec["value"] if task == "regression" else rec["label"])

    if task == "regression":
        pr, _ = pearsonr(preds, trues); sr, _ = spearmanr(preds, trues)
        return {"pearson": float(pr), "spearman": float(sr), "n": len(trues)}
    else:
        probs = torch.sigmoid(torch.tensor(preds)).numpy()
        labels = list(trues)
        preds_bin = (probs > 0.5).astype(int)
        try: auroc = float(roc_auc_score(labels, probs))
        except: auroc = float("nan")
        mcc = float(matthews_corrcoef(labels, preds_bin))
        return {"auroc": auroc, "mcc": mcc, "n": len(trues)}


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark B1: Cross-species MIC transfer
# ══════════════════════════════════════════════════════════════════════════════

def load_grampa_species(species, seed=42, max_len=50):
    path = PROJECT_ROOT / "data" / "grampa.csv"
    recs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            seq = r["sequence"].strip().upper()
            if (r["is_modified"].strip() == "False"
                    and r["bacterium"].strip() == species
                    and 3 <= len(seq) <= max_len
                    and all(c in AA for c in seq)):
                try: recs.append({"seq": seq, "value": float(r["value"])})
                except ValueError: continue

    unique = sorted({r["seq"] for r in recs})
    rng = random.Random(seed); rng.shuffle(unique)
    n = len(unique)
    test_set = set(unique[:max(1, int(n * 0.15))])
    val_set  = set(unique[max(1, int(n*0.15)):max(1, int(n*0.15))+max(1, int(n*0.10))])
    train, val, test = [], [], []
    for r in recs:
        if r["seq"] in test_set: test.append(r)
        elif r["seq"] in val_set: val.append(r)
        else: train.append(r)
    return train, val, test


def run_B1(models_dict, device, ttt_steps, ttt_lr, n_masks, results):
    print("\n" + "═"*60 + "\nB1: Cross-species MIC transfer\n" + "═"*60)
    b1 = results.setdefault("B1_cross_species", {})
    ttt_key = f"ttt_k{ttt_steps}"

    for model_name, (model, d_model, emb_fn) in models_dict.items():
        mr = b1.setdefault(model_name, {})
        for src, tgt in SPECIES_PAIRS:
            pair_key = f"{src}→{tgt}"
            pr = mr.setdefault(pair_key, {})
            for seed in SEEDS:
                sk = str(seed)
                sr = pr.setdefault(sk, {})
                if "no_ttt" in sr and ttt_key in sr:
                    print(f"  [skip] {model_name} {pair_key} s={seed}")
                    continue

                src_tr, src_val, src_te = load_grampa_species(src, seed)
                _,      _,      tgt_te  = load_grampa_species(tgt, seed)
                print(f"\n  {model_name}  {pair_key}  seed={seed}  "
                      f"src_train={len(src_tr)} tgt_test={len(tgt_te)}")

                head = RegressionHead(d_model).to(device)
                train_head(emb_fn, head, src_tr, src_val, device, task="regression")
                head.eval()
                for p in head.parameters(): p.requires_grad_(False)

                if "no_ttt" not in sr:
                    sr["no_ttt"] = {
                        "in_domain": eval_regression(emb_fn, head, src_te, device),
                        "zero_shot": eval_regression(emb_fn, head, tgt_te, device),
                    }
                if ttt_key not in sr:
                    zs = eval_with_ttt(model, model_name, head, tgt_te, device,
                                       "regression", ttt_steps, ttt_lr, n_masks)
                    sr[ttt_key] = {
                        "in_domain": sr["no_ttt"]["in_domain"],
                        "zero_shot": zs,
                    }
                    Δ = zs["pearson"] - sr["no_ttt"]["zero_shot"]["pearson"]
                    print(f"    no_ttt={sr['no_ttt']['zero_shot']['pearson']:.3f}  "
                          f"ttt={zs['pearson']:.3f}  Δ={Δ:+.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark B2: AMP classification — AMPlify → APD3 (dataset OOD)
# ══════════════════════════════════════════════════════════════════════════════

def load_amp_cls_data():
    bdir = PROJECT_ROOT / "data" / "benchmarks"

    def fasta_seqs(path, label):
        from Bio import SeqIO
        return [{"seq": str(r.seq)[:50].upper(), "label": label}
                for r in SeqIO.parse(path, "fasta")
                if 3 <= len(str(r.seq)) <= 50 and all(c in AA for c in str(r.seq)[:50])]

    train_pos = fasta_seqs(bdir / "amplify_train_pos.fasta", 1)
    train_neg = fasta_seqs(bdir / "amplify_train_neg.fasta", 0)
    test_indist_pos = fasta_seqs(bdir / "amplify_test_pos.fasta", 1)
    test_indist_neg = fasta_seqs(bdir / "amplify_test_neg.fasta", 0)
    # APD3 OOD positives paired with in-dist negatives as proxy
    test_ood_pos = fasta_seqs(bdir / "apd3_independent_test.fasta", 1)
    test_ood_neg = random.sample(test_indist_neg,
                                  min(len(test_ood_pos), len(test_indist_neg)))

    train = train_pos + train_neg
    random.shuffle(train)
    n_val = max(1, int(len(train) * 0.1))
    val, train = train[:n_val], train[n_val:]

    return (train, val,
            test_indist_pos + test_indist_neg,
            test_ood_pos + test_ood_neg)


def run_B2(models_dict, device, ttt_steps, ttt_lr, n_masks, results):
    print("\n" + "═"*60 + "\nB2: AMP classification OOD (AMPlify→APD3)\n" + "═"*60)
    b2 = results.setdefault("B2_amp_classification", {})
    ttt_key = f"ttt_k{ttt_steps}"

    random.seed(42)
    train, val, test_indist, test_ood = load_amp_cls_data()
    print(f"  train={len(train)} val={len(val)} "
          f"test_indist={len(test_indist)} test_ood={len(test_ood)}")

    for model_name, (model, d_model, emb_fn) in models_dict.items():
        mr = b2.setdefault(model_name, {})
        if "no_ttt" in mr and ttt_key in mr:
            print(f"  [skip] {model_name}")
            continue

        print(f"\n  Training classifier head: {model_name}")
        head = ClassificationHead(d_model).to(device)
        train_head(emb_fn, head, train, val, device, task="classification")
        head.eval()
        for p in head.parameters(): p.requires_grad_(False)

        if "no_ttt" not in mr:
            mr["no_ttt"] = {
                "in_dist": eval_classification(emb_fn, head, test_indist, device),
                "ood_apd3": eval_classification(emb_fn, head, test_ood, device),
            }
            print(f"    no_ttt  in_dist AUROC={mr['no_ttt']['in_dist']['auroc']:.3f}  "
                  f"ood AUROC={mr['no_ttt']['ood_apd3']['auroc']:.3f}")

        if ttt_key not in mr:
            print(f"    ttt ({ttt_steps} steps) on {len(test_ood)} OOD seqs...")
            ood_ttt = eval_with_ttt(model, model_name, head, test_ood, device,
                                    "classification", ttt_steps, ttt_lr, n_masks)
            mr[ttt_key] = {
                "in_dist":  mr["no_ttt"]["in_dist"],
                "ood_apd3": ood_ttt,
            }
            Δ = ood_ttt["auroc"] - mr["no_ttt"]["ood_apd3"]["auroc"]
            print(f"    ttt     ood  AUROC={ood_ttt['auroc']:.3f}  Δ={Δ:+.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark B3: Low-data MIC regression (label-scarce regime)
# ══════════════════════════════════════════════════════════════════════════════

def run_B3(models_dict, device, ttt_steps, ttt_lr, n_masks, results,
           frac: float = 0.05):
    label = f"frac{int(frac*100)}pct"
    print("\n" + "═"*60 + f"\nB3: Low-data MIC ({int(frac*100)}% labels)\n" + "═"*60)
    b3 = results.setdefault(f"B3_lowdata_{label}", {})
    ttt_key = f"ttt_k{ttt_steps}"

    for model_name, (model, d_model, emb_fn) in models_dict.items():
        mr = b3.setdefault(model_name, {})
        for seed in SEEDS:
            sk = str(seed)
            sr = mr.setdefault(sk, {})
            if "no_ttt" in sr and ttt_key in sr:
                print(f"  [skip] {model_name} s={seed}")
                continue

            # Use E.coli as the source (largest species in GRAMPA)
            tr_full, val, test = load_grampa_species("E. coli", seed)
            rng = random.Random(seed + 1)
            rng.shuffle(tr_full)
            tr_small = tr_full[:max(1, int(len(tr_full) * frac))]
            print(f"\n  {model_name} seed={seed}  "
                  f"train={len(tr_small)} (of {len(tr_full)}) test={len(test)}")

            head = RegressionHead(d_model).to(device)
            train_head(emb_fn, head, tr_small, val, device, task="regression")
            head.eval()
            for p in head.parameters(): p.requires_grad_(False)

            if "no_ttt" not in sr:
                sr["no_ttt"] = eval_regression(emb_fn, head, test, device)
                print(f"    no_ttt  Pearson={sr['no_ttt']['pearson']:.3f}")

            if ttt_key not in sr:
                ttt_res = eval_with_ttt(model, model_name, head, test, device,
                                        "regression", ttt_steps, ttt_lr, n_masks)
                sr[ttt_key] = ttt_res
                Δ = ttt_res["pearson"] - sr["no_ttt"]["pearson"]
                print(f"    ttt     Pearson={ttt_res['pearson']:.3f}  Δ={Δ:+.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark B4: Head-free perplexity-based AMP scoring (zero-shot)
# ══════════════════════════════════════════════════════════════════════════════

def run_B4(models_dict, device, n_masks, results):
    """
    No task head. Use the model's own reconstruction loss on each sequence
    as an AMP-ness proxy (lower = more AMP-like = closer to training distribution).

    For JEPA: loss = MSE between predicted and target latent at masked positions.
    For MLM:  loss = CE on masked token prediction.

    Evaluate: AUROC of (-loss) as AMP vs non-AMP classifier on AMPlify test set.
    """
    from sklearn.metrics import roc_auc_score
    print("\n" + "═"*60 + "\nB4: Head-free perplexity scoring (zero-shot)\n" + "═"*60)
    b4 = results.setdefault("B4_perplexity", {})

    bdir = PROJECT_ROOT / "data" / "benchmarks"
    from Bio import SeqIO
    test_recs = (
        [{"seq": str(r.seq)[:50].upper(), "label": 1}
         for r in SeqIO.parse(bdir / "amplify_test_pos.fasta", "fasta")
         if 3 <= len(str(r.seq)) <= 50 and all(c in AA for c in str(r.seq)[:50])] +
        [{"seq": str(r.seq)[:50].upper(), "label": 0}
         for r in SeqIO.parse(bdir / "amplify_test_neg.fasta", "fasta")
         if 3 <= len(str(r.seq)) <= 50 and all(c in AA for c in str(r.seq)[:50])]
    )
    print(f"  Test set: {len(test_recs)} sequences")

    for model_name, (model, d_model, _) in models_dict.items():
        if model_name in b4:
            print(f"  [skip] {model_name}")
            continue
        model.eval()
        scores, labels = [], []
        for rec in test_recs:
            ids = encode_batch([rec["seq"]], device)
            with torch.no_grad():
                loss = ttt_loss_fn(model, model_name, ids, n_masks=n_masks).item()
            scores.append(-loss)   # higher score = lower loss = more AMP-like
            labels.append(rec["label"])

        try: auroc = float(roc_auc_score(labels, scores))
        except: auroc = float("nan")
        b4[model_name] = {"auroc": auroc, "n": len(test_recs)}
        print(f"  {model_name}  AUROC={auroc:.3f} (perplexity → AMP vs non-AMP)")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all_models(device):
    models = {}

    # JEPA
    from src.models.jepa import JEPA
    from src.data.tokenizer import PAD_ID
    ckpt = torch.load(PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
                      map_location=device, weights_only=False)
    jepa = JEPA(**ckpt["cfg"]["model"]).to(device)
    jepa.load_state_dict(ckpt["model_state"])
    d_jepa = ckpt["cfg"]["model"]["d_model"]
    def jepa_emb(seqs):
        with torch.no_grad():
            ids = encode_batch(seqs, device)
            h = jepa.context_encoder(ids)
            return mean_pool(h, ids, pad_id=PAD_ID)
    models["jepa"] = (jepa, d_jepa, jepa_emb)
    print(f"  JEPA loaded  d={d_jepa}")

    # MLM
    mlm_ckpt_path = PROJECT_ROOT / "checkpoints/mlm_pretrain_868k/best_mlm.pt"
    if mlm_ckpt_path.exists():
        from src.models.mlm import MLMModel
        ckpt = torch.load(mlm_ckpt_path, map_location=device, weights_only=False)
        cfg = {k: v for k, v in ckpt["cfg"]["model"].items()
               if k not in ("predictor_depth", "ema_decay")}
        mlm = MLMModel(**cfg).to(device)
        mlm.load_state_dict(ckpt["model_state"])
        d_mlm = cfg["d_model"]
        def mlm_emb(seqs):
            with torch.no_grad():
                ids = encode_batch(seqs, device)
                out = mlm(ids, torch.zeros_like(ids, dtype=torch.bool))
                return mean_pool(out["h"], ids, pad_id=PAD_ID)
        models["mlm"] = (mlm, d_mlm, mlm_emb)
        print(f"  MLM  loaded  d={d_mlm}")
    else:
        print(f"  MLM checkpoint not found, skipping: {mlm_ckpt_path}")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# Summary writer
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(results, ttt_steps):
    tk = f"ttt_k{ttt_steps}"
    lines = [
        f"# TTT Benchmark Summary (K={ttt_steps} steps/sequence)",
        "",
        "Frozen task head throughout. Δ = TTT minus no-TTT.",
        "",
    ]

    # B1
    if "B1_cross_species" in results:
        lines += ["## B1: Cross-species MIC Transfer (Pearson)", "",
                  "| Route | Model | No-TTT | +TTT | Δ |",
                  "|-------|-------|--------|------|---|"]
        for src, tgt in SPECIES_PAIRS:
            pk = f"{src}→{tgt}"
            for mn in ("jepa", "mlm"):
                pair = results["B1_cross_species"].get(mn, {}).get(pk, {})
                base = [v["no_ttt"]["zero_shot"]["pearson"]
                        for v in pair.values() if "no_ttt" in v]
                ttt  = [v[tk]["zero_shot"]["pearson"]
                        for v in pair.values() if tk in v]
                if not base: continue
                bm, tm = np.mean(base), np.mean(ttt) if ttt else float("nan")
                d = f"{tm-bm:+.3f}" if not np.isnan(tm) else "—"
                tm_s = f"{tm:.3f}" if not np.isnan(tm) else "—"
                lines.append(f"| {pk} | {mn} | {bm:.3f} | {tm_s} | {d} |")
        lines.append("")

    # B2
    if "B2_amp_classification" in results:
        lines += ["## B2: AMP Classification OOD / APD3 (AUROC)", "",
                  "| Model | In-dist (no-TTT) | OOD no-TTT | OOD +TTT | Δ |",
                  "|-------|-----------------|------------|----------|---|"]
        for mn in ("jepa", "mlm"):
            mr = results["B2_amp_classification"].get(mn, {})
            if not mr: continue
            b_in = mr.get("no_ttt", {}).get("in_dist", {}).get("auroc", float("nan"))
            b_od = mr.get("no_ttt", {}).get("ood_apd3", {}).get("auroc", float("nan"))
            t_od = mr.get(tk, {}).get("ood_apd3", {}).get("auroc", float("nan"))
            d = f"{t_od-b_od:+.3f}" if not np.isnan(t_od) else "—"
            t_od_str = f"{t_od:.3f}" if not np.isnan(t_od) else "—"
            lines.append(f"| {mn} | {b_in:.3f} | {b_od:.3f} | {t_od_str} | {d} |")
        lines.append("")

    # B3
    b3_key = next((k for k in results if k.startswith("B3")), None)
    if b3_key:
        lines += [f"## B3: Low-data MIC (Pearson)", "",
                  "| Model | No-TTT | +TTT | Δ |",
                  "|-------|--------|------|---|"]
        for mn in ("jepa", "mlm"):
            mr = results[b3_key].get(mn, {})
            base = [v["no_ttt"]["pearson"] for v in mr.values() if "no_ttt" in v]
            ttt  = [v[tk]["pearson"] for v in mr.values() if tk in v]
            if not base: continue
            bm = np.mean(base); tm = np.mean(ttt) if ttt else float("nan")
            d = f"{tm-bm:+.3f}" if not np.isnan(tm) else "—"
            lines.append(f"| {mn} | {bm:.3f} | {tm:.3f if not np.isnan(tm) else '—'} | {d} |")
        lines.append("")

    # B4
    if "B4_perplexity" in results:
        lines += ["## B4: Head-free Perplexity Scoring (AUROC, zero-shot)", "",
                  "| Model | AUROC |", "|-------|-------|"]
        for mn, v in results["B4_perplexity"].items():
            lines.append(f"| {mn} | {v['auroc']:.3f} |")
        lines.append("")

    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nSummary → {OUT_DIR}/SUMMARY.md")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",        type=int,   default=0)
    parser.add_argument("--ttt_steps",  type=int,   default=10)
    parser.add_argument("--ttt_lr",     type=float, default=1e-3)
    parser.add_argument("--n_masks",    type=int,   default=3)
    parser.add_argument("--benchmarks", nargs="+",
                        default=["B1", "B2", "B3", "B4"],
                        choices=["B1", "B2", "B3", "B4"],
                        help="Which benchmarks to run")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_file = OUT_DIR / "metrics.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}

    print(f"\nTTT config: steps={args.ttt_steps} lr={args.ttt_lr} "
          f"n_masks={args.n_masks} benchmarks={args.benchmarks}")

    print("\n[Loading models]")
    models = load_all_models(device)

    if "B1" in args.benchmarks:
        results = run_B1(models, device, args.ttt_steps, args.ttt_lr,
                         args.n_masks, results)
        out_file.write_text(json.dumps(results, indent=2))

    if "B2" in args.benchmarks:
        results = run_B2(models, device, args.ttt_steps, args.ttt_lr,
                         args.n_masks, results)
        out_file.write_text(json.dumps(results, indent=2))

    if "B3" in args.benchmarks:
        results = run_B3(models, device, args.ttt_steps, args.ttt_lr,
                         args.n_masks, results)
        out_file.write_text(json.dumps(results, indent=2))

    if "B4" in args.benchmarks:
        results = run_B4(models, device, args.n_masks, results)
        out_file.write_text(json.dumps(results, indent=2))

    write_summary(results, args.ttt_steps)
    print(f"\nAll done → {OUT_DIR}")


if __name__ == "__main__":
    main()
