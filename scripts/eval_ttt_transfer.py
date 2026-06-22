"""
Test-Time Training (TTT) for cross-species MIC transfer.

Follows ProteinTTT (arxiv 2411.02109): task head is frozen throughout;
only the backbone is adapted using its own pre-training objective on the
single test sequence.

Three backbones compared:
  jepa  — JEPA objective: MSE on latent representations (continuous)
  mlm   — MLM  objective: CE on masked token identities (discrete)
  esm2  — ESM-2 with MLM objective (ProteinTTT-style, optional)

The key hypothesis:
  JEPA's continuous MSE objective leads to more stable test-time adaptation
  than MLM's discrete CE objective, because:
    1. Gradients are smoother (MSE vs CE on one-hot targets)
    2. The EMA target encoder acts as a stable reference that prevents
       the representation from collapsing to memorise a single sequence.

Usage:
    uv run python scripts/eval_ttt_transfer.py --gpu 0
    uv run python scripts/eval_ttt_transfer.py --gpu 0 --ttt_steps 20 --ttt_lr 5e-4
    uv run python scripts/eval_ttt_transfer.py --gpu 0 --include_esm2
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "eval_results" / "ttt_transfer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIES_PAIRS = [
    ("E. coli",       "S. aureus"),
    ("E. coli",       "P. aeruginosa"),
    ("S. aureus",     "E. coli"),
    ("S. aureus",     "P. aeruginosa"),
    ("P. aeruginosa", "E. coli"),
]
SEEDS = [42, 123, 7]


# ── data helpers ──────────────────────────────────────────────────────────────

import csv

def load_species(csv_path, species, max_len=50, seed=42):
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
    rng = random.Random(seed)
    rng.shuffle(unique_seqs)
    n = len(unique_seqs)
    n_test = max(1, int(n * 0.15))
    n_val  = max(1, int(n * 0.10))
    test_set = set(unique_seqs[:n_test])
    val_set  = set(unique_seqs[n_test:n_test + n_val])

    train, val, test = [], [], []
    for r in recs:
        if r["seq"] in test_set:   test.append(r)
        elif r["seq"] in val_set:  val.append(r)
        else:                       train.append(r)
    return train, val, test


# ── tokenisation ──────────────────────────────────────────────────────────────

def encode_batch(seqs, device):
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out                           # (B, L)


def mean_pool(h, ids, pad_id=0):
    """Mean pool over non-padding positions."""
    mask = (ids != pad_id).unsqueeze(-1).float()  # (B, L, 1)
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, D)


# ── block masking (identical to pre-training) ─────────────────────────────────

def random_block_mask(seq_len, block_size=4, num_target_blocks=2, device="cpu"):
    """Returns context_mask and target_mask tensors of shape (seq_len,)."""
    n_blocks = max(1, seq_len // block_size)
    block_ids = list(range(n_blocks))
    random.shuffle(block_ids)
    target_blocks = set(block_ids[:num_target_blocks])

    context_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    target_mask  = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for b in range(n_blocks):
        start = b * block_size
        end   = min(start + block_size, seq_len)
        if b in target_blocks:
            target_mask[start:end] = True
        else:
            context_mask[start:end] = True
    return context_mask, target_mask


# ── TTT step functions ────────────────────────────────────────────────────────

def jepa_ttt_loss(jepa_model, ids, n_masks=3):
    """
    JEPA TTT objective: MSE between predicted and target latent representations.
    Target encoder (EMA copy) is used as the stable reference — no gradient.
    Averaged over n_masks random block masks for variance reduction.
    """
    from src.data.tokenizer import MASK_ID
    B, L = ids.shape
    total_loss = torch.tensor(0.0, device=ids.device)

    for _ in range(n_masks):
        ctx_mask, tgt_mask = random_block_mask(L, device=ids.device)
        ctx_mask = ctx_mask.unsqueeze(0).expand(B, -1)  # (B, L)
        tgt_mask = tgt_mask.unsqueeze(0).expand(B, -1)

        masked_ids = ids.clone()
        masked_ids[tgt_mask] = MASK_ID

        context_h = jepa_model.context_encoder(masked_ids)   # (B, L, D)

        with torch.no_grad():
            target_h = jepa_model.target_encoder(ids)        # (B, L, D)

        pred_h = jepa_model.predictor(context_h, ctx_mask, tgt_mask)

        p = F.layer_norm(pred_h[tgt_mask],   pred_h.shape[-1:])
        t = F.layer_norm(target_h[tgt_mask], target_h.shape[-1:])
        total_loss = total_loss + F.mse_loss(p, t)

    return total_loss / n_masks


def mlm_ttt_loss(mlm_model, ids, n_masks=3):
    """
    MLM TTT objective: CE on masked token identities.
    Averaged over n_masks random block masks.
    """
    B, L = ids.shape
    total_loss = torch.tensor(0.0, device=ids.device)

    for _ in range(n_masks):
        _, tgt_mask = random_block_mask(L, device=ids.device)
        tgt_mask = tgt_mask.unsqueeze(0).expand(B, -1)
        out = mlm_model(ids, tgt_mask)
        total_loss = total_loss + out["loss"]

    return total_loss / n_masks


# ── task head ─────────────────────────────────────────────────────────────────

class MICHead(nn.Module):
    def __init__(self, d_model, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_head(get_emb_fn, head, train_recs, val_recs, device,
               epochs=60, batch_size=128, lr=3e-4, patience=12):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    best_val, wait, best_state = float("inf"), 0, None

    def run_epoch(recs, train):
        random.shuffle(recs)
        losses = []
        for i in range(0, len(recs), batch_size):
            batch = recs[i:i + batch_size]
            seqs = [r["seq"] for r in batch]
            y = torch.tensor([r["log2_mic"] for r in batch],
                             dtype=torch.float32, device=device)
            with torch.set_grad_enabled(train):
                emb = get_emb_fn(seqs)
                pred = head(emb)
                loss = F.huber_loss(pred, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    head.train()
    for _ in range(epochs):
        run_epoch(train_recs, True)
        head.eval()
        vl = run_epoch(val_recs, False)
        head.train()
        if vl < best_val - 1e-4:
            best_val = vl
            best_state = copy.deepcopy(head.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience: break

    if best_state:
        head.load_state_dict(best_state)


def eval_head_no_ttt(get_emb_fn, head, test_recs, device):
    from scipy.stats import pearsonr, spearmanr
    head.eval()
    preds, trues = [], []
    for i in range(0, len(test_recs), 256):
        batch = test_recs[i:i + 256]
        seqs  = [r["seq"] for r in batch]
        trues.extend(r["log2_mic"] for r in batch)
        with torch.no_grad():
            preds.extend(head(get_emb_fn(seqs)).cpu().tolist())
    p, _ = pearsonr(preds, trues)
    r, _ = spearmanr(preds, trues)
    return {"pearson": float(p), "spearman": float(r), "n": len(trues)}


def eval_head_with_ttt(model, model_type, head, test_recs, device,
                       ttt_steps=10, ttt_lr=1e-3, n_masks=3):
    """
    For each test sequence:
      1. Clone the model (so original weights are not modified across sequences)
      2. Run ttt_steps gradient steps using the pre-training objective
      3. Extract embedding from the adapted encoder
      4. Predict with frozen task head
    """
    from scipy.stats import pearsonr, spearmanr
    from src.data.tokenizer import PAD_ID

    head.eval()
    preds, trues = [], []

    for rec in test_recs:
        # --- fresh copy of the model for each sequence ---
        m = copy.deepcopy(model).to(device)
        m.train()
        opt = torch.optim.SGD(m.parameters(), lr=ttt_lr, momentum=0.0,
                               weight_decay=0.0)

        ids = encode_batch([rec["seq"]], device)   # (1, L)

        # --- TTT loop ---
        for _ in range(ttt_steps):
            opt.zero_grad()
            if model_type == "jepa":
                loss = jepa_ttt_loss(m, ids, n_masks=n_masks)
            else:
                loss = mlm_ttt_loss(m, ids, n_masks=n_masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

        # --- extract embedding from adapted encoder ---
        m.eval()
        with torch.no_grad():
            if model_type == "jepa":
                h = m.context_encoder(ids)
            else:
                h = m.encoder(ids)
            emb = mean_pool(h, ids, pad_id=PAD_ID)   # (1, D)
            pred = head(emb).item()

        preds.append(pred)
        trues.append(rec["log2_mic"])

    p, _ = pearsonr(preds, trues)
    r, _ = spearmanr(preds, trues)
    return {"pearson": float(p), "spearman": float(r), "n": len(trues)}


# ── model loading ─────────────────────────────────────────────────────────────

def load_jepa(device):
    from src.models.jepa import JEPA
    ckpt = torch.load(
        PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
        map_location=device, weights_only=False,
    )
    jepa = JEPA(**ckpt["cfg"]["model"])
    jepa.load_state_dict(ckpt["model_state"])
    jepa.to(device)
    d_model = ckpt["cfg"]["model"]["d_model"]
    print(f"  Loaded JEPA  d={d_model}")
    return jepa, d_model


def load_mlm(device):
    from src.models.mlm import MLMModel
    ckpt = torch.load(
        PROJECT_ROOT / "checkpoints/mlm_pretrain_868k/best_mlm.pt",
        map_location=device, weights_only=False,
    )
    cfg = {k: v for k, v in ckpt["cfg"]["model"].items()
           if k not in ("predictor_depth", "ema_decay")}
    mlm = MLMModel(**cfg)
    mlm.load_state_dict(ckpt["model_state"])
    mlm.to(device)
    d_model = cfg["d_model"]
    print(f"  Loaded MLM   d={d_model}")
    return mlm, d_model


def load_esm2(device):
    from src.models.esm_head import load_esm2 as _load
    esm, alphabet, _ = _load("esm2_t12_35M")
    esm.to(device)
    bc = alphabet.get_batch_converter()
    d_model = 480
    print(f"  Loaded ESM-2 d={d_model}")
    return esm, alphabet, bc, d_model


def get_jepa_emb_fn(jepa, device):
    from src.data.tokenizer import PAD_ID
    def fn(seqs):
        with torch.no_grad():
            ids = encode_batch(seqs, device)
            h   = jepa.context_encoder(ids)
            return mean_pool(h, ids, pad_id=PAD_ID)
    return fn


def get_mlm_emb_fn(mlm, device):
    from src.data.tokenizer import PAD_ID
    def fn(seqs):
        with torch.no_grad():
            ids = encode_batch(seqs, device)
            out = mlm(ids, torch.zeros_like(ids, dtype=torch.bool))
            return mean_pool(out["h"], ids, pad_id=PAD_ID)
    return fn


def get_esm2_emb_fn(esm, alphabet, bc, device):
    pad_idx = alphabet.padding_idx
    def fn(seqs):
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = bc(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = esm(tokens, repr_layers=[12], return_contacts=False)
        h = out["representations"][12]
        return mean_pool(h, tokens, pad_id=pad_idx)
    return fn


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",          type=int,   default=0)
    parser.add_argument("--ttt_steps",    type=int,   default=10,
                        help="Gradient steps per test sequence")
    parser.add_argument("--ttt_lr",       type=float, default=1e-3,
                        help="Learning rate for TTT optimizer")
    parser.add_argument("--n_masks",      type=int,   default=3,
                        help="Random masks averaged per TTT step")
    parser.add_argument("--include_esm2", action="store_true",
                        help="Also run ESM-2 + TTT (slower)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grampa = PROJECT_ROOT / "data" / "grampa.csv"
    out_file = OUT_DIR / "metrics.json"

    results: dict = json.loads(out_file.read_text()) if out_file.exists() else {}
    cfg_tag = f"steps{args.ttt_steps}_lr{args.ttt_lr}_masks{args.n_masks}"
    print(f"\nTTT config: {cfg_tag}")

    # ── load models ───────────────────────────────────────────────────────────
    print("\n[Loading models]")
    jepa, d_jepa = load_jepa(device)
    mlm,  d_mlm  = load_mlm(device)

    esm2_bundle = None
    if args.include_esm2:
        esm2_bundle = load_esm2(device)

    model_configs = [
        ("jepa", jepa, d_jepa, get_jepa_emb_fn(jepa, device)),
        ("mlm",  mlm,  d_mlm,  get_mlm_emb_fn(mlm,  device)),
    ]
    if esm2_bundle:
        esm, alph, bc, d_esm = esm2_bundle
        model_configs.append(("esm2", esm, d_esm, get_esm2_emb_fn(esm, alph, bc, device)))

    # ── evaluation loop ───────────────────────────────────────────────────────
    for model_name, model, d_model, emb_fn in model_configs:
        print(f"\n{'='*60}\n{model_name.upper()}  d={d_model}\n{'='*60}")
        model_res = results.setdefault(model_name, {})

        for src_species, tgt_species in SPECIES_PAIRS:
            pair_key = f"{src_species}→{tgt_species}"
            pair_res = model_res.setdefault(pair_key, {})

            for seed in SEEDS:
                seed_key = str(seed)
                ttt_key  = f"ttt_{cfg_tag}"

                # Check if both no-ttt and ttt results exist
                seed_res = pair_res.setdefault(seed_key, {})
                have_base = "no_ttt" in seed_res
                have_ttt  = ttt_key in seed_res
                if have_base and have_ttt:
                    r_b = seed_res["no_ttt"]["zero_shot"]["pearson"]
                    r_t = seed_res[ttt_key]["zero_shot"]["pearson"]
                    print(f"  [skip] {pair_key} s={seed}  "
                          f"no_ttt={r_b:.3f}  ttt={r_t:.3f}")
                    continue

                print(f"\n  {pair_key}  seed={seed}")
                src_tr, src_val, src_te = load_species(grampa, src_species, seed=seed)
                _,      _,       tgt_te = load_species(grampa, tgt_species, seed=seed)
                print(f"    src train={len(src_tr)}  tgt test={len(tgt_te)}")

                # Train task head on source species (frozen encoder)
                head = MICHead(d_model).to(device)
                for p in head.parameters(): p.requires_grad_(True)
                train_head(emb_fn, head, src_tr, src_val, device)
                head.eval()
                for p in head.parameters(): p.requires_grad_(False)

                # Baseline: no TTT
                if not have_base:
                    in_dom  = eval_head_no_ttt(emb_fn, head, src_te, device)
                    zero_sh = eval_head_no_ttt(emb_fn, head, tgt_te, device)
                    seed_res["no_ttt"] = {
                        "in_domain": in_dom, "zero_shot": zero_sh,
                    }
                    print(f"    [no-ttt] in-domain={in_dom['pearson']:.3f}  "
                          f"zero-shot={zero_sh['pearson']:.3f}")

                # TTT: adapt backbone on each test sequence
                if not have_ttt:
                    print(f"    [ttt   ] running {args.ttt_steps} steps/seq "
                          f"on {len(tgt_te)} target seqs ...")
                    in_dom_ttt  = eval_head_no_ttt(emb_fn, head, src_te, device)
                    zero_sh_ttt = eval_head_with_ttt(
                        model, model_name, head, tgt_te, device,
                        ttt_steps=args.ttt_steps,
                        ttt_lr=args.ttt_lr,
                        n_masks=args.n_masks,
                    )
                    seed_res[ttt_key] = {
                        "in_domain": in_dom_ttt, "zero_shot": zero_sh_ttt,
                    }
                    delta = zero_sh_ttt["pearson"] - seed_res["no_ttt"]["zero_shot"]["pearson"]
                    print(f"    [ttt   ] zero-shot={zero_sh_ttt['pearson']:.3f}  "
                          f"Δ={delta:+.3f}")

                out_file.write_text(json.dumps(results, indent=2))

    _write_summary(results, cfg_tag, args.ttt_steps)
    print(f"\nDone. Results → {OUT_DIR}")


def _write_summary(results, cfg_tag, ttt_steps):
    ttt_key = f"ttt_{cfg_tag}"

    def agg(pair_res, seed_key, split, metric="pearson"):
        vals = []
        for v in pair_res.values():
            if seed_key in v and split in v[seed_key] and metric in v[seed_key][split]:
                vals.append(v[seed_key][split][metric])
        if not vals: return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    lines = [
        f"# TTT Transfer: JEPA vs MLM ({ttt_steps} steps per sequence)",
        "",
        "Frozen task head. TTT adapts backbone on each target sequence individually.",
        "3-seed mean ± std. Δ = TTT − no-TTT on zero-shot Pearson.",
        "",
        "| Route | Model | No-TTT | +TTT | Δ |",
        "|-------|-------|--------|------|---|",
    ]

    for src, tgt in SPECIES_PAIRS:
        pair_key = f"{src}→{tgt}"
        for model_name in ("jepa", "mlm", "esm2"):
            pair_res = results.get(model_name, {}).get(pair_key, {})
            if not pair_res: continue

            base_vals, ttt_vals = [], []
            for sv in pair_res.values():
                if "no_ttt" in sv:
                    base_vals.append(sv["no_ttt"]["zero_shot"]["pearson"])
                if ttt_key in sv:
                    ttt_vals.append(sv[ttt_key]["zero_shot"]["pearson"])

            if not base_vals: continue
            bm = np.mean(base_vals); bs = np.std(base_vals)
            tm = np.mean(ttt_vals) if ttt_vals else float("nan")
            ts = np.std(ttt_vals) if ttt_vals else float("nan")
            delta = tm - bm if not np.isnan(tm) else float("nan")

            ttt_str = f"{tm:.3f}±{ts:.3f}" if not np.isnan(tm) else "—"
            d_str   = f"{delta:+.3f}" if not np.isnan(delta) else "—"
            lines.append(f"| {pair_key} | {model_name} | "
                         f"{bm:.3f}±{bs:.3f} | {ttt_str} | {d_str} |")

    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT_DIR}/SUMMARY.md")


if __name__ == "__main__":
    main()
