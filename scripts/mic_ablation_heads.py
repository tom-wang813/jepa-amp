"""
MIC ablation: no-conditioning vs per-species separate heads.

Uses the same frozen encoder+adapter from formal_mic_868k_transformer,
then trains two ablation conditions:

  A) no_cond   — one shared MLPHead on all species, no bacteria_idx used
  B) per_species — 20 separate MLPHeads, one per species, trained on that
                   species' data only; evaluated per-species and aggregated

Test split identical to formal models (seed=42, val_ratio=0.1, test_ratio=0.1).

Outputs: eval_results/mic_ablation/metrics.json
         eval_results/mic_ablation/SUMMARY.md
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.supervised_dataset import load_grampa, GRAMPA_TOP20, N_BACTERIA
from src.data.tokenizer import PAD_ID
from src.models.pretrain_utils import load_pretrained_encoder
from src.models.generator import Adapter

# ── constants matching formal run ────────────────────────────────────────────
PRETRAIN_CKPT  = PROJECT_ROOT / "checkpoints" / "jepa_pretrain_868k" / "last_jepa.pt"
FORMAL_CKPT    = PROJECT_ROOT / "checkpoints" / "formal_mic_868k_transformer" / "best_model.pt"
GRAMPA_CSV     = PROJECT_ROOT / "data" / "grampa.csv"
OUT_DIR        = PROJECT_ROOT / "eval_results" / "mic_ablation"
D_MODEL        = 384
HIDDEN         = 256
DROPOUT        = 0.4
ADAPTER_BN     = 64
BATCH          = 512
EPOCHS         = 60
LR             = 3e-4
PATIENCE       = 10
SEED           = 42
MAX_LEN        = 50
LABEL_NOISE    = 0.3


# ── model pieces ─────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    def __init__(self, d_in: int = D_MODEL, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.net  = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)).squeeze(-1)


# ── encode all sequences with frozen encoder+adapter ─────────────────────────

def build_encoder(device: torch.device):
    encoder, _ = load_pretrained_encoder(str(PRETRAIN_CKPT), device)
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    adapter = Adapter(D_MODEL, bottleneck=ADAPTER_BN).to(device)
    # load adapter weights from formal checkpoint
    state = torch.load(FORMAL_CKPT, map_location=device, weights_only=False)["model_state"]
    adapter.load_state_dict({
        k.replace("adapter.", ""): v
        for k, v in state.items() if k.startswith("adapter.")
    })
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    return encoder, adapter


@torch.no_grad()
def embed_dataset(dataset, encoder, adapter, device: torch.device, batch_size: int = 256):
    """Mean-pool token embeddings after adapter. Dataset returns {input_ids, bacteria_idx, log2_mic}."""
    from src.data.supervised_dataset import collate_supervised
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_supervised, num_workers=0)
    embs, labs, bidxs = [], [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = (ids == PAD_ID)
        h    = encoder(ids)                              # (B, L, D)
        h    = adapter(h)                                # (B, L, D)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1)  # (B, D)
        embs.append(pooled.cpu())
        labs.append(batch["log2_mic"].cpu())
        bidxs.append(batch["bacteria_idx"].cpu())
    return torch.cat(embs), torch.cat(labs), torch.cat(bidxs)


# ── train a single MLPHead on (emb_tensor, label_tensor) ─────────────────────

def train_head(
    emb_tr: torch.Tensor, y_tr: torch.Tensor,
    emb_val: torch.Tensor, y_val: torch.Tensor,
    device: torch.device,
    epochs: int = EPOCHS, lr: float = LR, patience: int = PATIENCE,
    label_noise: float = LABEL_NOISE,
) -> MLPHead:
    head = MLPHead().to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds_tr  = TensorDataset(emb_tr, y_tr)
    ds_val = TensorDataset(emb_val, y_val)
    dl_tr  = DataLoader(ds_tr,  batch_size=BATCH, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False)

    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        head.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            if label_noise > 0:
                yb = yb + torch.randn_like(yb) * label_noise
            loss = F.huber_loss(head(xb), yb, delta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            val_loss = sum(
                F.huber_loss(head(xb.to(device)), yb.to(device), delta=1.0).item() * len(xb)
                for xb, yb in dl_val
            ) / len(ds_val)

        if val_loss < best_val - 1e-4:
            best_val, best_state, wait = val_loss, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break

    head.load_state_dict(best_state)
    return head


def evaluate_head(head: MLPHead, emb: torch.Tensor, y: torch.Tensor, device: torch.device):
    head.eval()
    with torch.no_grad():
        preds = head(emb.to(device)).cpu().numpy()
    trues = y.numpy()
    if len(trues) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "rmse": float("nan"), "n": len(trues)}
    r, _  = pearsonr(trues, preds)
    rho,_ = spearmanr(trues, preds)
    rmse  = float(np.sqrt(np.mean((trues - preds) ** 2)))
    return {"pearson": round(float(r), 6), "spearman": round(float(rho), 6),
            "rmse": round(rmse, 6), "n": int(len(trues))}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 epochs, 3 species only")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load data ──────────────────────────────────────────────────────────
    print("Loading GRAMPA split (seed=42)...")
    train_ds, val_ds, test_ds = load_grampa(
        GRAMPA_CSV, max_len=MAX_LEN, val_ratio=0.1, test_ratio=0.1,
        seed=SEED, label_noise_std=0.0,   # noise added manually during training
    )
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ── build encoder+adapter ──────────────────────────────────────────────
    print("Building frozen encoder+adapter (from formal checkpoint)...")
    encoder, adapter = build_encoder(device)

    # ── embed everything ───────────────────────────────────────────────────
    print("Embedding train / val / test sets...")
    emb_tr, y_tr, bidx_tr = embed_dataset(train_ds, encoder, adapter, device)
    emb_va, y_va, bidx_va = embed_dataset(val_ds,   encoder, adapter, device)
    emb_te, y_te, bidx_te = embed_dataset(test_ds,  encoder, adapter, device)
    print(f"  Embeddings: train={emb_tr.shape}, val={emb_va.shape}, test={emb_te.shape}")

    results = {}

    # ── Condition A: no_cond ───────────────────────────────────────────────
    print("\n=== A) No-conditioning (shared head, all species, no bacteria_emb) ===")
    head_nc = train_head(emb_tr, y_tr, emb_va, y_va, device)
    overall_nc = evaluate_head(head_nc, emb_te, y_te, device)
    print(f"  Overall: Pearson={overall_nc['pearson']:.4f}  RMSE={overall_nc['rmse']:.4f}")

    per_species_nc = {}
    for sp_idx, sp_name in enumerate(GRAMPA_TOP20):
        mask = (bidx_te == sp_idx)
        if mask.sum() < 3:
            continue
        m = evaluate_head(head_nc, emb_te[mask], y_te[mask], device)
        per_species_nc[sp_name] = m
        print(f"  {sp_name:30s}  Pearson={m['pearson']:.3f}  n={m['n']}")

    results["no_cond"] = {"overall": overall_nc, "per_species": per_species_nc}

    # ── Condition B: per_species ───────────────────────────────────────────
    print("\n=== B) Per-species separate heads ===")
    species_to_eval = GRAMPA_TOP20
    if args.smoke:
        species_to_eval = GRAMPA_TOP20[:3]

    per_species_ps = {}
    all_preds, all_trues = [], []

    for sp_idx, sp_name in enumerate(species_to_eval):
        tr_mask = (bidx_tr == sp_idx)
        va_mask = (bidx_va == sp_idx)
        te_mask = (bidx_te == sp_idx)

        n_tr = int(tr_mask.sum())
        n_te = int(te_mask.sum())
        if n_tr < 10 or n_te < 3:
            print(f"  {sp_name:30s}  SKIP (n_train={n_tr})")
            continue

        sp_head = train_head(
            emb_tr[tr_mask], y_tr[tr_mask],
            emb_va[va_mask] if va_mask.sum() >= 3 else emb_tr[tr_mask][:5],
            y_va[va_mask]   if va_mask.sum() >= 3 else y_tr[tr_mask][:5],
            device,
        )
        m = evaluate_head(sp_head, emb_te[te_mask], y_te[te_mask], device)
        per_species_ps[sp_name] = m
        print(f"  {sp_name:30s}  Pearson={m['pearson']:.3f}  RMSE={m['rmse']:.3f}  n={m['n']}")

        sp_head.eval()
        with torch.no_grad():
            preds = sp_head(emb_te[te_mask].to(device)).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_trues.extend(y_te[te_mask].numpy().tolist())

    if len(all_trues) > 2:
        r, _  = pearsonr(all_trues, all_preds)
        rho,_ = spearmanr(all_trues, all_preds)
        rmse  = float(np.sqrt(np.mean((np.array(all_trues) - np.array(all_preds))**2)))
        overall_ps = {"pearson": round(float(r), 6), "spearman": round(float(rho), 6),
                      "rmse": round(rmse, 6), "n": len(all_trues)}
    else:
        overall_ps = {}

    print(f"\n  Overall (concat): Pearson={overall_ps.get('pearson', 'N/A'):.4f}")
    results["per_species_heads"] = {"overall": overall_ps, "per_species": per_species_ps}

    # ── save ───────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    # ── summary ────────────────────────────────────────────────────────────
    formal_pearson = 0.640230  # from formal_mic_868k_transformer test_metrics.json
    lines = [
        "# MIC Ablation: bacteria conditioning vs per-species heads\n",
        f"Formal checkpoint adapter reused (frozen). Test split: seed=42.\n",
        "",
        "## Overall Pearson",
        f"| Model | Pearson | RMSE |",
        f"|---|---:|---:|",
        f"| SpecFiLM-Transformer (formal) | {formal_pearson:.4f} | 0.6266 |",
        f"| No-conditioning (ablation A)  | {overall_nc['pearson']:.4f} | {overall_nc['rmse']:.4f} |",
        f"| Per-species heads (ablation B)| {overall_ps.get('pearson', float('nan')):.4f} | {overall_ps.get('rmse', float('nan')):.4f} |",
        "",
        "## Per-Species Pearson",
        f"| Species | SpecFiLM (formal) | No-cond | Per-species |",
        f"|---|---:|---:|---:|",
    ]

    formal_per_species = {
        "E. coli": 0.612, "S. aureus": 0.713, "P. aeruginosa": 0.507,
        "C. albicans": 0.481, "B. subtilis": 0.627, "S. typhimurium": 0.750,
        "M. luteus": 0.605, "S. epidermidis": 0.827, "K. pneumoniae": 0.521,
        "E. faecalis": 0.636, "B. cereus": 0.360, "L. monocytogenes": 0.590,
        "A. baumannii": 0.742, "B. megaterium": 0.672, "S. enterica": 0.533,
        "E. cloacae": 0.344, "B. pyocyaneus": 0.490, "E. faecium": 0.433,
        "P. syringae": 0.521, "S. cerevisiae": 0.487,
    }
    all_sp = set(list(per_species_nc.keys()) + list(per_species_ps.keys()))
    for sp in GRAMPA_TOP20:
        if sp not in all_sp:
            continue
        f_r = formal_per_species.get(sp, float("nan"))
        nc_r = per_species_nc.get(sp, {}).get("pearson", float("nan"))
        ps_r = per_species_ps.get(sp, {}).get("pearson", float("nan"))
        lines.append(f"| {sp} | {f_r:.3f} | {nc_r:.3f} | {ps_r:.3f} |")

    summary_path = OUT_DIR / "SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
