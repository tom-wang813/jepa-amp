"""
Latent-space charge interpolation experiment.

Fixes context sequences and sweeps charge target continuously from -9 to +13,
showing the generated sequences' actual charge tracks the target smoothly.

Also tests: does GRAVY track if we sweep GRAVY with charge fixed?

Outputs: eval_results/charge_interpolation/
  charge_sweep.png       — scatter: target charge vs generated charge (proposed v4)
  charge_sweep_grid.png  — grid: 5 representative contexts × 11 charge targets
  gravy_sweep.png        — scatter: target GRAVY vs generated GRAVY
  metrics.json           — R², MAE, slope per sweep
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = PROJECT_ROOT / "eval_results" / "charge_interpolation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA       = "ACDEFGHIKLMNPQRSTVWY"
POSITIVE = set("KR")
NEGATIVE = set("DE")
KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
      "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
      "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}


def physchem(seq: str) -> dict:
    n = max(len(seq), 1)
    return {
        "charge": sum(1 if c in POSITIVE else -1 if c in NEGATIVE else 0 for c in seq),
        "gravy":  sum(KD.get(c, 0) for c in seq) / n,
        "length": len(seq),
    }


def ids_to_seq(ids) -> str:
    out = []
    for t in ids:
        if t in (0, 1): break
        if 2 <= t <= 21: out.append(AA[t - 2])
    return "".join(out)


def load_model():
    from src.models.jepa import JEPA
    from src.models.generator import ConditionalGeneratorV4
    from src.models.encoder import TransformerEncoder

    pt = torch.load(PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
                    map_location=DEVICE, weights_only=False)
    jepa = JEPA(**pt["cfg"]["model"])
    enc = TransformerEncoder(**{k: pt["cfg"]["model"][k] for k in
          ["d_model","nhead","num_layers","dim_feedforward","dropout","max_seq_len"]})
    enc.load_state_dict(jepa.context_encoder.state_dict())

    gen_ckpt = torch.load(PROJECT_ROOT / "checkpoints/generator_868k_v4/best_generator.pt",
                          map_location=DEVICE, weights_only=False)
    gen_cfg  = gen_ckpt["cfg"]["generator"]
    gen = ConditionalGeneratorV4(encoder=enc, d_model=pt["cfg"]["model"]["d_model"],
                                 freeze_encoder=True, **gen_cfg).to(DEVICE)
    gen.load_state_dict(gen_ckpt["model_state"])
    gen.eval()
    return gen


def cond_vec(charge: float, gravy: float, length: float = 20.0) -> torch.Tensor:
    return torch.tensor([
        length / 50.0,
        math.tanh(charge / 5.0),
        math.tanh(gravy),
    ], dtype=torch.float32, device=DEVICE)


def load_contexts(n: int = 200):
    """Sample n diverse context sequences from the training corpus."""
    seqs = []
    with open(PROJECT_ROOT / "data/processed/amp_corpus.fasta") as f:
        for line in f:
            if not line.startswith(">"):
                s = line.strip()
                if s and all(c in AA for c in s) and len(s) >= 10:
                    seqs.append(s)
    rng = np.random.default_rng(42)
    chosen = [seqs[i] for i in rng.choice(len(seqs), min(n, len(seqs)), replace=False)]
    return chosen


@torch.no_grad()
def generate_for_cond(model, contexts, cond: torch.Tensor,
                      n_per_ctx: int = 5, temperature: float = 0.9) -> list[str]:
    from src.data.tokenizer import encode, PAD_ID
    seqs = []
    for ctx in contexts:
        enc_ids = encode(ctx[:50])
        ctx_ids = torch.tensor([enc_ids], dtype=torch.long, device=DEVICE)  # (1, L)
        for _ in range(n_per_ctx):
            ids = model.generate(ctx_ids, conditions=cond.unsqueeze(0),
                                 max_new_tokens=50, temperature=temperature, top_p=0.9)
            s = ids_to_seq(ids[0].cpu().tolist())
            if s and all(c in AA for c in s):
                seqs.append(s)
    return seqs


def sweep_charge(model, contexts, charge_targets, fixed_gravy=0.0,
                 fixed_length=20.0, n_per_ctx=3) -> dict:
    results = []
    for t_charge in charge_targets:
        cv = cond_vec(t_charge, fixed_gravy, fixed_length)
        gen = generate_for_cond(model, contexts, cv, n_per_ctx=n_per_ctx)
        for s in gen:
            p = physchem(s)
            results.append({"target_charge": t_charge, "seq": s, **p})
    return results


def sweep_gravy(model, contexts, gravy_targets, fixed_charge=5.0,
                fixed_length=20.0, n_per_ctx=3) -> dict:
    results = []
    for t_gravy in gravy_targets:
        cv = cond_vec(fixed_charge, t_gravy, fixed_length)
        gen = generate_for_cond(model, contexts, cv, n_per_ctx=n_per_ctx)
        for s in gen:
            p = physchem(s)
            results.append({"target_gravy": t_gravy, "seq": s, **p})
    return results


def r2(x, y) -> float:
    x, y = np.array(x), np.array(y)
    ss_res = ((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / max(ss_tot, 1e-9))


def main():
    print("Loading model …")
    model = load_model()

    print("Loading contexts …")
    contexts = load_contexts(50)   # 50 context seqs × 3 per cond = 150 samples/target

    charge_targets = np.linspace(-9, 13, 23).tolist()   # step=1
    gravy_targets  = np.linspace(-2.0, 2.0, 17).tolist()

    # ── charge sweep ──────────────────────────────────────────────────────────
    print(f"Sweeping charge ({len(charge_targets)} targets) …")
    c_rows = sweep_charge(model, contexts, charge_targets)
    t_charge = [r["target_charge"] for r in c_rows]
    g_charge = [r["charge"] for r in c_rows]

    from sklearn.linear_model import LinearRegression
    lm = LinearRegression().fit(np.array(t_charge).reshape(-1,1),
                                np.array(g_charge))
    slope_c = float(lm.coef_[0])
    r2_c    = r2(t_charge, g_charge)
    mae_c   = float(np.mean(np.abs(np.array(g_charge) - np.array(t_charge))))
    print(f"  Charge: R²={r2_c:.3f}  MAE={mae_c:.3f}  slope={slope_c:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(t_charge, g_charge, s=5, alpha=0.3, c="#2196F3", label="generated")
    xs = np.linspace(-9, 13, 100)
    ax.plot(xs, lm.predict(xs.reshape(-1,1)), "r-", lw=1.5, label=f"fit (slope={slope_c:.2f})")
    ax.plot(xs, xs, "k--", lw=0.8, alpha=0.5, label="ideal")
    ax.set_xlabel("Target charge"); ax.set_ylabel("Generated charge")
    ax.set_title(f"Charge control: R²={r2_c:.3f}, MAE={mae_c:.2f}")
    ax.legend(fontsize=8); plt.tight_layout()
    fig.savefig(OUT_DIR / "charge_sweep.png", dpi=150); plt.close(fig)

    # representative grid: 5 contexts × 11 charge targets
    grid_charges = np.linspace(-8, 12, 11).tolist()
    rep_ctxs = contexts[:5]
    grid_rows = sweep_charge(model, rep_ctxs, grid_charges, n_per_ctx=1)

    fig2, axes = plt.subplots(1, 5, figsize=(14, 3), sharey=True)
    for ci, ctx in enumerate(rep_ctxs):
        rows_c = [r for r in grid_rows if r in grid_rows]  # all rows for this ctx position
        rows_c = grid_rows[ci * len(grid_charges):(ci+1) * len(grid_charges)]
        tc = [r["target_charge"] for r in rows_c]
        gc = [r["charge"] for r in rows_c]
        axes[ci].plot(tc, gc, "o-", ms=5)
        axes[ci].plot(tc, tc, "k--", alpha=0.4)
        axes[ci].set_xlabel("target"); axes[ci].set_title(f"ctx {ci+1} ({len(ctx)} AA)")
    axes[0].set_ylabel("generated charge")
    fig2.suptitle("Charge control per context sequence", fontsize=11)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "charge_sweep_grid.png", dpi=150); plt.close(fig2)

    # ── GRAVY sweep ───────────────────────────────────────────────────────────
    print(f"Sweeping GRAVY ({len(gravy_targets)} targets) …")
    gv_rows = sweep_gravy(model, contexts, gravy_targets)
    t_gravy = [r["target_gravy"] for r in gv_rows]
    g_gravy = [r["gravy"] for r in gv_rows]

    r2_g  = r2(t_gravy, g_gravy)
    mae_g = float(np.mean(np.abs(np.array(g_gravy) - np.array(t_gravy))))
    print(f"  GRAVY: R²={r2_g:.3f}  MAE={mae_g:.3f}")

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.scatter(t_gravy, g_gravy, s=5, alpha=0.3, c="#FF5722")
    xs_g = np.linspace(-2, 2, 100)
    ax3.plot(xs_g, xs_g, "k--", lw=0.8, alpha=0.5)
    ax3.set_xlabel("Target GRAVY"); ax3.set_ylabel("Generated GRAVY")
    ax3.set_title(f"GRAVY control: R²={r2_g:.3f}, MAE={mae_g:.3f} (limited control)")
    plt.tight_layout()
    fig3.savefig(OUT_DIR / "gravy_sweep.png", dpi=150); plt.close(fig3)

    # ── save metrics ──────────────────────────────────────────────────────────
    out_metrics = {
        "charge_sweep": {
            "n_points": len(c_rows), "r2": r2_c, "mae": mae_c, "slope": slope_c,
            "target_range": [-9, 13], "targets": charge_targets,
        },
        "gravy_sweep": {
            "n_points": len(gv_rows), "r2": r2_g, "mae": mae_g,
            "target_range": [-2.0, 2.0], "targets": gravy_targets,
        },
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(out_metrics, f, indent=2)

    print("\nCharge sweep:", f"R²={r2_c:.3f}, MAE={mae_c:.3f}")
    print("GRAVY sweep: ", f"R²={r2_g:.3f}, MAE={mae_g:.3f}")
    print("Figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
