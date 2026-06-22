"""
MC-Dropout uncertainty quantification on the locked formal MIC checkpoints.

Uses checkpoints/formal_mic_868k_transformer/ (the evidence-locked artifact).
Evaluates:
  1. Standard inference (model.eval()) — should match locked test_metrics.json
  2. MC-Dropout mean prediction (T=50 passes) — expected RMSE improvement
  3. Calibration: does MC-Dropout std correlate with actual error?
     — Pearson between |error| and std; reliability curve

Outputs: eval_results/mc_dropout_formal/
  metrics.json          — paired comparison table
  calibration.png       — uncertainty vs absolute error scatter
  reliability_curve.png — expected calibration error (ECE-style)
  SUMMARY.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR  = PROJECT_ROOT / "eval_results" / "mc_dropout_formal"
OUT_DIR.mkdir(parents=True, exist_ok=True)
T        = 50   # MC-Dropout forward passes


def load_formal_model():
    """Load the evidence-locked JEPA Transformer MIC model."""
    from src.models.jepa import JEPA
    from src.models.supervised_head import JEPAMICPredictor
    from src.data.supervised_dataset import N_BACTERIA

    ckpt_dir = PROJECT_ROOT / "checkpoints" / "formal_mic_868k_transformer"
    cfg_path  = ckpt_dir / "config_resolved.yaml"
    ckpt_path = ckpt_dir / "best_model.pt"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    pt_ckpt = torch.load(PROJECT_ROOT / cfg.get("pretrain_checkpoint",
                         "checkpoints/jepa_pretrain_868k/last_jepa.pt"),
                         map_location=DEVICE, weights_only=False)
    jepa = JEPA(**pt_ckpt["cfg"]["model"])
    jepa.load_state_dict(pt_ckpt["model_state"])

    head_cfg  = cfg["head"].copy()
    head_type = head_cfg.pop("head_type", "transformer")
    model = JEPAMICPredictor(
        encoder=jepa.context_encoder,
        d_model=pt_ckpt["cfg"]["model"]["d_model"],
        n_bacteria=N_BACTERIA,
        head_type=head_type,
        freeze_encoder=True,
        **head_cfg,
    ).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state"])
    return model, cfg


def load_test_loader(cfg: dict):
    from src.data.supervised_dataset import load_grampa, collate_supervised
    from torch.utils.data import DataLoader

    data_cfg = cfg.get("data", cfg)
    _, _, test_ds = load_grampa(
        PROJECT_ROOT / data_cfg["grampa_csv"],
        val_ratio=data_cfg.get("val_ratio", 0.1),
        test_ratio=data_cfg.get("test_ratio", 0.1),
        seed=data_cfg.get("seed", 42),
        label_noise_std=0.0,   # no noise at evaluation
    )
    return DataLoader(test_ds, batch_size=256, shuffle=False,
                      collate_fn=collate_supervised, num_workers=0)


def enable_dropout(model):
    """Put model in eval mode but keep dropout layers active."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def run_inference(model, loader, mc: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (preds, true, stds). stds is zeros for standard inference."""
    all_preds, all_true, all_stds = [], [], []
    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        bact  = batch["bacteria_idx"].to(DEVICE)
        true  = batch["log2_mic"].numpy()

        if mc:
            enable_dropout(model)
            samples = torch.stack(
                [model(ids, bact) for _ in range(T)], dim=0
            )  # (T, B)
            pred = samples.mean(0).cpu().numpy()
            std  = samples.std(0).cpu().numpy()
        else:
            model.eval()
            pred = model(ids, bact).cpu().numpy()
            std  = np.zeros_like(pred)

        all_preds.append(pred)
        all_true.append(true)
        all_stds.append(std)

    return (np.concatenate(all_preds),
            np.concatenate(all_true),
            np.concatenate(all_stds))


def compute_metrics(preds, true):
    rmse = float(np.sqrt(np.mean((preds - true) ** 2)))
    mae  = float(np.mean(np.abs(preds - true)))
    r, _ = pearsonr(preds, true)
    rho, _ = spearmanr(preds, true)
    return {"rmse": rmse, "mae": mae, "pearson": float(r), "spearman": float(rho)}


def plot_calibration(stds: np.ndarray, errors: np.ndarray, path: Path) -> dict:
    """Scatter of MC-Dropout std vs absolute error + Pearson."""
    r, pval = pearsonr(stds, errors)
    rho, _  = spearmanr(stds, errors)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(stds, errors, s=4, alpha=0.25, c="#9C27B0")
    # trend line
    m, b = np.polyfit(stds, errors, 1)
    xs = np.linspace(stds.min(), stds.max(), 100)
    ax.plot(xs, m * xs + b, "r-", lw=1.5)
    ax.set_xlabel("MC-Dropout std (uncertainty)")
    ax.set_ylabel("|prediction error|")
    ax.set_title(f"Uncertainty calibration\nPearson={r:.3f}  p={pval:.1e}  Spearman={rho:.3f}")
    plt.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  saved: {path.name}")
    return {"pearson_uncertainty_error": float(r), "pval": float(pval),
            "spearman_uncertainty_error": float(rho)}


def plot_reliability(stds: np.ndarray, errors: np.ndarray, n_bins: int, path: Path) -> None:
    """Bin by uncertainty; show mean|error| per bin."""
    q = np.percentile(stds, np.linspace(0, 100, n_bins + 1))
    bin_centers, bin_errs, bin_stds = [], [], []
    for lo, hi in zip(q[:-1], q[1:]):
        mask = (stds >= lo) & (stds <= hi)
        if mask.sum() < 5:
            continue
        bin_centers.append(stds[mask].mean())
        bin_errs.append(errors[mask].mean())
        bin_stds.append(stds[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(len(bin_centers)), bin_errs, color="#FF9800", alpha=0.7,
           label="mean |error| per uncertainty bin")
    ax.plot(range(len(bin_centers)), bin_stds, "o-", color="navy",
            ms=5, label="mean MC-std per bin")
    ax.set_xticks(range(len(bin_centers)))
    ax.set_xticklabels([f"{c:.2f}" for c in bin_centers], rotation=45, fontsize=7)
    ax.set_xlabel("Uncertainty bin (MC-Dropout std)")
    ax.set_ylabel("Mean absolute error / std")
    ax.set_title("Reliability: higher uncertainty → higher error?")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  saved: {path.name}")


def main():
    print("Loading formal Transformer MIC model …")
    model, cfg = load_formal_model()

    print("Loading test set …")
    loader = load_test_loader(cfg)

    print("Standard inference …")
    std_preds, true, _ = run_inference(model, loader, mc=False)
    std_m = compute_metrics(std_preds, true)
    print(f"  Standard: RMSE={std_m['rmse']:.4f}  Pearson={std_m['pearson']:.4f}")

    print(f"MC-Dropout inference (T={T}) …")
    mc_preds, _, mc_stds = run_inference(model, loader, mc=True)
    mc_m = compute_metrics(mc_preds, true)
    print(f"  MC-Dropout: RMSE={mc_m['rmse']:.4f}  Pearson={mc_m['pearson']:.4f}")

    rmse_delta = mc_m["rmse"] - std_m["rmse"]
    rmse_delta_pct = 100 * rmse_delta / std_m["rmse"]
    print(f"  RMSE delta: {rmse_delta:+.4f} ({rmse_delta_pct:+.1f}%)")

    errors = np.abs(mc_preds - true)
    print("Calibration plot …")
    cal_m = plot_calibration(mc_stds, errors, OUT_DIR / "calibration.png")
    plot_reliability(mc_stds, errors, n_bins=10, path=OUT_DIR / "reliability_curve.png")

    # ── save ──────────────────────────────────────────────────────────────────
    metrics = {
        "checkpoint": "checkpoints/formal_mic_868k_transformer/best_model.pt",
        "n_test": int(len(true)),
        "mc_samples": T,
        "standard_inference": std_m,
        "mc_dropout_inference": mc_m,
        "rmse_delta": float(rmse_delta),
        "rmse_delta_pct": float(rmse_delta_pct),
        "calibration": cal_m,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    summary = [
        "# MC-Dropout Formal Evaluation",
        "",
        f"Checkpoint: `checkpoints/formal_mic_868k_transformer/best_model.pt`",
        f"MC samples (T): {T}",
        "",
        "## Prediction Quality",
        "| Mode | RMSE | MAE | Pearson | Spearman |",
        "|---|---|---|---|---|",
        f"| Standard | {std_m['rmse']:.4f} | {std_m['mae']:.4f} | {std_m['pearson']:.4f} | {std_m['spearman']:.4f} |",
        f"| MC-Dropout | {mc_m['rmse']:.4f} | {mc_m['mae']:.4f} | {mc_m['pearson']:.4f} | {mc_m['spearman']:.4f} |",
        f"| Δ (MC - Std) | {rmse_delta:+.4f} | — | — | — |",
        "",
        "## Calibration",
        f"Uncertainty-error Pearson: {cal_m['pearson_uncertainty_error']:.3f}  "
        f"(p={cal_m['pval']:.1e})",
        f"Uncertainty-error Spearman: {cal_m['spearman_uncertainty_error']:.3f}",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    print("\nDone. All results in", OUT_DIR)


if __name__ == "__main__":
    main()
