"""
MC-Dropout evaluation for MIC prediction.

Compares three inference modes:
  1. Standard (deterministic, model.eval())
  2. MC-Dropout mean  (30 stochastic forward passes, mean)
  3. MC-Dropout std   (uncertainty — does std correlate with error?)

Also produces calibration plots: reliability curves and error-vs-uncertainty.

Usage:
  uv run python scripts/eval_mic_mc_dropout.py [--gpu 1]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "eval_results" / "mic_mc_dropout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MC_SAMPLES = 30

MODELS = {
    "MLP (FiLM)": {
        "ckpt": PROJECT_ROOT / "checkpoints/mic_868k_mlp/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/mic_868k.yaml",
        "type": "jepa",
    },
    "Transformer": {
        "ckpt": PROJECT_ROOT / "checkpoints/mic_868k_transformer/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/mic_868k_transformer.yaml",
        "type": "jepa",
    },
}


def load_jepa_model(ckpt_path, cfg_path, device):
    from src.models.jepa import JEPA
    from src.models.supervised_head import JEPAMICPredictor
    from src.data.supervised_dataset import N_BACTERIA

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    pretrain_ckpt = torch.load(
        PROJECT_ROOT / cfg["pretrain_checkpoint"], map_location=device, weights_only=False)
    jepa = JEPA(**pretrain_ckpt["cfg"]["model"])
    jepa.load_state_dict(pretrain_ckpt["model_state"])

    head_cfg = cfg["head"].copy()
    head_type = head_cfg.pop("head_type", "mlp")
    model = JEPAMICPredictor(
        encoder=jepa.context_encoder,
        d_model=pretrain_ckpt["cfg"]["model"]["d_model"],
        n_bacteria=N_BACTERIA,
        head_type=head_type,
        freeze_encoder=True,
        **head_cfg,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, pretrain_ckpt["cfg"]["model"].get("max_seq_len", 52)


def run_inference(model, test_ds, device, batch_size=128):
    """Returns (std_preds, mc_preds, mc_stds, targets, bact_idxs)."""
    from src.data.supervised_dataset import collate_supervised

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_supervised)

    std_preds, mc_means, mc_stds, targets, bact_idxs = [], [], [], [], []

    for batch in loader:
        ids  = batch["input_ids"].to(device)
        bidx = batch["bacteria_idx"].to(device)

        # standard deterministic inference
        model.eval()
        with torch.no_grad():
            p_std = model(ids, bidx).cpu().float()
        std_preds.extend(p_std.tolist())

        # MC-Dropout inference
        mean, std = model.mc_predict(ids, bidx, n_samples=MC_SAMPLES)
        mc_means.extend(mean.cpu().float().tolist())
        mc_stds.extend(std.cpu().float().tolist())

        targets.extend(batch["log2_mic"].tolist())
        bact_idxs.extend(bidx.cpu().tolist())
        model.eval()  # restore eval mode

    return (np.array(std_preds), np.array(mc_means),
            np.array(mc_stds), np.array(targets), np.array(bact_idxs))


def print_comparison(name, std_preds, mc_preds, targets):
    r_std, _ = stats.pearsonr(targets, std_preds)
    r_mc,  _ = stats.pearsonr(targets, mc_preds)
    rmse_std = np.sqrt(np.mean((std_preds - targets)**2))
    rmse_mc  = np.sqrt(np.mean((mc_preds  - targets)**2))
    print(f"\n{name}")
    print(f"  Standard  : Pearson={r_std:.4f}  RMSE={rmse_std:.4f}")
    print(f"  MC-Dropout: Pearson={r_mc:.4f}  RMSE={rmse_mc:.4f}  "
          f"(Δ RMSE={rmse_mc - rmse_std:+.4f})")
    return r_std, r_mc, rmse_std, rmse_mc


def plot_calibration(name, mc_stds, errors, save_path):
    """
    Calibration: sort by predicted uncertainty (std), bin into deciles.
    Plot mean |error| per bin vs mean std per bin.
    A well-calibrated model should show roughly mean|error| ≈ std.
    """
    n_bins = 10
    order = np.argsort(mc_stds)
    bins_std = np.array_split(mc_stds[order], n_bins)
    bins_err = np.array_split(np.abs(errors[order]), n_bins)

    mean_stds = [b.mean() for b in bins_std]
    mean_errs = [b.mean() for b in bins_err]

    corr, p = stats.pearsonr(mc_stds, np.abs(errors))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Calibration curve
    ax = axes[0]
    ax.plot(mean_stds, mean_errs, "o-", color="#2196F3", label="Empirical")
    lo, hi = min(min(mean_stds), min(mean_errs)), max(max(mean_stds), max(mean_errs))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("MC-Dropout std (predicted uncertainty)")
    ax.set_ylabel("Mean |error| (actual error)")
    ax.set_title(f"{name}\nCalibration  r(std, |err|)={corr:.3f}  p={p:.3e}")
    ax.legend()

    # Uncertainty histogram
    ax = axes[1]
    ax.hist(mc_stds, bins=30, color="#FF5722", alpha=0.7, edgecolor="white")
    ax.axvline(mc_stds.mean(), color="k", linestyle="--",
               label=f"Mean std={mc_stds.mean():.3f}")
    ax.set_xlabel("MC-Dropout std")
    ax.set_ylabel("Count")
    ax.set_title(f"{name}\nUncertainty distribution")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_error_vs_uncertainty(all_results):
    """One scatter plot per model: |error| vs predicted std."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        mc_stds  = res["mc_stds"]
        errors   = np.abs(res["mc_preds"] - res["targets"])
        corr, _  = stats.pearsonr(mc_stds, errors)

        ax.scatter(mc_stds, errors, alpha=0.2, s=8, color="#9C27B0")
        ax.set_xlabel("MC-Dropout std")
        ax.set_ylabel("|Prediction error|")
        ax.set_title(f"{name}\nr={corr:.3f}")

    plt.tight_layout()
    out = OUT_DIR / "error_vs_uncertainty.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--grampa", type=str, default="data/grampa.csv")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  MC samples: {MC_SAMPLES}")

    from src.data.supervised_dataset import load_grampa, GRAMPA_TOP20

    _, _, test_ds = load_grampa(
        PROJECT_ROOT / args.grampa, max_len=48,
        val_ratio=0.1, test_ratio=0.1, label_noise_std=0.0,
    )
    print(f"Test set: {len(test_ds)} samples\n")

    print(f"{'Model':<20} {'Std Pearson':>12} {'MC Pearson':>12} "
          f"{'Std RMSE':>10} {'MC RMSE':>10} {'Δ RMSE':>8}")
    print("-" * 76)

    all_results = {}
    for name, spec in MODELS.items():
        if not spec["ckpt"].exists():
            print(f"[SKIP] {name}")
            continue

        model, _ = load_jepa_model(spec["ckpt"], spec["cfg"], device)
        std_preds, mc_preds, mc_stds, targets, bact_idxs = run_inference(
            model, test_ds, device)

        r_std, r_mc, rmse_std, rmse_mc = print_comparison(
            name, std_preds, mc_preds, targets)

        delta_rmse = rmse_mc - rmse_std
        print(f"  {'↑ MC better' if delta_rmse < 0 else '↓ MC worse'}")

        # calibration plot
        errors = mc_preds - targets
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_calibration(name, mc_stds, errors,
                         OUT_DIR / f"calibration_{safe_name}.png")

        all_results[name] = {
            "std_preds": std_preds, "mc_preds": mc_preds,
            "mc_stds": mc_stds, "targets": targets,
        }

    plot_error_vs_uncertainty(all_results)

    # Summary table
    print(f"\n{'='*76}")
    print(f"{'Model':<20} {'Std Pearson':>12} {'MC Pearson':>12} "
          f"{'Std RMSE':>10} {'MC RMSE':>10} {'Δ RMSE':>8}")
    print("-" * 76)
    for name, res in all_results.items():
        r_std, _ = stats.pearsonr(res["targets"], res["std_preds"])
        r_mc,  _ = stats.pearsonr(res["targets"], res["mc_preds"])
        rmse_std = np.sqrt(np.mean((res["std_preds"] - res["targets"])**2))
        rmse_mc  = np.sqrt(np.mean((res["mc_preds"]  - res["targets"])**2))
        print(f"{name:<20} {r_std:>12.4f} {r_mc:>12.4f} "
              f"{rmse_std:>10.4f} {rmse_mc:>10.4f} {rmse_mc-rmse_std:>+8.4f}")
    print(f"{'='*76}")
    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
