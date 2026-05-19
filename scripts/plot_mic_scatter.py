"""
Per-bacteria scatter plots: predicted vs actual log2(MIC).
Runs both MLP and Transformer heads on the GRAMPA test set.

Usage:
  uv run python scripts/plot_mic_scatter.py [--gpu 1]
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
OUT_DIR = PROJECT_ROOT / "eval_results" / "mic_scatter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "MLP (FiLM)": {
        "ckpt": PROJECT_ROOT / "checkpoints/mic_868k_mlp/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/mic_868k.yaml",
    },
    "Transformer": {
        "ckpt": PROJECT_ROOT / "checkpoints/mic_868k_transformer/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/mic_868k_transformer.yaml",
    },
}


def load_model(ckpt_path: Path, cfg_path: Path, device: torch.device):
    from src.models.jepa import JEPA
    from src.models.supervised_head import JEPAMICPredictor
    from src.data.supervised_dataset import N_BACTERIA

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    pretrain_ckpt = torch.load(
        PROJECT_ROOT / cfg["pretrain_checkpoint"], map_location=device, weights_only=False
    )
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
    model.eval()
    return model, pretrain_ckpt["cfg"]["model"].get("max_seq_len", 52)


@torch.no_grad()
def predict(model, test_ds, device, batch_size=256):
    from src.data.supervised_dataset import collate_supervised
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_supervised)
    preds, targets, bact_idxs = [], [], []
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        bidx  = batch["bacteria_idx"].to(device)
        p = model(ids, bidx).cpu().float().numpy()
        preds.extend(p.tolist())
        targets.extend(batch["log2_mic"].tolist())
        bact_idxs.extend(bidx.cpu().tolist())
    return np.array(preds), np.array(targets), np.array(bact_idxs)


def plot_per_bacteria(all_results: dict, bacteria_names: list[str]):
    """Grid of per-bacteria scatter plots for each model."""
    # Find bacteria with enough test samples
    any_model = next(iter(all_results.values()))
    _, targets, bact_idxs = any_model
    min_samples = 10
    active_bact = [i for i, name in enumerate(bacteria_names)
                   if np.sum(bact_idxs == i) >= min_samples]

    n_bact = len(active_bact)
    n_models = len(all_results)
    cols = 4
    rows = int(np.ceil(n_bact / cols))

    for model_name, (preds, targets_arr, bact_arr) in all_results.items():
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten()
        fig.suptitle(f"MIC Prediction — {model_name}\n(log₂ MIC, test set)",
                     fontsize=14, fontweight="bold")

        for ax_idx, bact_i in enumerate(active_bact):
            ax = axes[ax_idx]
            mask = bact_arr == bact_i
            y_true = targets_arr[mask]
            y_pred = preds[mask]
            n = mask.sum()

            r, p_val = stats.pearsonr(y_true, y_pred)
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

            ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#2196F3")
            lo = min(y_true.min(), y_pred.min()) - 0.5
            hi = max(y_true.max(), y_pred.max()) + 0.5
            ax.plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.7)
            ax.set_xlabel("Actual log₂(MIC)", fontsize=8)
            ax.set_ylabel("Predicted log₂(MIC)", fontsize=8)
            ax.set_title(f"{bacteria_names[bact_i]}\nr={r:.3f}  RMSE={rmse:.2f}  n={n}",
                         fontsize=8)
            ax.tick_params(labelsize=7)

        for ax in axes[len(active_bact):]:
            ax.set_visible(False)

        plt.tight_layout()
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        out = OUT_DIR / f"mic_scatter_{safe_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


def plot_combined_overview(all_results: dict, bacteria_names: list[str]):
    """One plot per model: all bacteria overlaid with color coding."""
    cmap = plt.cm.get_cmap("tab20", len(bacteria_names))

    for model_name, (preds, targets_arr, bact_arr) in all_results.items():
        fig, ax = plt.subplots(figsize=(8, 7))
        for i, bname in enumerate(bacteria_names):
            mask = bact_arr == i
            if mask.sum() < 5:
                continue
            ax.scatter(targets_arr[mask], preds[mask], alpha=0.5, s=12,
                       color=cmap(i), label=bname)

        lo = min(targets_arr.min(), preds.min()) - 0.5
        hi = max(targets_arr.max(), preds.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, alpha=0.7, label="y=x")

        r, _ = stats.pearsonr(targets_arr, preds)
        rmse = np.sqrt(np.mean((preds - targets_arr) ** 2))
        ax.set_title(f"{model_name}\nOverall: r={r:.3f}  RMSE={rmse:.2f}", fontsize=13)
        ax.set_xlabel("Actual log₂(MIC)")
        ax.set_ylabel("Predicted log₂(MIC)")
        ax.legend(fontsize=6, ncol=2, loc="upper left")
        plt.tight_layout()
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        out = OUT_DIR / f"mic_overview_{safe_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


def print_per_bacteria_table(all_results: dict, bacteria_names: list[str]):
    print("\n" + "=" * 80)
    print("PER-BACTERIA MIC PREDICTION (Pearson r, test set)")
    print("=" * 80)
    model_names = list(all_results.keys())
    header = f"{'Bacterium':<28}" + "".join(f"{'r '+m:>18}" for m in model_names) + f"{'n':>6}"
    print(header)
    print("-" * 80)

    _, _, bact_arr0 = next(iter(all_results.values()))
    rows = []
    for i, bname in enumerate(bacteria_names):
        mask = bact_arr0 == i
        n = mask.sum()
        if n < 10:
            continue
        rs = []
        for model_name, (preds, targets_arr, bact_arr) in all_results.items():
            m = bact_arr == i
            if m.sum() < 5:
                rs.append(float("nan"))
            else:
                r, _ = stats.pearsonr(targets_arr[m], preds[m])
                rs.append(r)
        rows.append((bname, rs, n))

    for bname, rs, n in sorted(rows, key=lambda x: -x[1][0] if not np.isnan(x[1][0]) else -999):
        row = f"{bname:<28}" + "".join(f"{r:>18.3f}" if not np.isnan(r) else f"{'N/A':>18}" for r in rs)
        print(row + f"{n:>6}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--grampa", type=str, default="data/grampa.csv")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from src.data.supervised_dataset import load_grampa, GRAMPA_TOP20
    print("Loading GRAMPA test set …")
    _, _, test_ds = load_grampa(
        PROJECT_ROOT / args.grampa, max_len=48,
        val_ratio=0.1, test_ratio=0.1, label_noise_std=0.0,
    )
    print(f"  Test set: {len(test_ds)} samples")

    all_results = {}
    for model_name, spec in MODELS.items():
        if not spec["ckpt"].exists():
            print(f"[SKIP] {model_name}")
            continue
        print(f"Running {model_name} …")
        model, _ = load_model(spec["ckpt"], spec["cfg"], device)
        preds, targets, bact_idxs = predict(model, test_ds, device)
        r, _ = stats.pearsonr(targets, preds)
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        print(f"  Overall: Pearson={r:.4f}  RMSE={rmse:.4f}  n={len(preds)}")
        all_results[model_name] = (preds, targets, bact_idxs)

    print("\nGenerating plots …")
    plot_per_bacteria(all_results, GRAMPA_TOP20)
    plot_combined_overview(all_results, GRAMPA_TOP20)
    print_per_bacteria_table(all_results, GRAMPA_TOP20)
    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
