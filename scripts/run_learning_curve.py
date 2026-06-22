"""
Data-efficiency / learning curve analysis.

Trains JEPA-AMP and ESM-2 MIC predictors (frozen encoder) at four training-set
fractions (10 / 25 / 50 / 100 %) across three seeds.  All runs share the same
val / test split so comparisons are valid.

Usage:
    uv run python scripts/run_learning_curve.py [--gpu 0] [--dry-run]

Outputs (eval_results/learning_curve/):
    metrics.json      – {model: {fraction: {seed: {pearson, rmse}}}}
    SUMMARY.md        – table ready to paste into the paper
    learning_curve.png
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS     = [42, 123, 7]
OUT_DIR   = PROJECT_ROOT / "eval_results" / "learning_curve"

# ── base configs ──────────────────────────────────────────────────────────────

JEPA_BASE = {
    "task": "mic",
    "pretrain_checkpoint": "checkpoints/jepa_pretrain_868k/last_jepa.pt",
    "data": {
        "grampa_csv": "data/grampa.csv",
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "label_noise_std": 0.0,   # no noise for controlled comparison
        "train_fraction": 1.0,
        "seed": 42,               # split seed fixed; train_fraction seed varies below
    },
    "head": {
        "head_type": "transformer",
        "bacteria_dim": 64,
        "hidden": 256,
        "dropout": 0.4,
        "adapter_bottleneck": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
    },
    "train": {
        "freeze_encoder": True,
        "epochs": 60,
        "batch_size": 256,
        "lr": 3.0e-4,
        "weight_decay": 0.1,
        "fp16": True,
        "num_workers": 0,
        "patience": 15,
        "save_every": 999,
    },
}

ESM2_BASE = {
    "task": "mic",
    "esm_model": "esm2_t12_35M",
    "data": {
        "grampa_csv": "data/grampa.csv",
        "max_len": 48,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "label_noise_std": 0.0,
        "train_fraction": 1.0,
        "seed": 42,
    },
    "head": {
        "head_type": "mlp",
        "bacteria_dim": 64,
        "hidden": 256,
        "dropout": 0.3,
    },
    "train": {
        "freeze_encoder": True,
        "epochs": 60,
        "batch_size": 64,
        "lr": 3.0e-4,
        "weight_decay": 0.05,
        "fp16": True,
        "num_workers": 0,
        "patience": 15,
        "save_every": 999,
    },
}


def _make_config(base: dict, fraction: float, seed: int, save_dir: Path) -> dict:
    cfg = copy.deepcopy(base)
    cfg["data"]["train_fraction"] = fraction
    cfg["data"]["seed"] = seed          # controls both split shuffle and fraction subsample
    cfg["train"]["save_dir"] = str(save_dir)
    return cfg


def _run_config(cfg: dict, entrypoint: str, gpu: int) -> dict | None:
    """Write config to a temp file, run training, return test_metrics.json content."""
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir=PROJECT_ROOT / "configs",
        prefix="_lc_tmp_", delete=False
    ) as f:
        yaml.dump(cfg, f)
        tmp_path = Path(f.name)

    try:
        cmd = [
            "uv", "run", "python", "-m", entrypoint,
            "--config", str(tmp_path.relative_to(PROJECT_ROOT)),
            "--gpu", str(gpu),
        ]
        print(f"  $ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)
        if result.returncode != 0:
            print(f"  [WARN] run exited with code {result.returncode}")
            return None
    finally:
        tmp_path.unlink(missing_ok=True)

    metrics_path = save_dir / "test_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    print(f"  [WARN] test_metrics.json not found in {save_dir}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running training")
    parser.add_argument("--model", choices=["jepa", "esm2", "both"], default="both")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"jepa": {}, "esm2": {}}

    # Resume from existing metrics.json if present
    existing_path = OUT_DIR / "metrics.json"
    if existing_path.exists():
        results = json.loads(existing_path.read_text())
        print(f"Resumed from {existing_path}")

    runs = []
    if args.model in ("jepa", "both"):
        runs += [("jepa", JEPA_BASE, "src.train.finetune_supervised")]
    if args.model in ("esm2", "both"):
        runs += [("esm2", ESM2_BASE, "src.train.train_esm_supervised")]

    for model_name, base_cfg, entrypoint in runs:
        model_results = results.setdefault(model_name, {})
        for frac in FRACTIONS:
            frac_key = str(frac)
            frac_results = model_results.setdefault(frac_key, {})
            for seed in SEEDS:
                seed_key = str(seed)
                if seed_key in frac_results:
                    print(f"[skip] {model_name} frac={frac} seed={seed} (already done)")
                    continue

                save_dir = OUT_DIR / f"{model_name}_frac{int(frac*100):03d}_seed{seed}"
                cfg = _make_config(base_cfg, frac, seed, save_dir)

                print(f"\n── {model_name.upper()} frac={frac:.0%} seed={seed} ──")
                if args.dry_run:
                    print(yaml.dump(cfg))
                    continue

                m = _run_config(cfg, entrypoint, args.gpu)
                if m is not None:
                    frac_results[seed_key] = {
                        "pearson": m.get("pearson", m.get("test_pearson")),
                        "rmse":    m.get("rmse",    m.get("test_rmse")),
                    }
                    existing_path.write_text(json.dumps(results, indent=2))
                    print(f"  pearson={frac_results[seed_key]['pearson']:.4f}  "
                          f"rmse={frac_results[seed_key]['rmse']:.4f}")

    if args.dry_run:
        return

    _write_summary(results)
    _plot(results)
    print(f"\nDone. Results in {OUT_DIR}")


def _write_summary(results: dict) -> None:
    import numpy as np

    lines = [
        "# Data Efficiency: JEPA-AMP vs ESM-2 (Frozen Encoder)",
        "",
        "MIC Pearson correlation on fixed GRAMPA test set (mean ± std over 3 seeds).",
        "",
        "| Train fraction | JEPA-AMP Pearson | ESM-2 Pearson | JEPA-AMP RMSE | ESM-2 RMSE |",
        "|---|---|---|---|---|",
    ]
    for frac in FRACTIONS:
        frac_key = str(frac)
        row = [f"{frac:.0%}"]
        for model_name in ("jepa", "esm2"):
            frac_data = results.get(model_name, {}).get(frac_key, {})
            pearsons = [v["pearson"] for v in frac_data.values()
                        if v and v.get("pearson") is not None]
            rmses    = [v["rmse"]    for v in frac_data.values()
                        if v and v.get("rmse")    is not None]
            if pearsons:
                row.append(f"{np.mean(pearsons):.3f} ± {np.std(pearsons):.3f}")
            else:
                row.append("—")
            if model_name == "esm2" and rmses:
                pass
            if rmses:
                row.append(f"{np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT_DIR / 'SUMMARY.md'}")


def _plot(results: dict) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"jepa": "#1f77b4", "esm2": "#ff7f0e"}
    labels = {"jepa": "JEPA-AMP (frozen)", "esm2": "ESM-2 (frozen)"}
    x = [f * 100 for f in FRACTIONS]

    for model_name in ("jepa", "esm2"):
        means, stds = [], []
        for frac in FRACTIONS:
            frac_data = results.get(model_name, {}).get(str(frac), {})
            vals = [v["pearson"] for v in frac_data.values()
                    if v and v.get("pearson") is not None]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            else:
                means.append(None)
                stds.append(0.0)

        valid = [(xi, m, s) for xi, m, s in zip(x, means, stds) if m is not None]
        if not valid:
            continue
        xi, m, s = zip(*valid)
        ax.plot(xi, m, "o-", color=colors[model_name], label=labels[model_name])
        ax.fill_between(xi,
                        [v - e for v, e in zip(m, s)],
                        [v + e for v, e in zip(m, s)],
                        alpha=0.15, color=colors[model_name])

    ax.set_xlabel("Training set size (%)")
    ax.set_ylabel("Pearson r (MIC, test set)")
    ax.set_title("Data Efficiency: JEPA-AMP vs ESM-2 (Frozen Encoder)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}%" for v in x])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
