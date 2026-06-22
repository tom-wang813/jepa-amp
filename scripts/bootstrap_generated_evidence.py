"""Bootstrap confidence intervals for generated-peptide evidence."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def ci(values: np.ndarray, level: float) -> dict[str, float]:
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    return {
        "mean_bootstrap": float(values.mean()),
        "ci_low": float(np.quantile(values, lo)),
        "ci_high": float(np.quantile(values, hi)),
    }


def stat(values: np.ndarray, spec: dict[str, Any]) -> float:
    kind = spec["statistic"]
    if kind == "mean":
        return float(values.mean())
    if kind == "median":
        return float(np.median(values))
    if kind == "p95":
        return float(np.quantile(values, 0.95))
    if kind == "fraction_ge":
        return float(np.mean(values >= float(spec["threshold"])))
    if kind == "fraction_gt":
        return float(np.mean(values > float(spec["threshold"])))
    if kind == "fraction_abs_gt":
        return float(np.mean(np.abs(values) > float(spec["threshold"])))
    raise ValueError(f"Unsupported statistic: {kind}")


def point(values: np.ndarray, spec: dict[str, Any]) -> float:
    return stat(values, spec)


def bootstrap_single(values: np.ndarray, spec: dict[str, Any], rng: np.random.Generator, n: int, level: float) -> dict[str, float]:
    boots = np.empty(n, dtype=float)
    size = len(values)
    for i in range(n):
        sample = values[rng.integers(0, size, size=size)]
        boots[i] = stat(sample, spec)
    return {"point": point(values, spec), **ci(boots, level)}


def bootstrap_delta(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    rng: np.random.Generator,
    n: int,
    level: float,
) -> dict[str, float]:
    scenario = [r for r in rows if r["scenario_key"] == spec["scenario"]]
    if not scenario:
        raise ValueError(f"No rows for scenario {spec['scenario']}")

    if spec.get("control") is not None:
        control = [r for r in rows if r["scenario_key"] == spec["control"]]
        if not control:
            raise ValueError(f"No rows for control {spec['control']}")
        x = np.asarray([float(r[spec["field"]]) for r in scenario], dtype=float)
        y = np.asarray([float(r[spec["field"]]) for r in control], dtype=float)
        point_delta = float(x.mean() - y.mean())
        boots = np.empty(n, dtype=float)
        for i in range(n):
            xb = x[rng.integers(0, len(x), size=len(x))]
            yb = y[rng.integers(0, len(y), size=len(y))]
            boots[i] = xb.mean() - yb.mean()
        return {"point": point_delta, **ci(boots, level)}

    x = np.asarray([float(r[spec["field_a"]]) - float(r[spec["field_b"]]) for r in scenario], dtype=float)
    point_delta = float(x.mean())
    boots = np.empty(n, dtype=float)
    for i in range(n):
        xb = x[rng.integers(0, len(x), size=len(x))]
        boots[i] = xb.mean()
    return {"point": point_delta, **ci(boots, level)}


def write_summary(out_dir: Path, metrics: dict[str, Any]) -> None:
    lines = ["# Generated Evidence Bootstrap Summary", ""]
    lines.append("## MIC Deltas")
    lines.append("")
    lines.append("| Metric | Point | 95% CI |")
    lines.append("|---|---:|---:|")
    for key, vals in metrics["mic_deltas"].items():
        lines.append(f"| {key} | {vals['point']:.3f} | [{vals['ci_low']:.3f}, {vals['ci_high']:.3f}] |")
    lines.append("")
    lines.append("## Plausibility")
    lines.append("")
    lines.append("| Metric | Point | 95% CI |")
    lines.append("|---|---:|---:|")
    for key, vals in metrics["plausibility"].items():
        lines.append(f"| {key} | {vals['point']:.3f} | [{vals['ci_low']:.3f}, {vals['ci_high']:.3f}] |")
    lines.append("")
    lines.append("Interpretation: intervals quantify sampling uncertainty over generated sequences only; they do not account for model-training uncertainty or wet-lab variability.")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(resolve(args.config)) as f:
        cfg = yaml.safe_load(f)

    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve(args.config), out_dir / "config_resolved.yaml")

    rng = np.random.default_rng(int(cfg.get("seed", 123)))
    n = int(cfg["bootstrap"].get("n_resamples", 5000))
    level = float(cfg["bootstrap"].get("ci", 0.95))

    mic_rows = load_jsonl(resolve(cfg["inputs"]["mic_generation_predictions"]))
    plaus_rows = load_jsonl(resolve(cfg["inputs"]["plausibility_predictions"]))

    mic_out = {
        spec["key"]: bootstrap_delta(mic_rows, spec, rng, n, level)
        for spec in cfg["mic_deltas"]
    }
    plaus_out = {}
    for spec in cfg["plausibility"]["metrics"]:
        values = np.asarray([float(r[spec["field"]]) for r in plaus_rows], dtype=float)
        plaus_out[spec["key"]] = bootstrap_single(values, spec, rng, n, level)

    metrics = {
        "experiment_id": cfg["experiment_id"],
        "research_decision": cfg["research_decision"],
        "n_resamples": n,
        "ci": level,
        "n_mic_rows": len(mic_rows),
        "n_plausibility_rows": len(plaus_rows),
        "mic_deltas": mic_out,
        "plausibility": plaus_out,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    write_summary(out_dir, metrics)
    manifest = {
        "experiment_id": cfg["experiment_id"],
        "config": str(out_dir / "config_resolved.yaml"),
        "metrics": str(out_dir / "metrics.json"),
        "summary": str(out_dir / "SUMMARY.md"),
        "status": "formal_artifact",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved bootstrap evidence artifacts to {out_dir}")


if __name__ == "__main__":
    main()
