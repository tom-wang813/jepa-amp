"""Plot paper-ready QMAP benchmark summary figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("paper/figures/qmap_benchmark_summary.png")


def main() -> None:
    labels = [
        "QMAP\nESM2 linear",
        "QMAP\nWitten 2019",
        "QMAP\nCai 2025",
        "JEPA\nhead",
        "JEPA\nconditional",
    ]
    full_ecoli = [0.36, 0.51, 0.52, 0.5011, 0.5122]
    high_eff = [0.16, 0.22, 0.29, 0.3725, 0.3877]

    hc50_labels = ["QMAP\nESM2 linear", "JEPA\nHC50 head", "JEPA\nconditional"]
    hc50 = [0.07, 0.3273, 0.3065]

    colors = {
        "baseline": "#7A8793",
        "head": "#1F77B4",
        "cond": "#2CA02C",
        "hc50": "#D62728",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), constrained_layout=True)

    x = np.arange(len(labels))
    bar_colors = [colors["baseline"], colors["baseline"], colors["baseline"], colors["head"], colors["cond"]]
    axes[0].bar(x, full_ecoli, color=bar_colors, width=0.72)
    axes[0].set_title("Full E. coli MIC")
    axes[0].set_ylabel("Mean Pearson r")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0, 0.6)
    axes[0].axhline(0.52, color="#444444", linewidth=0.8, linestyle="--")
    axes[0].text(2.65, 0.535, "best prior 0.52", fontsize=8, color="#444444")

    axes[1].bar(x, high_eff, color=bar_colors, width=0.72)
    axes[1].set_title("High-efficiency E. coli MIC")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylim(0, 0.45)
    axes[1].axhline(0.29, color="#444444", linewidth=0.8, linestyle="--")
    axes[1].text(2.45, 0.305, "best prior 0.29", fontsize=8, color="#444444")

    x_h = np.arange(len(hc50_labels))
    axes[2].bar(x_h, hc50, color=[colors["baseline"], colors["hc50"], colors["cond"]], width=0.72)
    axes[2].set_title("HC50 hemolysis")
    axes[2].set_xticks(x_h)
    axes[2].set_xticklabels(hc50_labels, fontsize=8)
    axes[2].set_ylim(0, 0.4)
    axes[2].axhline(0.07, color="#444444", linewidth=0.8, linestyle="--")
    axes[2].text(0.15, 0.085, "QMAP baseline 0.07", fontsize=8, color="#444444")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#E6E8EB", linewidth=0.8)
        ax.set_axisbelow(True)
        for patch in ax.patches:
            value = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("QMAP homology-aware benchmark performance", fontsize=12, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
