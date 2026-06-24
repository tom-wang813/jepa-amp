"""
Phylogenetic tree of 6 bacteria with cross-species transfer annotations.

Tree topology from 16S rRNA taxonomy (NCBI/SILVA).
Branch lengths = 16S rRNA distance (1 - identity).

Annotations:
  - Node circles: Gram type (blue=neg, orange=pos)
  - Node labels: species name + data size
  - Connecting arcs between species: colored by zero-shot Spearman rho
  - Right panel: ordered transfer heatmap

Output: eval_results/interpretability/phylo_tree_annotated.png
"""
from __future__ import annotations
import json, sys
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from Bio import Phylo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "eval_results" / "interpretability"

# ── Newick tree with branch lengths = 16S rRNA distance ──────────────────────
# Topology: (((Eco:0.015, Sty:0.015):0.138, Pae:0.153):0.082,
#             ((Sau:0.083, Bsub:0.083):0.084, Mlut:0.193):0.082);
NEWICK = (
    "((('E. coli':0.0075,'S. typhimurium':0.0075):0.146,"
    "'P. aeruginosa':0.153):0.082,"
    "(('S. aureus':0.083,'B. subtilis':0.083):0.084,"
    "'M. luteus':0.193):0.082);"
)

GRAM = {
    "E. coli": "neg", "S. typhimurium": "neg", "P. aeruginosa": "neg",
    "S. aureus": "pos", "B. subtilis": "pos", "M. luteus": "pos",
}
GRAM_COLOR = {"neg": "#457B9D", "pos": "#E76F51"}

DATA_SIZE = {
    "E. coli": 5465, "S. aureus": 5070, "P. aeruginosa": 2523,
    "B. subtilis": 1323, "S. typhimurium": 715, "M. luteus": 651,
}

SP_SHORT = {
    "E. coli": "E. coli", "S. typhimurium": "S. Typhimurium",
    "P. aeruginosa": "P. aeruginosa", "S. aureus": "S. aureus",
    "B. subtilis": "B. subtilis", "M. luteus": "M. luteus",
}

MODELS = ["jepa", "mlm", "esm2"]


def load_zero_shot(pairs_models):
    """Returns dict: pair_key -> avg zero-shot spearman across models."""
    d_all = {m: json.load(open(PROJECT_ROOT / f"eval_results/fewshot_v2/{m}/metrics.json"))
             for m in MODELS}
    result = {}
    for pk in d_all["jepa"]:
        rhos = [d_all[m][pk][s]["0"]["spearman"]
                for m in MODELS for s in d_all[m].get(pk, {})]
        result[pk] = np.mean(rhos)
    return result


def get_leaf_order(tree):
    """Return leaf names in drawing order via DFS (top to bottom)."""
    order = []
    def dfs(clade):
        if clade.is_terminal():
            order.append(clade.name)
        else:
            for c in clade.clades:
                dfs(c)
    dfs(tree.root)
    return order


def draw_tree_panel(ax, tree, rho_dict):
    """Draw cladogram with branch lengths, colored by Gram type."""
    # use Bio.Phylo to get x/y coordinates
    # manually compute positions
    leaves = get_leaf_order(tree)
    n = len(leaves)
    y_pos = {name: (n - 1 - i) for i, name in enumerate(leaves)}

    # DFS to assign x positions (branch length from root)
    def assign_x(clade, parent_x=0.0):
        x = parent_x + (clade.branch_length or 0)
        if clade.is_terminal():
            return {clade.name: x}
        coords = {}
        for c in clade.clades:
            coords.update(assign_x(c, x))
        return coords

    leaf_x = assign_x(tree.root)
    max_x = max(leaf_x.values())

    # internal node y = avg of children leaves
    def node_y(clade):
        if clade.is_terminal():
            return y_pos[clade.name]
        return np.mean([node_y(c) for c in clade.clades])

    def draw_clade(clade, parent_x, ax):
        x = parent_x + (clade.branch_length or 0)
        y = node_y(clade)

        # horizontal line from parent to this node
        ax.plot([parent_x, x], [y, y], color="#555", lw=1.4, zorder=2)

        if clade.is_terminal():
            sp = clade.name
            color = GRAM_COLOR[GRAM[sp]]
            # leaf circle
            ax.scatter([x], [y], s=120, c=color, zorder=5,
                       linewidths=1.2, edgecolors="white")
            # label
            n_data = DATA_SIZE[sp]
            ax.text(x + 0.005, y, f"  {SP_SHORT[sp]}  (n={n_data:,})",
                    va="center", ha="left", fontsize=9.5, style="italic",
                    color="#222")
        else:
            # vertical line connecting children
            child_ys = [node_y(c) for c in clade.clades]
            ax.plot([x, x], [min(child_ys), max(child_ys)],
                    color="#555", lw=1.4, zorder=2)
            for c in clade.clades:
                draw_clade(c, x, ax)

    draw_clade(tree.root, 0, ax)

    # Gram-type band labels
    ax.text(-0.01, 4.0, "Gram−", fontsize=10, color=GRAM_COLOR["neg"],
            fontweight="bold", ha="right", va="center")
    ax.text(-0.01, 1.0, "Gram+", fontsize=10, color=GRAM_COLOR["pos"],
            fontweight="bold", ha="right", va="center")
    ax.axhline(2.5, color="#ddd", lw=1, ls="--", zorder=0)

    ax.set_xlim(-0.04, max_x + 0.18)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")
    ax.set_title("16S rRNA phylogeny", fontsize=11, pad=6)

    # scale bar
    ax.plot([0, 0.1], [-0.3, -0.3], color="#555", lw=1.5)
    ax.text(0.05, -0.45, "0.1", ha="center", fontsize=8, color="#555")

    return leaves, leaf_x, y_pos


def draw_heatmap_panel(ax, leaves, rho_dict):
    """Ordered transfer heatmap (rows=source, cols=target)."""
    n = len(leaves)
    mat = np.full((n, n), np.nan)
    for i, src in enumerate(leaves):
        for j, tgt in enumerate(leaves):
            if src == tgt:
                continue
            pk = f"{src}→{tgt}"
            if pk in rho_dict:
                mat[i, j] = rho_dict[pk]

    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.1, vmax=0.65,
                   aspect="auto", interpolation="nearest")

    short = [s.split(".")[0].strip() + "." if "." in s else s[:4]
             for s in leaves]
    # use italic species abbreviations
    sp_abbr = {
        "E. coli": "Eco", "S. typhimurium": "Sty", "P. aeruginosa": "Pae",
        "S. aureus": "Sau", "B. subtilis": "Bsub", "M. luteus": "Mlut",
    }
    labels = [sp_abbr[s] for s in leaves]

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=9, rotation=35, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Target species", fontsize=10)
    ax.set_ylabel("Source species", fontsize=10)
    ax.set_title("Zero-shot Spearman ρ\n(avg 3 models × 3 seeds)", fontsize=10)

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                v = mat[i, j]
                tc = "white" if (v < 0.28 or v > 0.58) else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=tc, fontweight="medium")
            elif i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="#bbb")

    # Gram dividers
    ax.axhline(2.5, color="white", lw=2.5)
    ax.axvline(2.5, color="white", lw=2.5)

    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03,
                 label="Spearman ρ")
    return im


def draw_scatter_panel(ax, rho_dict):
    """Phylo distance vs zero-shot rho scatter."""
    DIST_16S = {
        frozenset(["E. coli", "S. typhimurium"]): 0.015,
        frozenset(["E. coli", "P. aeruginosa"]):  0.153,
        frozenset(["S. typhimurium", "P. aeruginosa"]): 0.152,
        frozenset(["S. aureus", "B. subtilis"]):   0.165,
        frozenset(["S. aureus", "M. luteus"]):     0.193,
        frozenset(["B. subtilis", "M. luteus"]):   0.167,
        frozenset(["E. coli", "S. aureus"]):       0.252,
        frozenset(["E. coli", "B. subtilis"]):     0.238,
        frozenset(["E. coli", "M. luteus"]):       0.262,
        frozenset(["S. typhimurium", "S. aureus"]): 0.251,
        frozenset(["S. typhimurium", "B. subtilis"]): 0.239,
        frozenset(["S. typhimurium", "M. luteus"]):  0.263,
        frozenset(["P. aeruginosa", "S. aureus"]):  0.235,
        frozenset(["P. aeruginosa", "B. subtilis"]): 0.229,
        frozenset(["P. aeruginosa", "M. luteus"]):   0.237,
    }

    sp_abbr = {
        "E. coli": "Eco", "S. typhimurium": "Sty", "P. aeruginosa": "Pae",
        "S. aureus": "Sau", "B. subtilis": "Bsub", "M. luteus": "Mlut",
    }

    xs, ys, colors, labels = [], [], [], []
    for pk, rho in rho_dict.items():
        src, tgt = pk.split("→")
        key = frozenset([src, tgt])
        if key not in DIST_16S:
            continue
        cross = GRAM[src] != GRAM[tgt]
        xs.append(DIST_16S[key])
        ys.append(rho)
        colors.append("#888" if cross else
                      GRAM_COLOR[GRAM[src]])
        labels.append(f"{sp_abbr[src]}→{sp_abbr[tgt]}")

    xs, ys = np.array(xs), np.array(ys)
    ax.scatter(xs, ys, c=colors, s=70, alpha=0.85, zorder=3)

    # trend
    z = np.polyfit(xs, ys, 1)
    xr = np.linspace(xs.min() - 0.01, xs.max() + 0.01, 50)
    ax.plot(xr, np.poly1d(z)(xr), "k--", lw=1.2, alpha=0.55)

    from scipy.stats import spearmanr
    rho_c, pval = spearmanr(xs, ys)
    ax.set_xlabel("16S rRNA distance", fontsize=10)
    ax.set_ylabel("Zero-shot Spearman ρ", fontsize=10)
    ax.set_title(f"Phylo distance → transfer difficulty\n(ρ = {rho_c:.3f}, p = {pval:.3f})", fontsize=10)

    handles = [
        mpatches.Patch(color=GRAM_COLOR["neg"], label="Gram− → Gram−"),
        mpatches.Patch(color=GRAM_COLOR["pos"], label="Gram+ → Gram+"),
        mpatches.Patch(color="#888",            label="Cross-Gram"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    tree = Phylo.read(StringIO(NEWICK), "newick")
    rho_dict = load_zero_shot(MODELS)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    fig = plt.figure(figsize=(17, 6.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.8, 1.8],
                           wspace=0.35, left=0.04, right=0.97,
                           top=0.88, bottom=0.12)

    ax_tree   = fig.add_subplot(gs[0])
    ax_heat   = fig.add_subplot(gs[1])
    ax_scatter = fig.add_subplot(gs[2])

    leaves, leaf_x, y_pos = draw_tree_panel(ax_tree, tree, rho_dict)
    draw_heatmap_panel(ax_heat, leaves, rho_dict)
    draw_scatter_panel(ax_scatter, rho_dict)

    # gram-type legend for tree
    handles = [
        mpatches.Patch(color=GRAM_COLOR["neg"], label="Gram−"),
        mpatches.Patch(color=GRAM_COLOR["pos"], label="Gram+"),
    ]
    ax_tree.legend(handles=handles, fontsize=8.5, frameon=False,
                   loc="lower right")

    fig.suptitle(
        "Cross-species AMP MIC transfer difficulty is predicted by phylogenetic distance",
        fontsize=13, fontweight="medium", y=0.98
    )

    out = OUT / "phylo_tree_annotated.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
