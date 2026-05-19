"""
End-to-end evaluation script for the JEPA-based AMP generator.

Usage:
    uv run python -m src.eval.run_eval

Workflow:
  1. Load pre-trained JEPA encoder + conditional generator checkpoints.
  2. Load prefix/suffix fine-tuning sequences as novelty reference set.
  3. Sample prefix batches from the training DataLoader and generate
     500 conditioned sequences (temperature=1.0, top_p=0.9).
  4. Train an AMPClassifier (positive = training AMP seqs,
     negative = auto-downloaded non-AMP UniProt peptides).
  5. Compute all metrics: validity, uniqueness, novelty, diversity,
     physicochemical stats, AMP scores, AA frequency.
  6. Print a formatted summary table.
  7. Save results to eval_results/eval_report.json.
  8. Save three plots to eval_results/:
       aa_freq_comparison.png
       amp_score_dist.png
       length_dist.png
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
from src.data.dataset import build_seq2seq_datasets
from src.data.tokenizer import decode, EOS_ID, AMINO_ACIDS
from src.models.encoder import TransformerEncoder
from src.models.jepa import JEPA
from src.models.generator import ConditionalGenerator
from src.eval.metrics import (
    validity,
    uniqueness,
    novelty,
    diversity,
    aa_frequency,
    physicochemical_stats,
)
from src.eval.amp_classifier import (
    AMPlifyClassifier, ESMAMPClassifier, MacrelClassifier, JEPAAMPClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (all relative to project root, resolved at runtime)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Defaults point to the 868k-trained models; override with CLI args.
PRETRAIN_CKPT = PROJECT_ROOT / "checkpoints" / "jepa_pretrain_868k" / "last_jepa.pt"
GENERATOR_CKPT = PROJECT_ROOT / "checkpoints" / "generator_868k" / "best_generator.pt"
PRETRAIN_CFG   = PROJECT_ROOT / "configs" / "jepa_pretrain_868k.yaml"
FINETUNE_CFG   = PROJECT_ROOT / "configs" / "finetune_868k.yaml"
EVAL_DIR       = PROJECT_ROOT / "eval_results"
MAX_CLASSIFIER_SEQS = 10_000  # cap neg samples for JEPA probe and representation eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ids_to_sequence(token_ids: list[int]) -> str:
    """Convert a list of token IDs (from generator output) to an AA string."""
    aa_set = set(AMINO_ACIDS)
    chars = []
    for tid in token_ids:
        if tid == EOS_ID:
            break
        aa = decode([tid], remove_special_tokens=True)
        for c in aa:
            if c in aa_set:
                chars.append(c)
    return "".join(chars)


def _print_table(rows: list[tuple[str, str]], title: str = "") -> None:
    """Print a simple two-column table."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    col_w = max(len(r[0]) for r in rows) + 2
    for name, val in rows:
        print(f"  {name:<{col_w}}{val}")
    print()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(device: torch.device) -> tuple[JEPA, ConditionalGenerator]:
    # Load generator checkpoint first — it embeds both pretrain and finetune configs.
    gen_ckpt = None
    if GENERATOR_CKPT.exists():
        gen_ckpt = torch.load(GENERATOR_CKPT, map_location=device, weights_only=False)
        pretrain_model_cfg = gen_ckpt["pretrain_cfg"]["model"]
        gen_cfg            = gen_ckpt["cfg"]["generator"]
    else:
        logger.warning("Generator checkpoint not found at %s — falling back to YAML.", GENERATOR_CKPT)
        pretrain_model_cfg = _load_yaml(PRETRAIN_CFG)["model"]
        gen_cfg            = _load_yaml(FINETUNE_CFG)["generator"]

    jepa = JEPA(**pretrain_model_cfg)

    if PRETRAIN_CKPT.exists():
        pt_ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
        jepa.load_state_dict(pt_ckpt["model_state"])
        logger.info("Loaded JEPA checkpoint (epoch %d, val_loss=%.4f)",
                    pt_ckpt.get("epoch", -1), pt_ckpt.get("val_loss", -1))
    else:
        logger.warning("Pre-train checkpoint not found at %s — using random weights.", PRETRAIN_CKPT)

    encoder = TransformerEncoder(
        d_model=pretrain_model_cfg["d_model"],
        nhead=pretrain_model_cfg["nhead"],
        num_layers=pretrain_model_cfg["num_layers"],
        dim_feedforward=pretrain_model_cfg["dim_feedforward"],
        dropout=pretrain_model_cfg["dropout"],
        max_seq_len=pretrain_model_cfg["max_seq_len"],
    )
    encoder.load_state_dict(jepa.context_encoder.state_dict())

    generator = ConditionalGenerator(
        encoder=encoder,
        d_model=pretrain_model_cfg["d_model"],
        freeze_encoder=True,
        **gen_cfg,
    )

    if gen_ckpt is not None:
        generator.load_state_dict(gen_ckpt["model_state"])
        logger.info("Loaded generator checkpoint (epoch %d, val_loss=%.4f)",
                    gen_ckpt.get("epoch", -1), gen_ckpt.get("val_loss", -1))

    jepa.to(device).eval()
    generator.to(device).eval()
    return jepa, generator


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def generate_sequences(
    generator: ConditionalGenerator,
    seq2seq_dataset,
    device: torch.device,
    n_generate: int = 500,
    batch_size: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> list[str]:
    """
    Draw prefix batches from seq2seq_dataset and call generator.generate()
    until we have n_generate decoded sequences with length >= 3.
    """
    loader = DataLoader(seq2seq_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_iter = iter(loader)

    generated_seqs: list[str] = []
    pbar = tqdm(total=n_generate, desc="Generating sequences")

    while len(generated_seqs) < n_generate:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        context_ids = batch["context_ids"].to(device)
        conditions  = batch["conditions"].to(device) if "conditions" in batch else None

        with torch.no_grad():
            out_ids = generator.generate(
                context_ids,
                conditions=conditions,
                max_new_tokens=30,
                temperature=temperature,
                top_p=top_p,
            )  # (B, T)

        before = len(generated_seqs)
        for row in out_ids:
            seq = _ids_to_sequence(row.tolist())
            if len(seq) >= 3:
                generated_seqs.append(seq)
                if len(generated_seqs) >= n_generate:
                    break
        added = len(generated_seqs) - before
        if added > 0:
            pbar.update(added)

    pbar.close()
    logger.info("Generated %d valid sequences.", len(generated_seqs))
    return generated_seqs[:n_generate]


def evaluate_representation_backbones(
    train_sequences: list[str],
    val_sequences: list[str],
    device: torch.device,
    jepa_encoder: TransformerEncoder,
    encoder_cfg: dict,
) -> dict[str, dict[str, float]]:
    """
    Compare JEPA encoder features against a random-init encoder on held-out AMP vs non-AMP classification.
    """
    from src.eval.amp_classifier import _fetch_non_amp_sequences

    n_train_neg = min(len(train_sequences), MAX_CLASSIFIER_SEQS)
    n_val_neg   = min(len(val_sequences),   MAX_CLASSIFIER_SEQS // 4)
    neg_train = _fetch_non_amp_sequences(max_seqs=n_train_neg)
    neg_val   = _fetch_non_amp_sequences(max_seqs=n_val_neg)

    if not neg_train or not neg_val:
        logger.warning("Representation eval skipped: could not fetch enough negative sequences.")
        return {}

    random_encoder = TransformerEncoder(
        d_model=encoder_cfg["d_model"],
        nhead=encoder_cfg["nhead"],
        num_layers=encoder_cfg["num_layers"],
        dim_feedforward=encoder_cfg["dim_feedforward"],
        dropout=encoder_cfg["dropout"],
        max_seq_len=encoder_cfg["max_seq_len"],
    )

    models = {
        "jepa_encoder": JEPAAMPClassifier(encoder=jepa_encoder, device=str(device)),
        "random_encoder": JEPAAMPClassifier(encoder=random_encoder, device=str(device)),
    }

    results: dict[str, dict[str, float]] = {}

    import random as _rng
    train_pos_sample = _rng.sample(train_sequences, n_train_neg) if len(train_sequences) > n_train_neg else train_sequences
    val_pos_sample   = _rng.sample(val_sequences,   n_val_neg)   if len(val_sequences)   > n_val_neg   else val_sequences

    y_true = np.array([1] * len(val_pos_sample) + [0] * len(neg_val), dtype=np.int32)
    eval_sequences = val_pos_sample + neg_val

    for name, clf in models.items():
        logger.info("Representation probe: training %s …", name)
        clf.fit(pos_seqs=train_pos_sample, neg_seqs=neg_train)
        scores = clf.predict_proba(eval_sequences)
        preds = (scores >= 0.5).astype(np.int32)
        results[name] = {
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "accuracy": float(accuracy_score(y_true, preds)),
            "mean_positive_score": float(np.mean(scores[: len(val_sequences)])),
            "mean_negative_score": float(np.mean(scores[len(val_sequences) :])),
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_aa_freq(gen_freq: dict, train_freq: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aas = AMINO_ACIDS
    x = np.arange(len(aas))
    gen_vals   = [gen_freq.get(aa, 0.0) for aa in aas]
    train_vals = [train_freq.get(aa, 0.0) for aa in aas]
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, train_vals, width, label="Training set", alpha=0.8)
    ax.bar(x + width / 2, gen_vals,   width, label="Generated",    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(aas)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Frequency")
    ax.set_title("AA Frequency: Generated vs Training set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved AA frequency plot to %s", out_path)


def plot_amp_score(scores: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=40, range=(0, 1), edgecolor="white", linewidth=0.5)
    ax.axvline(float(np.mean(scores)), color="red", linestyle="--",
               label=f"Mean = {np.mean(scores):.3f}")
    ax.set_xlabel("P(AMP)")
    ax.set_ylabel("Count")
    ax.set_title("AMP Score Distribution (Generated Sequences)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved AMP score distribution plot to %s", out_path)


def plot_length_dist(gen_seqs: list[str], train_seqs: list[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen_lens   = [len(s) for s in gen_seqs]
    train_lens = [len(s) for s in train_seqs]
    max_len = max(max(gen_lens, default=50), max(train_lens, default=50))
    bins = range(0, max_len + 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train_lens, bins=bins, alpha=0.6, label="Training set", density=True)
    ax.hist(gen_lens,   bins=bins, alpha=0.6, label="Generated",    density=True)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Density")
    ax.set_title("Length Distribution: Generated vs Training set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved length distribution plot to %s", out_path)


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",            type=int,  default=0)
    parser.add_argument("--pretrain-ckpt",  type=Path, default=None)
    parser.add_argument("--generator-ckpt", type=Path, default=None)
    parser.add_argument("--pretrain-cfg",   type=Path, default=None)
    parser.add_argument("--finetune-cfg",   type=Path, default=None)
    parser.add_argument("--n-generate",     type=int,  default=500)
    args = parser.parse_args()

    # allow CLI overrides of the module-level path constants
    global PRETRAIN_CKPT, GENERATOR_CKPT, PRETRAIN_CFG, FINETUNE_CFG
    if args.pretrain_ckpt:  PRETRAIN_CKPT  = args.pretrain_ckpt
    if args.generator_ckpt: GENERATOR_CKPT = args.generator_ckpt
    if args.pretrain_cfg:   PRETRAIN_CFG   = args.pretrain_cfg
    if args.finetune_cfg:   FINETUNE_CFG   = args.finetune_cfg

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load configs & data
    # ------------------------------------------------------------------
    pretrain_cfg = _load_yaml(PRETRAIN_CFG)
    finetune_cfg = _load_yaml(FINETUNE_CFG)
    data_cfg = pretrain_cfg["data"]
    fasta_paths = [PROJECT_ROOT / p for p in data_cfg["fasta_paths"]]

    logger.info("Loading training sequences …")
    train_ds, val_ds = build_seq2seq_datasets(
        fasta_paths=fasta_paths,
        max_len=data_cfg["max_len"],
        val_ratio=data_cfg["val_ratio"],
        seed=42,
        prefix_ratio=finetune_cfg["data"].get("prefix_ratio", 0.5),
        min_prefix_len=finetune_cfg["data"].get("min_prefix_len", 3),
        max_seq_len=finetune_cfg["generator"]["max_seq_len"],
    )
    train_sequences: list[str] = train_ds.sequences
    val_sequences: list[str] = val_ds.sequences
    logger.info("Training set size: %d sequences", len(train_sequences))

    # ------------------------------------------------------------------
    # 2. Load models
    # ------------------------------------------------------------------
    logger.info("Loading models …")
    jepa, generator = load_models(device)

    # ------------------------------------------------------------------
    # 3. Generate sequences
    # ------------------------------------------------------------------
    logger.info("Generating %d sequences …", args.n_generate)
    gen_seqs = generate_sequences(
        generator=generator,
        seq2seq_dataset=train_ds,
        device=device,
        n_generate=args.n_generate,
        batch_size=64,
        temperature=1.0,
        top_p=0.9,
    )

    # ------------------------------------------------------------------
    # 4. Score generated sequences with AMP classifiers
    # ------------------------------------------------------------------
    clf_scores: dict[str, np.ndarray] = {}

    # AMPlify — pre-trained transformer AMP classifier
    logger.info("Running AMPlify scorer …")
    amplify_scores = AMPlifyClassifier().predict_proba(gen_seqs)
    if not np.all(np.isnan(amplify_scores)):
        clf_scores["amplify"] = amplify_scores

    # ESM-2 fine-tuned AMP classifier (HuggingFace)
    logger.info("Running ESM-2 fine-tuned scorer …")
    esm_scores = ESMAMPClassifier(device=str(device)).predict_proba(gen_seqs)
    if not np.all(np.isnan(esm_scores)):
        clf_scores["esm2"] = esm_scores

    # Macrel — pre-trained CLI tool (graceful if not installed)
    logger.info("Running Macrel scorer …")
    macrel_scores = MacrelClassifier(threads=4).predict_proba(gen_seqs)
    if not np.all(np.isnan(macrel_scores)):
        clf_scores["macrel"] = macrel_scores

    # JEPA encoder probe — validates JEPA representation quality
    logger.info("Downloading negative sequences for JEPA probe …")
    from src.eval.amp_classifier import _fetch_non_amp_sequences
    neg_seqs = _fetch_non_amp_sequences(max_seqs=MAX_CLASSIFIER_SEQS)
    logger.info("Training JEPA-encoder probe …")
    import random as _rng2
    train_pos_cap = _rng2.sample(train_sequences, MAX_CLASSIFIER_SEQS) if len(train_sequences) > MAX_CLASSIFIER_SEQS else train_sequences
    jepa_clf = JEPAAMPClassifier(encoder=jepa.context_encoder, device=str(device))
    jepa_clf.fit(pos_seqs=train_pos_cap, neg_seqs=neg_seqs)
    clf_scores["jepa"] = jepa_clf.predict_proba(gen_seqs)

    # ensemble: nanmean of all valid classifiers
    valid_arrays = [v for v in clf_scores.values() if not np.all(np.isnan(v))]
    amp_scores: np.ndarray = np.nanmean(np.stack(valid_arrays, axis=0), axis=0)

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    logger.info("Computing metrics …")
    train_seq_set = set(train_sequences)

    val_score   = validity(gen_seqs)
    uniq_score  = uniqueness(gen_seqs)
    nov_score   = novelty(gen_seqs, train_seq_set)
    div_score   = diversity(gen_seqs, n_sample=1000)

    gen_pc  = physicochemical_stats(gen_seqs)
    train_pc = physicochemical_stats(train_sequences)

    gen_freq   = aa_frequency(gen_seqs)
    train_freq = aa_frequency(train_sequences)

    rep_eval = evaluate_representation_backbones(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        device=device,
        jepa_encoder=jepa.context_encoder,
        encoder_cfg=pretrain_cfg["model"],
    )

    # ensemble scores
    mean_amp  = float(np.nanmean(amp_scores))
    std_amp   = float(np.nanstd(amp_scores))
    pct_gt05  = float(np.nanmean(amp_scores > 0.5))

    # ------------------------------------------------------------------
    # 6. Print summary table
    # ------------------------------------------------------------------
    basic_rows = [
        ("Validity",    f"{val_score:.4f}"),
        ("Uniqueness",  f"{uniq_score:.4f}"),
        ("Novelty",     f"{nov_score:.4f}"),
        ("Diversity",   f"{div_score:.4f}"),
        ("AMP score (ensemble) mean ± std", f"{mean_amp:.4f} ± {std_amp:.4f}"),
        ("Fraction P(AMP) > 0.5 (ensemble)", f"{pct_gt05:.4f}"),
    ]
    # per-classifier breakdown
    for clf_name, scores in clf_scores.items():
        m = float(np.nanmean(scores))
        p = float(np.nanmean(scores > 0.5))
        basic_rows.append((f"  [{clf_name}] mean / P>0.5", f"{m:.4f} / {p:.4f}"))
    _print_table(basic_rows, title="Generated Sequences — Core Metrics")

    pc_rows = [
        ("",                         f"{'Generated':>14}  {'Training':>14}"),
        ("mean_length",              f"{gen_pc['mean_length']:>14.2f}  {train_pc['mean_length']:>14.2f}"),
        ("std_length",               f"{gen_pc['std_length']:>14.2f}  {train_pc['std_length']:>14.2f}"),
        ("mean_charge",              f"{gen_pc['mean_charge']:>14.3f}  {train_pc['mean_charge']:>14.3f}"),
        ("mean_hydrophobicity",      f"{gen_pc['mean_hydrophobicity']:>14.3f}  {train_pc['mean_hydrophobicity']:>14.3f}"),
        ("fraction_charged",         f"{gen_pc['fraction_charged']:>14.4f}  {train_pc['fraction_charged']:>14.4f}"),
        ("fraction_hydrophobic",     f"{gen_pc['fraction_hydrophobic']:>14.4f}  {train_pc['fraction_hydrophobic']:>14.4f}"),
    ]
    _print_table(pc_rows, title="Physicochemical Stats: Generated vs Training")

    # AA frequency comparison (top 5 largest discrepancies)
    discrepancies = sorted(
        AMINO_ACIDS,
        key=lambda aa: abs(gen_freq.get(aa, 0) - train_freq.get(aa, 0)),
        reverse=True,
    )
    freq_rows = [("", f"{'Generated':>12}  {'Training':>12}  {'Delta':>10}")]
    for aa in discrepancies[:10]:
        g = gen_freq.get(aa, 0)
        t = train_freq.get(aa, 0)
        freq_rows.append((f"  {aa}", f"{g:>12.4f}  {t:>12.4f}  {g - t:>+10.4f}"))
    _print_table(freq_rows, title="AA Frequency — Top 10 Discrepancies")

    if rep_eval:
        rep_rows = [("backbone", "ROC-AUC / Acc / pos_mean / neg_mean")]
        for name, metrics in rep_eval.items():
            rep_rows.append((
                name,
                f"{metrics['roc_auc']:.4f} / {metrics['accuracy']:.4f} / "
                f"{metrics['mean_positive_score']:.4f} / {metrics['mean_negative_score']:.4f}"
            ))
        _print_table(rep_rows, title="Representation Probe: Hold-out AMP Classification")

    # ------------------------------------------------------------------
    # 7. Save JSON report
    # ------------------------------------------------------------------
    report = {
        "n_generated": len(gen_seqs),
        "validity": val_score,
        "uniqueness": uniq_score,
        "novelty": nov_score,
        "diversity": div_score,
        "amp_score": {
            "ensemble_mean": mean_amp,
            "ensemble_std": std_amp,
            "ensemble_fraction_above_0.5": pct_gt05,
            "per_classifier": {
                name: {
                    "mean": float(np.nanmean(sc)),
                    "std": float(np.nanstd(sc)),
                    "fraction_above_0.5": float(np.nanmean(sc > 0.5)),
                }
                for name, sc in clf_scores.items()
            },
            "histogram": {
                "counts": np.histogram(amp_scores, bins=10, range=(0, 1))[0].tolist(),
                "bin_edges": np.histogram(amp_scores, bins=10, range=(0, 1))[1].tolist(),
            },
        },
        "physicochemical_generated": gen_pc,
        "physicochemical_training": train_pc,
        "aa_frequency_generated": gen_freq,
        "aa_frequency_training": train_freq,
        "representation_eval": rep_eval,
        "sample_sequences": gen_seqs[:20],
    }

    report_path = EVAL_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to %s", report_path)

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    logger.info("Generating plots …")
    plot_aa_freq(gen_freq, train_freq, EVAL_DIR / "aa_freq_comparison.png")
    plot_amp_score(amp_scores, EVAL_DIR / "amp_score_dist.png")
    plot_length_dist(gen_seqs, train_sequences, EVAL_DIR / "length_dist.png")

    print(f"\nEvaluation complete. Results saved to: {EVAL_DIR}/")
    print(f"  eval_report.json        — full numeric report")
    print(f"  aa_freq_comparison.png  — AA frequency bar chart")
    print(f"  amp_score_dist.png      — AMP score histogram")
    print(f"  length_dist.png         — length distribution")


if __name__ == "__main__":
    main()
