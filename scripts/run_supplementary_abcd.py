"""
Supplementary experiments A–D for JEPA-AMP paper.

A. JEPA vs MLM same-setting comparison (MIC, QMAP, cross-species)
B. MIC SOTA dataset-level alignment (literature mapping table)
C. Cross-species transfer statistical tests (bootstrap CI + permutation)
D. Homology leakage / identity distribution analysis

Usage:
    uv run python scripts/run_supplementary_abcd.py --experiments A C D --gpu 0
    uv run python scripts/run_supplementary_abcd.py --experiments all --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "eval_results" / "supplementary_abcd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS = [42, 123, 7]
SPECIES_PAIRS = [
    ("E. coli", "S. aureus"),
    ("E. coli", "P. aeruginosa"),
    ("S. aureus", "E. coli"),
    ("S. aureus", "P. aeruginosa"),
    ("P. aeruginosa", "E. coli"),
]


# ── encoder loaders ──────────────────────────────────────────────────────────

def load_jepa_encoder(device: torch.device):
    from src.models.jepa import JEPA
    ckpt = torch.load(
        PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
        map_location=device, weights_only=False,
    )
    jepa = JEPA(**ckpt["cfg"]["model"])
    jepa.load_state_dict(ckpt["model_state"])
    encoder = jepa.context_encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, ckpt["cfg"]["model"]["d_model"], ckpt["cfg"]["model"].get("max_seq_len", 52)


def load_mlm_encoder(device: torch.device):
    from src.models.mlm import MLMModel
    ckpt = torch.load(
        PROJECT_ROOT / "checkpoints/mlm_pretrain_868k/best_mlm.pt",
        map_location=device, weights_only=False,
    )
    mlm = MLMModel(**ckpt["cfg"]["model"])
    mlm.load_state_dict(ckpt["model_state"])
    encoder = mlm.encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    d_model = ckpt["cfg"]["model"]["d_model"]
    max_seq_len = ckpt["cfg"]["model"].get("max_seq_len", 52)
    return encoder, d_model, max_seq_len


def load_supervised_encoder(device: torch.device):
    """Random-init encoder with same architecture — no pretraining baseline."""
    from src.models.encoder import TransformerEncoder
    encoder = TransformerEncoder(
        d_model=384, nhead=8, num_layers=8,
        dim_feedforward=1536, dropout=0.1, max_seq_len=52,
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, 384, 52


def load_esm2_encoder(device: torch.device):
    from src.models.esm_head import load_esm2
    esm, alphabet, _ = load_esm2("esm2_t12_35M")
    esm = esm.to(device).eval()
    for p in esm.parameters():
        p.requires_grad_(False)
    return esm, alphabet, 480


# ── batch encoding / embedding ───────────────────────────────────────────────

def _batch_encode(seqs: list[str], device, max_aa_len: int = 50) -> torch.Tensor:
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:max_aa_len]) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out


def embed_batch_internal(encoder, seqs, device, max_aa_len=50):
    """Mean-pool embedding for JEPA/MLM/supervised encoder."""
    ids = _batch_encode(seqs, device, max_aa_len)
    with torch.no_grad():
        h = encoder(ids)  # (B, L, D)
    pad = ids == 0
    h = h.masked_fill(pad.unsqueeze(-1), 0.0)
    lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
    return h.sum(1) / lengths  # (B, d_model)


def embed_batch_esm2(esm, alphabet, seqs, device):
    bc = alphabet.get_batch_converter()
    data = [(f"s{i}", s) for i, s in enumerate(seqs)]
    _, _, tokens = bc(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = esm(tokens, repr_layers=[12], return_contacts=False)
    h = out["representations"][12]
    pad = tokens == alphabet.padding_idx
    h = h.masked_fill(pad.unsqueeze(-1), 0.0)
    lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
    return h.sum(1) / lengths


# ── data loading (shared) ────────────────────────────────────────────────────

def load_species(csv_path: Path, species: str, max_len: int = 50,
                 seed: int = 42) -> tuple[list, list, list]:
    recs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            seq = r["sequence"].strip().upper()
            if (
                r["is_modified"].strip() == "False"
                and r["bacterium"].strip() == species
                and 3 <= len(seq) <= max_len
                and all(c in AA for c in seq)
            ):
                try:
                    recs.append({"seq": seq, "log2_mic": float(r["value"])})
                except ValueError:
                    continue
    unique_seqs = sorted({r["seq"] for r in recs})
    rng = random.Random(seed)
    rng.shuffle(unique_seqs)
    n = len(unique_seqs)
    n_test = max(1, int(n * 0.15))
    n_val = max(1, int(n * 0.10))
    test_set = set(unique_seqs[:n_test])
    val_set = set(unique_seqs[n_test:n_test + n_val])
    train, val, test = [], [], []
    for r in recs:
        if r["seq"] in test_set:
            test.append(r)
        elif r["seq"] in val_set:
            val.append(r)
        else:
            train.append(r)
    return train, val, test


# ── simple MIC head ──────────────────────────────────────────────────────────

class MICHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_head(embed_fn, head, train_recs, val_recs, device,
               epochs=60, batch_size=128, lr=3e-4, patience=12):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    best_val, wait, best_state = float("inf"), 0, None

    def run_epoch(recs, is_train):
        random.shuffle(recs)
        losses = []
        for i in range(0, len(recs), batch_size):
            batch = recs[i:i + batch_size]
            seqs = [r["seq"] for r in batch]
            y = torch.tensor([r["log2_mic"] for r in batch], dtype=torch.float32, device=device)
            with torch.set_grad_enabled(is_train):
                with torch.no_grad():
                    emb = embed_fn(seqs)
                pred = head(emb)
                loss = F.huber_loss(pred, y)
            if is_train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    head.train()
    for ep in range(epochs):
        run_epoch(train_recs, True)
        head.eval()
        vl = run_epoch(val_recs, False)
        head.train()
        if vl < best_val - 1e-4:
            best_val, best_state, wait = vl, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        head.load_state_dict(best_state)


def eval_head(embed_fn, head, test_recs, device) -> dict:
    from scipy.stats import pearsonr, spearmanr
    head.eval()
    preds, trues = [], []
    for i in range(0, len(test_recs), 256):
        batch = test_recs[i:i + 256]
        seqs = [r["seq"] for r in batch]
        with torch.no_grad():
            emb = embed_fn(seqs)
            pred = head(emb).cpu().numpy()
        preds.extend(pred.tolist())
        trues.extend([r["log2_mic"] for r in batch])
    preds, trues = np.array(preds), np.array(trues)
    r, _ = pearsonr(preds, trues)
    rho, _ = spearmanr(preds, trues)
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    return {"pearson": float(r), "spearman": float(rho), "rmse": rmse, "n": len(trues)}


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT A: JEPA vs MLM vs supervised-only same-setting comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_A(device: torch.device):
    print("\n" + "=" * 70)
    print("EXPERIMENT A: JEPA vs MLM vs Supervised-only comparison")
    print("=" * 70)

    out_a = OUT_DIR / "A_jepa_vs_mlm"
    out_a.mkdir(parents=True, exist_ok=True)
    grampa = PROJECT_ROOT / "data" / "grampa.csv"

    # --- Part A1: Cross-species transfer for MLM + supervised ---
    print("\n--- A1: Cross-species transfer (MLM + supervised-only) ---")
    results_file = out_a / "cross_species_metrics.json"
    results = {}
    if results_file.exists():
        results = json.loads(results_file.read_text())

    model_loaders = {
        "jepa": ("internal", load_jepa_encoder),
        "mlm": ("internal", load_mlm_encoder),
        "supervised_only": ("internal", load_supervised_encoder),
    }

    for model_name, (enc_type, loader_fn) in model_loaders.items():
        if model_name in results and len(results[model_name]) >= len(SPECIES_PAIRS):
            existing_pairs = sum(1 for v in results[model_name].values() if len(v) >= len(SEEDS))
            if existing_pairs >= len(SPECIES_PAIRS):
                print(f"  [skip] {model_name} — already complete")
                continue

        print(f"\n  Loading {model_name}...")
        encoder, d_model, max_seq_len = loader_fn(device)

        def embed_fn(seqs, _enc=encoder, _dev=device, _msl=max_seq_len - 2):
            return embed_batch_internal(_enc, seqs, _dev, _msl)

        model_res = results.setdefault(model_name, {})

        for src_species, tgt_species in SPECIES_PAIRS:
            pair_key = f"{src_species}→{tgt_species}"
            pair_res = model_res.setdefault(pair_key, {})

            for seed in SEEDS:
                if str(seed) in pair_res:
                    continue
                print(f"    {model_name} {pair_key} seed={seed}")
                src_tr, src_val, _ = load_species(grampa, src_species, seed=seed)
                _, _, tgt_te = load_species(grampa, tgt_species, seed=seed)
                _, _, src_te = load_species(grampa, src_species, seed=seed)

                head = MICHead(d_model).to(device)
                train_head(embed_fn, head, src_tr, src_val, device)
                in_domain = eval_head(embed_fn, head, src_te, device)
                zero_shot = eval_head(embed_fn, head, tgt_te, device)
                pair_res[str(seed)] = {"in_domain": in_domain, "zero_shot": zero_shot}
                print(f"      in-domain={in_domain['pearson']:.3f}  zero-shot={zero_shot['pearson']:.3f}")
                results_file.write_text(json.dumps(results, indent=2))

        del encoder
        torch.cuda.empty_cache()

    # --- Part A2: QMAP for MLM (head-only fine-tune) ---
    print("\n--- A2: QMAP for MLM encoder (head finetune) ---")
    qmap_results_file = out_a / "qmap_mlm_metrics.json"
    if qmap_results_file.exists():
        print("  [skip] QMAP MLM already computed")
    else:
        _run_qmap_mlm(device, out_a)

    # --- Compile summary table ---
    _compile_A_summary(out_a, results)
    print(f"\nExperiment A outputs in {out_a}")


def _run_qmap_mlm(device: torch.device, out_dir: Path):
    """Run QMAP head-only finetune with MLM encoder, reusing finetune_qmap_jepa logic."""
    import subprocess
    mlm_ckpt = PROJECT_ROOT / "checkpoints/mlm_pretrain_868k/best_mlm.pt"
    qmap_out = out_dir / "qmap_mlm_head_finetune"

    # We need to modify the checkpoint loading. Instead of calling finetune_qmap_jepa.py
    # directly (which hardcodes JEPA), we run a modified version.
    # The simplest approach: create a JEPA-compatible wrapper checkpoint.
    print("  Creating MLM→JEPA-compatible checkpoint wrapper...")
    _ckpt = torch.load(mlm_ckpt, map_location="cpu", weights_only=False)
    mlm_cfg = _ckpt["cfg"]["model"]

    # The finetune_qmap_jepa.py calls load_encoder which does:
    #   JEPA(**cfg["model"]) then model.context_encoder
    # MLM encoder keys are "encoder.*", JEPA context_encoder keys are "context_encoder.*"
    # We need to create a fake JEPA checkpoint with context_encoder = MLM encoder
    from src.models.jepa import JEPA
    jepa_cfg = {
        **mlm_cfg,
        "predictor_depth": 2,
        "ema_decay": 0.996,
    }
    fake_jepa = JEPA(**jepa_cfg)
    # Copy MLM encoder weights into JEPA context_encoder
    mlm_encoder_state = {k: v for k, v in _ckpt["model_state"].items() if k.startswith("encoder.")}
    ctx_state = {k.replace("encoder.", ""): v for k, v in mlm_encoder_state.items()}
    fake_jepa.context_encoder.load_state_dict(ctx_state)
    # Also copy to target_encoder (EMA)
    fake_jepa.target_encoder.load_state_dict(ctx_state)

    wrapper_path = out_dir / "mlm_as_jepa_wrapper.pt"
    torch.save({
        "epoch": _ckpt["epoch"],
        "model_state": fake_jepa.state_dict(),
        "cfg": {"model": jepa_cfg},
    }, wrapper_path)
    print(f"  Wrapper checkpoint saved to {wrapper_path}")

    for seed in SEEDS:
        seed_out = qmap_out / f"seed_{seed}"
        if (seed_out / "summary.json").exists():
            print(f"  [skip] QMAP MLM seed={seed}")
            continue

        print(f"  Running QMAP MLM head finetune seed={seed}...")
        cmd = [
            sys.executable, "-m", "scripts.finetune_qmap_jepa",
            "--checkpoint", str(wrapper_path),
            "--out-dir", str(seed_out),
            "--target", "ecoli",
            "--mode", "head",
            "--device", f"cuda:{device.index or 0}",
            "--seed", str(seed),
            "--fp16",
            "--batch-size", "512",
            "--eval-batch-size", "512",
            "--num-workers", "2",
            "--mask-threads", "8",
        ]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    QMAP MLM seed={seed} FAILED:")
            print(result.stderr[-500:] if result.stderr else "no stderr")
        else:
            print(f"    QMAP MLM seed={seed} done")

    # Collect QMAP results
    qmap_results = {}
    for seed in SEEDS:
        summary_file = qmap_out / f"seed_{seed}" / "summary.json"
        if summary_file.exists():
            data = json.loads(summary_file.read_text())
            splits = data.get("splits", [])
            full_ecoli = [s["full_target_pearson"] for s in splits]
            high_eff = [s["high_eff_ecoli_pearson"] for s in splits if s.get("high_eff_ecoli_pearson") is not None]
            qmap_results[str(seed)] = {
                "full_ecoli": {"mean": float(np.mean(full_ecoli)), "sd": float(np.std(full_ecoli)), "splits": full_ecoli},
                "high_eff_ecoli": {"mean": float(np.mean(high_eff)), "sd": float(np.std(high_eff)), "splits": high_eff} if high_eff else None,
            }

    (out_dir / "qmap_mlm_metrics.json").write_text(json.dumps(qmap_results, indent=2))


def _compile_A_summary(out_dir: Path, cross_species_results: dict):
    """Build the main comparison table."""
    # Cross-species mean zero-shot
    summary = {}
    for model_name, model_res in cross_species_results.items():
        zs_vals = []
        for pair_key, pair_res in model_res.items():
            for seed_key, seed_data in pair_res.items():
                if "zero_shot" in seed_data:
                    zs_vals.append(seed_data["zero_shot"]["pearson"])
        summary[model_name] = {
            "mean_zero_shot_pearson": float(np.mean(zs_vals)) if zs_vals else None,
            "sd_zero_shot_pearson": float(np.std(zs_vals)) if zs_vals else None,
            "n_values": len(zs_vals),
        }

    # Add MIC from existing checkpoints
    jepa_mic = PROJECT_ROOT / "checkpoints/formal_mic_868k_transformer/test_metrics.json"
    mlm_mic = PROJECT_ROOT / "checkpoints/formal_mic_mlm_transformer/test_metrics.json"
    if jepa_mic.exists():
        summary.setdefault("jepa", {})["mic_pearson"] = json.loads(jepa_mic.read_text())["pearson"]
    if mlm_mic.exists():
        summary.setdefault("mlm", {})["mic_pearson"] = json.loads(mlm_mic.read_text())["pearson"]

    # Add QMAP from existing results + new MLM
    jepa_qmap = PROJECT_ROOT / "eval_results/qmap_stats_pack/metrics.json"
    if jepa_qmap.exists():
        qdata = json.loads(jepa_qmap.read_text())
        summary.setdefault("jepa", {})["qmap_high_eff_mean"] = qdata["seed_means"]["head_only"]["high_eff_ecoli"]["mean"]
        summary.setdefault("jepa", {})["qmap_full_ecoli_mean"] = qdata["seed_means"]["head_only"]["full_ecoli"]["mean"]

    mlm_qmap = out_dir / "qmap_mlm_metrics.json"
    if mlm_qmap.exists():
        qdata = json.loads(mlm_qmap.read_text())
        high_effs = [v["high_eff_ecoli"]["mean"] for v in qdata.values() if v.get("high_eff_ecoli")]
        full_ecolis = [v["full_ecoli"]["mean"] for v in qdata.values()]
        if high_effs:
            summary.setdefault("mlm", {})["qmap_high_eff_mean"] = float(np.mean(high_effs))
        if full_ecolis:
            summary.setdefault("mlm", {})["qmap_full_ecoli_mean"] = float(np.mean(full_ecolis))

    (out_dir / "A_summary.json").write_text(json.dumps(summary, indent=2))

    # Write markdown table
    lines = [
        "# Experiment A: JEPA vs MLM vs Supervised-only",
        "",
        "| Model | MIC Pearson | QMAP full E.coli | QMAP high-eff | Cross-species mean zero-shot |",
        "|---|---|---|---|---|",
    ]
    for model in ["jepa", "mlm", "supervised_only", "esm2"]:
        s = summary.get(model, {})
        mic = f"{s['mic_pearson']:.3f}" if s.get("mic_pearson") else "—"
        qf = f"{s['qmap_full_ecoli_mean']:.3f}" if s.get("qmap_full_ecoli_mean") else "—"
        qh = f"{s['qmap_high_eff_mean']:.3f}" if s.get("qmap_high_eff_mean") else "—"
        zs = f"{s['mean_zero_shot_pearson']:.3f} ± {s['sd_zero_shot_pearson']:.3f}" if s.get("mean_zero_shot_pearson") else "—"
        lines.append(f"| {model} | {mic} | {qf} | {qh} | {zs} |")

    (out_dir / "A_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"  Summary table written to {out_dir / 'A_SUMMARY.md'}")


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT B: MIC SOTA dataset-level alignment
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_B(device: torch.device):
    print("\n" + "=" * 70)
    print("EXPERIMENT B: MIC SOTA dataset-level alignment")
    print("=" * 70)

    out_b = OUT_DIR / "B_sota_alignment"
    out_b.mkdir(parents=True, exist_ok=True)

    grampa = PROJECT_ROOT / "data" / "grampa.csv"
    species_counts = {}
    total = 0
    with open(grampa) as f:
        for r in csv.DictReader(f):
            if r["is_modified"].strip() == "False":
                sp = r["bacterium"].strip()
                species_counts[sp] = species_counts.get(sp, 0) + 1
                total += 1

    table = {
        "our_data": {
            "source": "GRAMPA (curated DBAASP subset)",
            "total_records": total,
            "species_breakdown": dict(sorted(species_counts.items(), key=lambda x: -x[1])[:10]),
            "target": "log2 MIC (continuous)",
            "split": "sequence-level, no homology control in standard split; QMAP uses homology-aware splits",
        },
        "prior_methods": [
            {
                "method": "MBC-Attention (Yan et al. 2022)",
                "data": "DBAASP v3, E. coli only",
                "target": "MBC (binary: active/inactive at threshold)",
                "species": "E. coli",
                "metric": "Accuracy, AUC",
                "comparable": "Partially — binary vs continuous, same source DB",
                "notes": "Different threshold binarization; not directly comparable Pearson",
            },
            {
                "method": "ANIA (Li et al. 2024)",
                "data": "DBAASP, multi-species (E/S/P)",
                "target": "MIC regression (log2)",
                "species": "E. coli, S. aureus, P. aeruginosa",
                "metric": "Pearson, RMSE",
                "comparable": "Strong — same DB, same species, same target type",
                "notes": "Protocol may differ (split, preprocessing). Best candidate for head-to-head if splits align",
            },
            {
                "method": "LLAMP (Tran et al. 2023)",
                "data": "DBAASP + custom curation",
                "target": "Species-aware MIC (peptide + species pair)",
                "species": "Multiple",
                "metric": "Pearson, Spearman",
                "comparable": "Related but not direct — species is an input feature, different formulation",
                "notes": "Could compare on overlapping species if evaluation set is shared",
            },
            {
                "method": "QMAP Benchmark (Cai et al. 2025)",
                "data": "DBAASP v4, standardized homology-aware splits",
                "target": "log10 MIC (E. coli), HC50",
                "species": "E. coli (primary)",
                "metric": "Pearson (5 splits × mean)",
                "comparable": "Best — standardized protocol, public benchmark, our primary eval",
                "notes": "We already report this. JEPA high-eff 0.388 vs their top baseline 0.29",
            },
            {
                "method": "ESM2 + linear (QMAP baseline)",
                "data": "Same QMAP splits",
                "target": "log10 MIC",
                "species": "E. coli",
                "metric": "Pearson",
                "comparable": "Direct — same splits, same protocol",
                "notes": "full_ecoli=0.36, high_eff=0.16, hc50=0.07",
            },
        ],
        "recommendation": (
            "QMAP is the strongest apples-to-apples comparison (standardized, public). "
            "ANIA is the best candidate for additional head-to-head if they release evaluation splits. "
            "MBC-Attention uses binary targets so only qualitative comparison is possible. "
            "LLAMP has a different input formulation (species as feature) so comparison requires care."
        ),
    }

    (out_b / "B_sota_alignment.json").write_text(json.dumps(table, indent=2))

    lines = [
        "# Experiment B: MIC SOTA Dataset-Level Alignment",
        "",
        "## Our data (GRAMPA)",
        f"- Total records: {total}",
        f"- Top species: {', '.join(f'{k} ({v})' for k, v in list(sorted(species_counts.items(), key=lambda x: -x[1]))[:5])}",
        "",
        "## Comparison with prior methods",
        "",
        "| Prior method | Data | Target | Species | Directly comparable? |",
        "|---|---|---|---|---|",
    ]
    for m in table["prior_methods"]:
        lines.append(f"| {m['method']} | {m['data']} | {m['target']} | {m['species']} | {m['comparable']} |")

    lines += [
        "",
        "## Recommendation",
        table["recommendation"],
    ]

    (out_b / "B_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Experiment B outputs in {out_b}")


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT C: Cross-species transfer statistical tests
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_C(device: torch.device):
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Cross-species statistical tests")
    print("=" * 70)

    out_c = OUT_DIR / "C_statistical_tests"
    out_c.mkdir(parents=True, exist_ok=True)

    # Load existing cross-species results
    existing = PROJECT_ROOT / "eval_results" / "cross_species_transfer" / "metrics.json"
    supp_a = OUT_DIR / "A_jepa_vs_mlm" / "cross_species_metrics.json"

    results = {}
    if existing.exists():
        results = json.loads(existing.read_text())
    if supp_a.exists():
        supp_data = json.loads(supp_a.read_text())
        for model, model_res in supp_data.items():
            if model not in results:
                results[model] = model_res

    if "jepa" not in results or "esm2" not in results:
        print("  ERROR: Need both JEPA and ESM2 cross-species results. Run original cross_species_transfer.py first.")
        return

    # Collect paired zero-shot values
    jepa_zs, esm2_zs, mlm_zs = [], [], []
    pair_labels = []
    for src, tgt in SPECIES_PAIRS:
        key = f"{src}→{tgt}"
        for seed in SEEDS:
            seed_key = str(seed)
            j_val = results.get("jepa", {}).get(key, {}).get(seed_key, {}).get("zero_shot", {}).get("pearson")
            e_val = results.get("esm2", {}).get(key, {}).get(seed_key, {}).get("zero_shot", {}).get("pearson")
            m_val = results.get("mlm", {}).get(key, {}).get(seed_key, {}).get("zero_shot", {}).get("pearson")
            if j_val is not None and e_val is not None:
                jepa_zs.append(j_val)
                esm2_zs.append(e_val)
                pair_labels.append(f"{key}_s{seed}")
            if m_val is not None:
                mlm_zs.append(m_val)

    jepa_zs = np.array(jepa_zs)
    esm2_zs = np.array(esm2_zs)
    deltas = jepa_zs - esm2_zs

    # 1. Mean / route-level stats
    route_stats = {}
    for src, tgt in SPECIES_PAIRS:
        key = f"{src}→{tgt}"
        j_vals = [results["jepa"][key][str(s)]["zero_shot"]["pearson"] for s in SEEDS
                  if str(s) in results["jepa"].get(key, {})]
        e_vals = [results["esm2"][key][str(s)]["zero_shot"]["pearson"] for s in SEEDS
                  if str(s) in results["esm2"].get(key, {})]
        if j_vals and e_vals:
            j_arr, e_arr = np.array(j_vals), np.array(e_vals)
            route_stats[key] = {
                "jepa_mean": float(j_arr.mean()),
                "jepa_sd": float(j_arr.std()),
                "esm2_mean": float(e_arr.mean()),
                "esm2_sd": float(e_arr.std()),
                "delta_mean": float((j_arr - e_arr).mean()),
                "delta_sd": float((j_arr - e_arr).std()),
            }

    # 2. Paired bootstrap CI for mean delta
    n_bootstrap = 10000
    rng = np.random.RandomState(42)
    boot_means = np.empty(n_bootstrap)
    n = len(deltas)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = deltas[idx].mean()
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    # 3. Permutation test (H0: JEPA and ESM2 are exchangeable)
    n_perm = 10000
    observed_mean_delta = float(deltas.mean())
    perm_deltas = np.empty(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_deltas[p] = (deltas * signs).mean()
    p_value = float((np.abs(perm_deltas) >= np.abs(observed_mean_delta)).mean())

    # 4. Route-level bootstrap CIs
    route_cis = {}
    for key, rs in route_stats.items():
        j_vals = [results["jepa"][key][str(s)]["zero_shot"]["pearson"] for s in SEEDS
                  if str(s) in results["jepa"].get(key, {})]
        e_vals = [results["esm2"][key][str(s)]["zero_shot"]["pearson"] for s in SEEDS
                  if str(s) in results["esm2"].get(key, {})]
        j_arr, e_arr = np.array(j_vals), np.array(e_vals)
        route_deltas = j_arr - e_arr
        boot_route = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.randint(0, len(route_deltas), size=len(route_deltas))
            boot_route[b] = route_deltas[idx].mean()
        route_cis[key] = {
            "delta_mean": float(route_deltas.mean()),
            "ci_95": [float(np.percentile(boot_route, 2.5)), float(np.percentile(boot_route, 97.5))],
        }

    output = {
        "global": {
            "n_paired_observations": int(n),
            "jepa_mean_zero_shot": float(jepa_zs.mean()),
            "esm2_mean_zero_shot": float(esm2_zs.mean()),
            "mean_delta": observed_mean_delta,
            "bootstrap_ci_95": [ci_lo, ci_hi],
            "permutation_p_value": p_value,
            "n_bootstrap": n_bootstrap,
            "n_permutations": n_perm,
        },
        "per_route": route_stats,
        "per_route_bootstrap_ci": route_cis,
    }

    # --- JEPA vs MLM statistical tests ---
    jepa_vs_mlm = {}
    mlm_route_stats = {}
    mlm_route_cis = {}
    if len(mlm_zs) == len(jepa_zs) and len(mlm_zs) > 0:
        mlm_arr = np.array(mlm_zs)
        deltas_jm = jepa_zs - mlm_arr
        observed_jm = float(deltas_jm.mean())

        boot_jm = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.randint(0, len(deltas_jm), size=len(deltas_jm))
            boot_jm[b] = deltas_jm[idx].mean()
        ci_jm_lo = float(np.percentile(boot_jm, 2.5))
        ci_jm_hi = float(np.percentile(boot_jm, 97.5))

        perm_jm = np.empty(n_perm)
        for p in range(n_perm):
            signs = rng.choice([-1, 1], size=len(deltas_jm))
            perm_jm[p] = (deltas_jm * signs).mean()
        p_value_jm = float((np.abs(perm_jm) >= np.abs(observed_jm)).mean())

        jepa_vs_mlm = {
            "n_paired_observations": int(len(deltas_jm)),
            "jepa_mean_zero_shot": float(jepa_zs.mean()),
            "mlm_mean_zero_shot": float(mlm_arr.mean()),
            "mean_delta": observed_jm,
            "bootstrap_ci_95": [ci_jm_lo, ci_jm_hi],
            "permutation_p_value": p_value_jm,
        }

        for src, tgt in SPECIES_PAIRS:
            key = f"{src}→{tgt}"
            j_vals = [results["jepa"][key][str(s)]["zero_shot"]["pearson"] for s in SEEDS
                      if str(s) in results["jepa"].get(key, {})]
            m_vals = [results["mlm"][key][str(s)]["zero_shot"]["pearson"] for s in SEEDS
                      if str(s) in results.get("mlm", {}).get(key, {})]
            if j_vals and m_vals:
                j_a, m_a = np.array(j_vals), np.array(m_vals)
                mlm_route_stats[key] = {
                    "mlm_mean": float(m_a.mean()),
                    "mlm_sd": float(m_a.std()),
                }
                rd = j_a - m_a
                boot_rd = np.empty(n_bootstrap)
                for b in range(n_bootstrap):
                    idx = rng.randint(0, len(rd), size=len(rd))
                    boot_rd[b] = rd[idx].mean()
                mlm_route_cis[key] = {
                    "delta_mean": float(rd.mean()),
                    "ci_95": [float(np.percentile(boot_rd, 2.5)), float(np.percentile(boot_rd, 97.5))],
                }
    elif mlm_zs:
        mlm_arr = np.array(mlm_zs)
        jepa_vs_mlm = {"mlm_mean_zero_shot": float(mlm_arr.mean()), "note": "unequal pair count, no stat test"}

    # --- JEPA vs supervised-only statistical tests ---
    sup_zs = []
    for src, tgt in SPECIES_PAIRS:
        key = f"{src}→{tgt}"
        for seed in SEEDS:
            s_val = results.get("supervised_only", {}).get(key, {}).get(str(seed), {}).get("zero_shot", {}).get("pearson")
            if s_val is not None:
                sup_zs.append(s_val)

    jepa_vs_sup = {}
    sup_route_stats = {}
    sup_route_cis = {}
    if len(sup_zs) == len(jepa_zs) and len(sup_zs) > 0:
        sup_arr = np.array(sup_zs)
        deltas_js = jepa_zs - sup_arr
        observed_js = float(deltas_js.mean())

        boot_js = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.randint(0, len(deltas_js), size=len(deltas_js))
            boot_js[b] = deltas_js[idx].mean()
        ci_js_lo = float(np.percentile(boot_js, 2.5))
        ci_js_hi = float(np.percentile(boot_js, 97.5))

        perm_js = np.empty(n_perm)
        for p in range(n_perm):
            signs = rng.choice([-1, 1], size=len(deltas_js))
            perm_js[p] = (deltas_js * signs).mean()
        p_value_js = float((np.abs(perm_js) >= np.abs(observed_js)).mean())

        jepa_vs_sup = {
            "n_paired_observations": int(len(deltas_js)),
            "jepa_mean_zero_shot": float(jepa_zs.mean()),
            "supervised_mean_zero_shot": float(sup_arr.mean()),
            "mean_delta": observed_js,
            "bootstrap_ci_95": [ci_js_lo, ci_js_hi],
            "permutation_p_value": p_value_js,
        }

        for src, tgt in SPECIES_PAIRS:
            key = f"{src}→{tgt}"
            j_vals = [results["jepa"][key][str(s)]["zero_shot"]["pearson"] for s in SEEDS
                      if str(s) in results["jepa"].get(key, {})]
            s_vals = [results["supervised_only"][key][str(s)]["zero_shot"]["pearson"] for s in SEEDS
                      if str(s) in results.get("supervised_only", {}).get(key, {})]
            if j_vals and s_vals:
                j_a, s_a = np.array(j_vals), np.array(s_vals)
                sup_route_stats[key] = {
                    "sup_mean": float(s_a.mean()),
                    "sup_sd": float(s_a.std()),
                }
                rd = j_a - s_a
                boot_rd = np.empty(n_bootstrap)
                for b in range(n_bootstrap):
                    idx = rng.randint(0, len(rd), size=len(rd))
                    boot_rd[b] = rd[idx].mean()
                sup_route_cis[key] = {
                    "delta_mean": float(rd.mean()),
                    "ci_95": [float(np.percentile(boot_rd, 2.5)), float(np.percentile(boot_rd, 97.5))],
                }

    output = {
        "jepa_vs_esm2": {
            "n_paired_observations": int(n),
            "jepa_mean_zero_shot": float(jepa_zs.mean()),
            "esm2_mean_zero_shot": float(esm2_zs.mean()),
            "mean_delta": observed_mean_delta,
            "bootstrap_ci_95": [ci_lo, ci_hi],
            "permutation_p_value": p_value,
            "n_bootstrap": n_bootstrap,
            "n_permutations": n_perm,
        },
        "jepa_vs_mlm": jepa_vs_mlm,
        "jepa_vs_supervised": jepa_vs_sup,
        "per_route": route_stats,
        "per_route_bootstrap_ci": route_cis,
    }

    (out_c / "C_statistical_tests.json").write_text(json.dumps(output, indent=2))

    # Write summary
    lines = [
        "# Experiment C: Cross-Species Transfer Statistical Tests",
        "",
        "## 1. JEPA vs ESM2",
        f"- n = {n} paired observations (5 routes × 3 seeds)",
        f"- JEPA mean zero-shot Pearson: {jepa_zs.mean():.3f}",
        f"- ESM2 mean zero-shot Pearson: {esm2_zs.mean():.3f}",
        f"- Mean Δ (JEPA − ESM2): {observed_mean_delta:+.3f}",
        f"- 95% bootstrap CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]",
        f"- Permutation test p-value: {p_value:.4f}",
        "",
    ]

    if jepa_vs_mlm and "mean_delta" in jepa_vs_mlm:
        lines += [
            "## 2. JEPA vs MLM",
            f"- n = {jepa_vs_mlm['n_paired_observations']} paired observations",
            f"- JEPA mean zero-shot Pearson: {jepa_vs_mlm['jepa_mean_zero_shot']:.3f}",
            f"- MLM mean zero-shot Pearson: {jepa_vs_mlm['mlm_mean_zero_shot']:.3f}",
            f"- Mean Δ (JEPA − MLM): {jepa_vs_mlm['mean_delta']:+.3f}",
            f"- 95% bootstrap CI: [{jepa_vs_mlm['bootstrap_ci_95'][0]:+.3f}, {jepa_vs_mlm['bootstrap_ci_95'][1]:+.3f}]",
            f"- Permutation test p-value: {jepa_vs_mlm['permutation_p_value']:.4f}",
            "",
        ]

    if jepa_vs_sup and "mean_delta" in jepa_vs_sup:
        lines += [
            "## 3. JEPA vs Supervised-only",
            f"- n = {jepa_vs_sup['n_paired_observations']} paired observations",
            f"- JEPA mean zero-shot Pearson: {jepa_vs_sup['jepa_mean_zero_shot']:.3f}",
            f"- Supervised mean zero-shot Pearson: {jepa_vs_sup['supervised_mean_zero_shot']:.3f}",
            f"- Mean Δ (JEPA − Supervised): {jepa_vs_sup['mean_delta']:+.3f}",
            f"- 95% bootstrap CI: [{jepa_vs_sup['bootstrap_ci_95'][0]:+.3f}, {jepa_vs_sup['bootstrap_ci_95'][1]:+.3f}]",
            f"- Permutation test p-value: {jepa_vs_sup['permutation_p_value']:.4f}",
            "",
        ]

    lines += [
        "## Per-route (JEPA vs ESM2)",
        "",
        "| Route | JEPA | ESM2 | Δ (JEPA−ESM2) | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for key in [f"{s}→{t}" for s, t in SPECIES_PAIRS]:
        rs = route_stats.get(key, {})
        rc = route_cis.get(key, {})
        lines.append(
            f"| {key} | {rs.get('jepa_mean', 0):.3f} ± {rs.get('jepa_sd', 0):.3f} | "
            f"{rs.get('esm2_mean', 0):.3f} ± {rs.get('esm2_sd', 0):.3f} | "
            f"{rc.get('delta_mean', 0):+.3f} | "
            f"[{rc.get('ci_95', [0, 0])[0]:+.3f}, {rc.get('ci_95', [0, 0])[1]:+.3f}] |"
        )

    if mlm_route_cis:
        lines += [
            "",
            "## Per-route (JEPA vs MLM)",
            "",
            "| Route | JEPA | MLM | Δ (JEPA−MLM) | 95% CI |",
            "|---|---|---|---|---|",
        ]
        for key in [f"{s}→{t}" for s, t in SPECIES_PAIRS]:
            rs = route_stats.get(key, {})
            ms = mlm_route_stats.get(key, {})
            mc = mlm_route_cis.get(key, {})
            lines.append(
                f"| {key} | {rs.get('jepa_mean', 0):.3f} ± {rs.get('jepa_sd', 0):.3f} | "
                f"{ms.get('mlm_mean', 0):.3f} ± {ms.get('mlm_sd', 0):.3f} | "
                f"{mc.get('delta_mean', 0):+.3f} | "
                f"[{mc.get('ci_95', [0, 0])[0]:+.3f}, {mc.get('ci_95', [0, 0])[1]:+.3f}] |"
            )

    if sup_route_cis:
        lines += [
            "",
            "## Per-route (JEPA vs Supervised-only)",
            "",
            "| Route | JEPA | Supervised | Δ (JEPA−Sup) | 95% CI |",
            "|---|---|---|---|---|",
        ]
        for key in [f"{s}→{t}" for s, t in SPECIES_PAIRS]:
            rs = route_stats.get(key, {})
            ss = sup_route_stats.get(key, {})
            sc = sup_route_cis.get(key, {})
            lines.append(
                f"| {key} | {rs.get('jepa_mean', 0):.3f} ± {rs.get('jepa_sd', 0):.3f} | "
                f"{ss.get('sup_mean', 0):.3f} ± {ss.get('sup_sd', 0):.3f} | "
                f"{sc.get('delta_mean', 0):+.3f} | "
                f"[{sc.get('ci_95', [0, 0])[0]:+.3f}, {sc.get('ci_95', [0, 0])[1]:+.3f}] |"
            )

    (out_c / "C_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Experiment C outputs in {out_c}")


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT D: Homology leakage / identity distribution
# ══════════════════════════════════════════════════════════════════════════════

def _seq_identity(s1: str, s2: str) -> float:
    """Simple ungapped sequence identity = matches / max(len1, len2)."""
    min_len = min(len(s1), len(s2))
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    matches = sum(1 for a, b in zip(s1[:min_len], s2[:min_len]) if a == b)
    return matches / max_len


def _nearest_identity(query_seqs: list[str], ref_seqs: list[str],
                      sample_limit: int = 2000) -> np.ndarray:
    """For each query, find max identity to any ref sequence."""
    rng = random.Random(42)
    if len(query_seqs) > sample_limit:
        query_seqs = rng.sample(query_seqs, sample_limit)

    identities = np.zeros(len(query_seqs))
    for i, q in enumerate(query_seqs):
        best = 0.0
        for r in ref_seqs:
            ident = _seq_identity(q, r)
            if ident > best:
                best = ident
                if best >= 1.0:
                    break
        identities[i] = best
    return identities


def run_experiment_D(device: torch.device):
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Homology leakage / identity distribution")
    print("=" * 70)

    out_d = OUT_DIR / "D_homology_leakage"
    out_d.mkdir(parents=True, exist_ok=True)

    grampa = PROJECT_ROOT / "data" / "grampa.csv"
    all_species = ["E. coli", "S. aureus", "P. aeruginosa"]

    # 1. Source-target nearest identity for cross-species
    print("\n--- D1: Source-target nearest identity ---")
    d1_file = out_d / "D1_cross_species_identity.json"
    if d1_file.exists():
        print("  [skip] Already computed")
        d1_results = json.loads(d1_file.read_text())
    else:
        d1_results = {}
        for src_sp in all_species:
            src_tr, _, _ = load_species(grampa, src_sp, seed=42)
            src_seqs = [r["seq"] for r in src_tr]
            for tgt_sp in all_species:
                if src_sp == tgt_sp:
                    continue
                key = f"{src_sp}→{tgt_sp}"
                print(f"  {key}: computing nearest identity...")
                _, _, tgt_te = load_species(grampa, tgt_sp, seed=42)
                tgt_seqs = [r["seq"] for r in tgt_te]
                idents = _nearest_identity(tgt_seqs, src_seqs)
                d1_results[key] = {
                    "n_src_train": len(src_seqs),
                    "n_tgt_test": len(tgt_seqs),
                    "nearest_identity_mean": float(idents.mean()),
                    "nearest_identity_median": float(np.median(idents)),
                    "nearest_identity_p95": float(np.percentile(idents, 95)),
                    "nearest_identity_max": float(idents.max()),
                    "frac_above_80": float((idents >= 0.8).mean()),
                    "frac_above_70": float((idents >= 0.7).mean()),
                    "frac_above_60": float((idents >= 0.6).mean()),
                }
                print(f"    mean={idents.mean():.3f} median={np.median(idents):.3f} "
                      f"max={idents.max():.3f} >80%={d1_results[key]['frac_above_80']:.3f}")

        d1_file.write_text(json.dumps(d1_results, indent=2))

    # 2. Train-test identity distribution for MIC regression
    print("\n--- D2: Train-test identity for MIC (GRAMPA) ---")
    d2_file = out_d / "D2_mic_train_test_identity.json"
    if d2_file.exists():
        print("  [skip] Already computed")
        d2_results = json.loads(d2_file.read_text())
    else:
        d2_results = {}
        for sp in all_species:
            print(f"  {sp}: computing train→test identity...")
            tr, _, te = load_species(grampa, sp, seed=42)
            tr_seqs = [r["seq"] for r in tr]
            te_seqs = [r["seq"] for r in te]
            idents = _nearest_identity(te_seqs, tr_seqs)
            d2_results[sp] = {
                "n_train": len(tr_seqs),
                "n_test": len(te_seqs),
                "nearest_identity_mean": float(idents.mean()),
                "nearest_identity_median": float(np.median(idents)),
                "nearest_identity_p95": float(np.percentile(idents, 95)),
                "nearest_identity_max": float(idents.max()),
                "frac_above_80": float((idents >= 0.8).mean()),
                "frac_above_70": float((idents >= 0.7).mean()),
            }
            print(f"    mean={idents.mean():.3f} >80%={(idents >= 0.8).mean():.3f}")

        d2_file.write_text(json.dumps(d2_results, indent=2))

    # 3. Robustness: re-evaluate cross-species at identity cutoffs
    print("\n--- D3: Cross-species at identity cutoffs (>80%, >70% removed) ---")
    d3_file = out_d / "D3_identity_cutoff_robustness.json"
    if d3_file.exists():
        print("  [skip] Already computed")
    else:
        d3_results = _run_cutoff_robustness(device, grampa, out_d)
        d3_file.write_text(json.dumps(d3_results, indent=2))

    # Compile summary
    _compile_D_summary(out_d)
    print(f"\nExperiment D outputs in {out_d}")


def _run_cutoff_robustness(device: torch.device, grampa: Path, out_dir: Path) -> dict:
    """Re-run cross-species with high-identity test peptides removed."""
    results = {}

    print("  Loading JEPA encoder for cutoff robustness...")
    encoder, d_model, max_seq_len = load_jepa_encoder(device)

    def embed_fn(seqs, _enc=encoder, _dev=device, _msl=max_seq_len - 2):
        return embed_batch_internal(_enc, seqs, _dev, _msl)

    for cutoff in [0.8, 0.7]:
        cutoff_key = f"cutoff_{int(cutoff * 100)}"
        results[cutoff_key] = {}
        print(f"\n  Identity cutoff: {cutoff}")

        for src_sp, tgt_sp in SPECIES_PAIRS:
            pair_key = f"{src_sp}→{tgt_sp}"
            pair_results = []

            for seed in SEEDS:
                src_tr, src_val, _ = load_species(grampa, src_sp, seed=seed)
                _, _, tgt_te = load_species(grampa, tgt_sp, seed=seed)
                _, _, src_te = load_species(grampa, src_sp, seed=seed)

                src_seqs = [r["seq"] for r in src_tr]

                # Filter test set: remove peptides with identity > cutoff to any train peptide
                filtered_test = []
                for r in tgt_te:
                    max_id = max((_seq_identity(r["seq"], s) for s in src_seqs), default=0)
                    if max_id < cutoff:
                        filtered_test.append(r)

                n_orig = len(tgt_te)
                n_filt = len(filtered_test)

                if n_filt < 10:
                    pair_results.append({
                        "seed": seed, "n_original": n_orig, "n_filtered": n_filt,
                        "pearson": None, "note": "too few samples after filtering"
                    })
                    continue

                head = MICHead(d_model).to(device)
                train_head(embed_fn, head, src_tr, src_val, device)
                zs = eval_head(embed_fn, head, filtered_test, device)

                pair_results.append({
                    "seed": seed,
                    "n_original": n_orig,
                    "n_filtered": n_filt,
                    "pearson": zs["pearson"],
                    "spearman": zs["spearman"],
                })
                print(f"    {pair_key} seed={seed}: {n_orig}→{n_filt} peptides, Pearson={zs['pearson']:.3f}")

            results[cutoff_key][pair_key] = pair_results

    del encoder
    torch.cuda.empty_cache()
    return results


def _compile_D_summary(out_dir: Path):
    d1 = json.loads((out_dir / "D1_cross_species_identity.json").read_text()) if (out_dir / "D1_cross_species_identity.json").exists() else {}
    d2 = json.loads((out_dir / "D2_mic_train_test_identity.json").read_text()) if (out_dir / "D2_mic_train_test_identity.json").exists() else {}
    d3 = json.loads((out_dir / "D3_identity_cutoff_robustness.json").read_text()) if (out_dir / "D3_identity_cutoff_robustness.json").exists() else {}

    lines = [
        "# Experiment D: Homology Leakage / Identity Distribution",
        "",
        "## D1: Cross-species source→target nearest identity",
        "",
        "| Route | Mean | Median | P95 | Max | >80% | >70% |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, v in d1.items():
        lines.append(
            f"| {key} | {v['nearest_identity_mean']:.3f} | {v['nearest_identity_median']:.3f} | "
            f"{v['nearest_identity_p95']:.3f} | {v['nearest_identity_max']:.3f} | "
            f"{v['frac_above_80']:.3f} | {v['frac_above_70']:.3f} |"
        )

    lines += [
        "",
        "## D2: MIC train→test nearest identity (same species)",
        "",
        "| Species | Mean | Median | P95 | >80% | >70% |",
        "|---|---|---|---|---|---|",
    ]
    for sp, v in d2.items():
        lines.append(
            f"| {sp} | {v['nearest_identity_mean']:.3f} | {v['nearest_identity_median']:.3f} | "
            f"{v['nearest_identity_p95']:.3f} | {v['frac_above_80']:.3f} | {v['frac_above_70']:.3f} |"
        )

    lines += [
        "",
        "## D3: Cross-species Pearson after removing high-identity peptides",
        "",
        "| Route | No cutoff | <80% only | <70% only |",
        "|---|---|---|---|",
    ]
    # Get baseline from existing results
    existing = PROJECT_ROOT / "eval_results" / "cross_species_transfer" / "metrics.json"
    baseline = {}
    if existing.exists():
        bdata = json.loads(existing.read_text())
        for key in [f"{s}→{t}" for s, t in SPECIES_PAIRS]:
            vals = [bdata.get("jepa", {}).get(key, {}).get(str(s), {}).get("zero_shot", {}).get("pearson")
                    for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                baseline[key] = f"{np.mean(vals):.3f}"

    for key in [f"{s}→{t}" for s, t in SPECIES_PAIRS]:
        bl = baseline.get(key, "—")
        c80_vals = [r["pearson"] for r in d3.get("cutoff_80", {}).get(key, []) if r.get("pearson") is not None]
        c70_vals = [r["pearson"] for r in d3.get("cutoff_70", {}).get(key, []) if r.get("pearson") is not None]
        c80 = f"{np.mean(c80_vals):.3f}" if c80_vals else "—"
        c70 = f"{np.mean(c70_vals):.3f}" if c70_vals else "—"
        lines.append(f"| {key} | {bl} | {c80} | {c70} |")

    (out_dir / "D_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"  Summary written to {out_dir / 'D_SUMMARY.md'}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run supplementary experiments A–D")
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        choices=["all", "A", "B", "C", "D"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exps = set(args.experiments)
    run_all = "all" in exps

    if run_all or "B" in exps:
        run_experiment_B(device)

    if run_all or "D" in exps:
        run_experiment_D(device)

    if run_all or "A" in exps:
        run_experiment_A(device)

    if run_all or "C" in exps:
        run_experiment_C(device)

    print("\n" + "=" * 70)
    print("ALL DONE. Results in eval_results/supplementary_abcd/")
    print("=" * 70)


if __name__ == "__main__":
    main()
