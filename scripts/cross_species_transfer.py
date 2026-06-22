"""
Cross-species zero-shot MIC transfer analysis.

Train a frozen-encoder MIC head on one source species (E. coli),
then evaluate on target species (S. aureus, P. aeruginosa) WITHOUT
any target-species fine-tuning.

Also runs the reverse (train S. aureus → test E. coli) and a
multi-source condition (train all species → test each).

Compares JEPA-AMP vs ESM-2 to test whether JEPA's representations
generalize better across species — a direct test of the "unified
backbone" claim.

Outputs:
    eval_results/cross_species_transfer/
        metrics.json
        SUMMARY.md
        transfer_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "eval_results" / "cross_species_transfer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIES_PAIRS = [
    ("E. coli",       "S. aureus"),
    ("E. coli",       "P. aeruginosa"),
    ("S. aureus",     "E. coli"),
    ("S. aureus",     "P. aeruginosa"),
    ("P. aeruginosa", "E. coli"),
]
SEEDS = [42, 123, 7]


# ── data ──────────────────────────────────────────────────────────────────────

def load_species(csv_path: Path, species: str, max_len: int = 50,
                 seed: int = 42) -> tuple[list, list, list]:
    """Returns (train_recs, val_recs, test_recs) for one species."""
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
    n_val  = max(1, int(n * 0.10))
    test_set = set(unique_seqs[:n_test])
    val_set  = set(unique_seqs[n_test:n_test + n_val])

    train, val, test = [], [], []
    for r in recs:
        if r["seq"] in test_set:
            test.append(r)
        elif r["seq"] in val_set:
            val.append(r)
        else:
            train.append(r)
    return train, val, test


class SeqMICDataset(Dataset):
    def __init__(self, recs: list, max_len: int = 50):
        self.recs = recs
        self.max_len = max_len

    def __len__(self): return len(self.recs)

    def __getitem__(self, idx):
        r = self.recs[idx]
        return r["seq"][:self.max_len], torch.tensor(r["log2_mic"], dtype=torch.float32)


# ── encoder wrappers ──────────────────────────────────────────────────────────

def _batch_encode_jepa(seqs: list[str], device) -> torch.Tensor:
    from src.data.tokenizer import encode, PAD_ID
    encoded = [encode(s[:50]) for s in seqs]
    L = max(len(e) for e in encoded)
    out = torch.full((len(encoded), L), PAD_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out


class JEPAEmbedder(nn.Module):
    def __init__(self, device):
        super().__init__()
        from src.models.jepa import JEPA
        ckpt = torch.load(
            PROJECT_ROOT / "checkpoints/jepa_pretrain_868k/last_jepa.pt",
            map_location=device, weights_only=False,
        )
        jepa = JEPA(**ckpt["cfg"]["model"])
        jepa.load_state_dict(ckpt["model_state"])
        self.enc = jepa.context_encoder
        self.d_model = ckpt["cfg"]["model"]["d_model"]
        for p in self.enc.parameters():
            p.requires_grad_(False)

    def forward(self, seqs: list[str], device) -> torch.Tensor:
        ids = _batch_encode_jepa(seqs, device)
        h = self.enc(ids)
        pad = ids == 0
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
        return h.sum(1) / lengths   # (B, d_model)


class ESM2Embedder(nn.Module):
    def __init__(self, device):
        super().__init__()
        from src.models.esm_head import load_esm2
        self.esm, self.alphabet, _ = load_esm2("esm2_t12_35M")
        self.bc = self.alphabet.get_batch_converter()
        self.d_model = 480
        for p in self.esm.parameters():
            p.requires_grad_(False)

    def forward(self, seqs: list[str], device) -> torch.Tensor:
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = self.bc(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = self.esm(tokens, repr_layers=[12], return_contacts=False)
        h = out["representations"][12]
        pad = tokens == self.alphabet.padding_idx
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        lengths = (~pad).sum(1, keepdim=True).float().clamp(min=1)
        return h.sum(1) / lengths


# ── simple regression head ────────────────────────────────────────────────────

class MICHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x): return self.net(x).squeeze(-1)


# ── train / eval ──────────────────────────────────────────────────────────────

def train_head(embedder, head, train_recs, val_recs, device,
               epochs: int = 60, batch_size: int = 128, lr: float = 3e-4,
               patience: int = 12) -> None:
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    best_val, wait = float("inf"), 0
    best_state = None

    def run_epoch(recs, train: bool):
        random.shuffle(recs)
        losses = []
        for i in range(0, len(recs), batch_size):
            batch = recs[i:i + batch_size]
            seqs = [r["seq"] for r in batch]
            y = torch.tensor([r["log2_mic"] for r in batch],
                              dtype=torch.float32, device=device)
            with torch.set_grad_enabled(train):
                with torch.no_grad():
                    emb = embedder(seqs, device)
                pred = head(emb)
                loss = F.huber_loss(pred, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    head.train()
    for ep in range(epochs):
        run_epoch(train_recs, train=True)
        head.eval()
        val_loss = run_epoch(val_recs, train=False)
        head.train()
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        head.load_state_dict(best_state)


def eval_head(embedder, head, test_recs, device) -> dict:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error

    head.eval()
    preds, trues = [], []
    for i in range(0, len(test_recs), 256):
        batch = test_recs[i:i + 256]
        seqs = [r["seq"] for r in batch]
        y = [r["log2_mic"] for r in batch]
        with torch.no_grad():
            emb = embedder(seqs, device)
            pred = head(emb).cpu().numpy()
        preds.extend(pred.tolist())
        trues.extend(y)

    preds, trues = np.array(preds), np.array(trues)
    r, _   = pearsonr(preds, trues)
    rho, _ = spearmanr(preds, trues)
    rmse   = float(np.sqrt(mean_squared_error(trues, preds)))
    return {"pearson": float(r), "spearman": float(rho), "rmse": rmse,
            "n": len(trues)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grampa = PROJECT_ROOT / "data" / "grampa.csv"

    results: dict = {}
    existing = OUT_DIR / "metrics.json"
    if existing.exists():
        results = json.loads(existing.read_text())

    for model_name, EmbClass in [("jepa", JEPAEmbedder), ("esm2", ESM2Embedder)]:
        print(f"\n{'='*60}\n{model_name.upper()}\n{'='*60}")
        embedder = EmbClass(device).to(device).eval()
        d = embedder.d_model

        model_res = results.setdefault(model_name, {})

        for src_species, tgt_species in SPECIES_PAIRS:
            pair_key = f"{src_species}→{tgt_species}"
            pair_res = model_res.setdefault(pair_key, {})

            for seed in SEEDS:
                seed_key = str(seed)
                if seed_key in pair_res:
                    print(f"  [skip] {pair_key} seed={seed}")
                    continue

                print(f"\n  {pair_key}  seed={seed}")
                src_tr, src_val, _ = load_species(grampa, src_species, seed=seed)
                _,      _,   tgt_te = load_species(grampa, tgt_species, seed=seed)

                print(f"    src train={len(src_tr)}  tgt test={len(tgt_te)}")

                head = MICHead(d).to(device)
                train_head(embedder, head, src_tr, src_val, device)

                # in-domain test (same species, different split)
                _, _, src_te = load_species(grampa, src_species, seed=seed)
                in_domain  = eval_head(embedder, head, src_te,  device)
                zero_shot  = eval_head(embedder, head, tgt_te,  device)

                pair_res[seed_key] = {
                    "in_domain": in_domain,
                    "zero_shot": zero_shot,
                }
                print(f"    in-domain  Pearson={in_domain['pearson']:.3f}")
                print(f"    zero-shot  Pearson={zero_shot['pearson']:.3f}")
                existing.write_text(json.dumps(results, indent=2))

    _write_summary(results)
    _plot_heatmap(results)
    print(f"\nDone. Results in {OUT_DIR}")


def _agg(pair_res: dict, key: str, metric: str) -> tuple[float, float]:
    vals = [v[key][metric] for v in pair_res.values()
            if v and key in v and v[key].get(metric) is not None]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _write_summary(results: dict) -> None:
    lines = [
        "# Cross-Species MIC Transfer: JEPA-AMP vs ESM-2",
        "",
        "Frozen encoder trained on source species, evaluated on target (zero-shot).",
        "3-seed mean ± std.",
        "",
        "| Source → Target | JEPA in-domain | JEPA zero-shot | ESM-2 in-domain | ESM-2 zero-shot |",
        "|---|---|---|---|---|",
    ]
    for src, tgt in SPECIES_PAIRS:
        key = f"{src}→{tgt}"
        row = [key]
        for model in ("jepa", "esm2"):
            pair_res = results.get(model, {}).get(key, {})
            for split in ("in_domain", "zero_shot"):
                m, s = _agg(pair_res, split, "pearson")
                row.append(f"{m:.3f} ± {s:.3f}" if not np.isnan(m) else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Transfer gap (in-domain minus zero-shot Pearson)",
        "",
        "Smaller gap = better transfer. If JEPA gap < ESM-2 gap, JEPA generalizes better.",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT_DIR / 'SUMMARY.md'}")


def _plot_heatmap(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    species = ["E. coli", "S. aureus", "P. aeruginosa"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, model in zip(axes, ("jepa", "esm2")):
        mat = np.full((len(species), len(species)), np.nan)
        for i, src in enumerate(species):
            for j, tgt in enumerate(species):
                if src == tgt:
                    mat[i, j] = np.nan
                    continue
                key = f"{src}→{tgt}"
                pair_res = results.get(model, {}).get(key, {})
                m, _ = _agg(pair_res, "zero_shot", "pearson")
                mat[i, j] = m

        im = ax.imshow(mat, vmin=0, vmax=0.7, cmap="Blues")
        ax.set_xticks(range(len(species))); ax.set_xticklabels(
            [s.replace(". ", ".\n") for s in species], fontsize=8)
        ax.set_yticks(range(len(species))); ax.set_yticklabels(
            [s.replace(". ", ".\n") for s in species], fontsize=8)
        ax.set_xlabel("Target species"); ax.set_ylabel("Source species")
        ax.set_title(f"{model.upper()} zero-shot Pearson")
        for i in range(len(species)):
            for j in range(len(species)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=9, color="white" if mat[i,j] > 0.4 else "black")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "transfer_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'transfer_heatmap.png'}")


if __name__ == "__main__":
    main()
