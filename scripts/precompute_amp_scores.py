"""
Precompute JEPA AMP classifier scores for all sequences in the training corpus.

Produces: data/processed/amp_scores_cache.json  {sequence: float P(AMP)}

Usage:
  uv run python scripts/precompute_amp_scores.py --gpu 0
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_classifier(device: torch.device):
    """Load the AMPlify-identical JEPA AMP classifier."""
    import yaml
    from src.models.jepa import JEPA
    from src.models.supervised_head import JEPAClassifier

    cfg_path  = PROJECT_ROOT / "configs/amp_classifier_amplify_identical.yaml"
    ckpt_path = PROJECT_ROOT / "checkpoints/amp_classifier_amplify_identical/best_model.pt"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    pt_ckpt = torch.load(PROJECT_ROOT / cfg["pretrain_checkpoint"],
                         map_location=device, weights_only=False)
    jepa = JEPA(**pt_ckpt["cfg"]["model"])
    jepa.load_state_dict(pt_ckpt["model_state"])

    head_cfg  = cfg["head"].copy()
    head_cfg.pop("head_type", None)
    model = JEPAClassifier(
        encoder=jepa.context_encoder,
        d_model=pt_ckpt["cfg"]["model"]["d_model"],
        freeze_encoder=True,
        **head_cfg,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # strict=False: checkpoint may include head_tox we don't need
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"  Missing keys (ignored): {missing}")
    model.eval()
    print(f"Loaded AMP classifier from {ckpt_path}")
    return model


@torch.no_grad()
def score_sequences(model, sequences: list[str], device: torch.device,
                    batch_size: int = 512) -> dict[str, float]:
    from src.data.tokenizer import encode, PAD_ID

    scores: dict[str, float] = {}
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        encoded = [encode(s[:50]) for s in batch]
        max_len = max(len(e) for e in encoded)
        ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long, device=device)
        for j, e in enumerate(encoded):
            ids[j, :len(e)] = torch.tensor(e, dtype=torch.long)

        out = model(ids)   # dict with 'amp_logit'
        logits = out["amp_logit"]   # (B,)
        probs = torch.sigmoid(logits).cpu().tolist()

        for seq, p in zip(batch, probs):
            scores[seq] = float(p)

        if (i // batch_size) % 20 == 0:
            print(f"  {i+len(batch):,} / {len(sequences):,} scored …")

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fasta",
                        default=str(PROJECT_ROOT / "data/processed/amp_corpus.fasta"))
    parser.add_argument("--out",
                        default=str(PROJECT_ROOT / "data/processed/amp_scores_cache.json"))
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load sequences
    seqs = []
    with open(args.fasta) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                if all(c in AA for c in line):
                    seqs.append(line)
    seqs = list(set(seqs))
    print(f"Sequences to score: {len(seqs):,}")

    # Load model and score
    model = load_classifier(device)
    print(f"Scoring {len(seqs):,} sequences (batch={args.batch_size}) …")
    scores = score_sequences(model, seqs, device, batch_size=args.batch_size)

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(scores, f)
    print(f"Saved {len(scores):,} AMP scores → {out}")

    # Quick stats
    vals = list(scores.values())
    import statistics
    print(f"Score stats: mean={statistics.mean(vals):.3f}  "
          f"median={statistics.median(vals):.3f}  "
          f"P(AMP>0.5)={sum(v>0.5 for v in vals)/len(vals):.1%}")


if __name__ == "__main__":
    main()
