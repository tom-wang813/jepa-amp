"""
Standalone representation evaluation for JEPA encoder quality.

Usage:
    uv run python -m src.eval.rep_eval
"""

from pathlib import Path
import json
import logging

import torch

from src.eval.run_eval import _load_yaml, evaluate_representation_backbones
from src.data.dataset import build_seq2seq_datasets
from src.models.jepa import JEPA

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAIN_CKPT = PROJECT_ROOT / "checkpoints" / "jepa_pretrain" / "best_jepa.pt"
PRETRAIN_CFG = PROJECT_ROOT / "configs" / "jepa_pretrain.yaml"
FINETUNE_CFG = PROJECT_ROOT / "configs" / "finetune.yaml"
EVAL_DIR = PROJECT_ROOT / "eval_results"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    pretrain_cfg = _load_yaml(PRETRAIN_CFG)
    finetune_cfg = _load_yaml(FINETUNE_CFG)

    train_ds, val_ds = build_seq2seq_datasets(
        fasta_paths=[PROJECT_ROOT / p for p in pretrain_cfg["data"]["fasta_paths"]],
        max_len=pretrain_cfg["data"]["max_len"],
        val_ratio=pretrain_cfg["data"]["val_ratio"],
        seed=42,
        prefix_ratio=finetune_cfg["data"].get("prefix_ratio", 0.5),
        min_prefix_len=finetune_cfg["data"].get("min_prefix_len", 3),
        max_seq_len=finetune_cfg["generator"]["max_seq_len"],
    )

    jepa = JEPA(**pretrain_cfg["model"])
    ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    jepa.load_state_dict(ckpt["model_state"])
    jepa.to(device).eval()

    metrics = evaluate_representation_backbones(
        train_sequences=train_ds.sequences,
        val_sequences=val_ds.sequences,
        device=device,
        jepa_encoder=jepa.context_encoder,
        encoder_cfg=pretrain_cfg["model"],
    )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "rep_eval.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved representation eval to: {out_path}")


if __name__ == "__main__":
    main()
