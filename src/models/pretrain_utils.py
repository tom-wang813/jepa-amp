"""
Utility for loading a pre-trained encoder from either a JEPA or MLM checkpoint.

All fine-tuning scripts should call load_pretrained_encoder() instead of
instantiating JEPA directly so that MLM ablation checkpoints are handled
transparently.
"""

from __future__ import annotations

import torch
from src.models.encoder import TransformerEncoder


def load_pretrained_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[TransformerEncoder, dict]:
    """
    Load the pre-trained encoder from a JEPA or MLM checkpoint.

    Returns:
        encoder  – TransformerEncoder with loaded weights
        cfg      – full config dict saved alongside the checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrain_type = ckpt.get("pretrain_type", "jepa")
    epoch     = ckpt.get("epoch", "?")
    val_loss  = ckpt.get("val_loss", float("nan"))

    if pretrain_type == "mlm":
        from src.models.mlm import MLMModel
        # MLM config only contains encoder keys; drop any stray JEPA-only fields
        _jepa_only = {"predictor_depth", "ema_decay"}
        model_cfg = {k: v for k, v in ckpt["cfg"]["model"].items()
                     if k not in _jepa_only}
        model = MLMModel(**model_cfg)
        model.load_state_dict(ckpt["model_state"])
        encoder = model.encoder
    else:
        from src.models.jepa import JEPA
        jepa = JEPA(**ckpt["cfg"]["model"])
        jepa.load_state_dict(ckpt["model_state"])
        encoder = jepa.context_encoder

    print(f"Loaded {pretrain_type.upper()} encoder  epoch={epoch}  val_loss={val_loss:.4f}")
    return encoder, ckpt["cfg"]
