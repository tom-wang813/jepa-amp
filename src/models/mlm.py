"""
MLM pre-training model — ablation baseline for JEPA.

Same TransformerEncoder backbone (d_model=384, 8 layers) as JEPA.
Uses identical block masking strategy so the only difference is the
prediction target: token identity (cross-entropy) vs. latent representation (MSE).

Ablation question: does predicting in latent space (JEPA) yield representations
with higher fine-tuning plasticity than predicting discrete tokens (MLM)?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import TransformerEncoder
from src.data.tokenizer import MASK_ID, VOCAB_SIZE


class MLMModel(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_seq_len: int = 52,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        # Projection head: only used during pre-training, discarded for downstream tasks
        self.mlm_head = nn.Linear(d_model, VOCAB_SIZE, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,    # (B, L)
        target_mask: torch.Tensor,  # (B, L) bool: True = masked position to predict
    ) -> dict[str, torch.Tensor]:
        # Replace target positions with MASK token — identical to JEPA's context encoder input
        masked_ids = input_ids.clone()
        masked_ids[target_mask] = MASK_ID

        h = self.encoder(masked_ids)         # (B, L, D)
        logits = self.mlm_head(h)            # (B, L, vocab_size)

        # Cross-entropy only at masked positions
        pred_logits = logits[target_mask]    # (N, vocab_size)
        true_tokens = input_ids[target_mask] # (N,)
        loss = F.cross_entropy(pred_logits, true_tokens)

        return {"loss": loss, "h": h}
