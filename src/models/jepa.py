"""
JEPA pre-training model for AMP sequences.

Architecture:
  - Context encoder f_theta: encodes visible (context) tokens
  - Target encoder f_xi: EMA copy of f_theta, encodes target tokens (stop-grad)
  - Predictor g_phi: predicts target representations from context representations

Training objective:
  L = mean over target positions of ||g_phi(h_c)[pos] - sg(f_xi(x)[pos])||^2
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import TransformerEncoder
from src.data.tokenizer import MASK_ID


class Predictor(nn.Module):
    """
    Small Transformer that takes context embeddings and predicts target embeddings.
    Uses learnable mask tokens for target positions.
    """

    def __init__(self, d_model: int = 256, predictor_depth: int = 2, max_seq_len: int = 52):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=predictor_depth)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        context_h: torch.Tensor,      # (B, L, d_model)
        context_mask: torch.Tensor,    # (B, L) bool: True = context position
        target_mask: torch.Tensor,     # (B, L) bool: True = target position
    ) -> torch.Tensor:
        """
        Returns predicted embeddings at target positions: (B, L, d_model)
        Only positions where target_mask=True are meaningful.
        """
        B, L, D = context_h.shape
        positions = torch.arange(L, device=context_h.device).unsqueeze(0)  # (1, L)
        pos_emb = self.pos_emb(positions)  # (1, L, D)

        # Build input: context positions keep their embeddings; target positions get mask token
        x = context_h + pos_emb
        mask_tokens = self.mask_token.expand(B, L, D) + pos_emb
        target_mask_3d = target_mask.unsqueeze(-1).float()
        x = x * (1 - target_mask_3d) + mask_tokens * target_mask_3d

        h = self.transformer(x)
        h = self.norm(h)
        return self.proj(h)  # (B, L, d_model)


class JEPA(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 52,
        predictor_depth: int = 2,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        encoder_kwargs = dict(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, max_seq_len=max_seq_len,
        )
        # context encoder: updated by gradient
        self.context_encoder = TransformerEncoder(**encoder_kwargs)

        # target encoder: EMA copy, no gradient
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(d_model=d_model, predictor_depth=predictor_depth, max_seq_len=max_seq_len)
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update: xi <- tau * xi + (1 - tau) * theta"""
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(self.ema_decay).add_(p_ctx.data, alpha=1.0 - self.ema_decay)

    def forward(
        self,
        input_ids: torch.Tensor,      # (B, L)
        context_mask: torch.Tensor,   # (B, L) bool
        target_mask: torch.Tensor,    # (B, L) bool
    ) -> dict[str, torch.Tensor]:
        # --- context encoder: target positions replaced with MASK token before encoding
        # This is the core JEPA requirement: encoder must NOT see the tokens it needs to predict
        masked_ids = input_ids.clone()
        masked_ids[target_mask] = MASK_ID
        context_h = self.context_encoder(masked_ids)  # (B, L, D)

        # --- target encoder (stop-gradient)
        with torch.no_grad():
            target_h = self.target_encoder(input_ids)  # (B, L, D)

        # --- predictor: predict target representations from context
        pred_h = self.predictor(context_h, context_mask, target_mask)  # (B, L, D)

        # --- loss: MSE at target positions only
        # target_mask: (B, L) bool
        pred_at_targets   = pred_h[target_mask]    # (N, D)
        target_at_targets = target_h[target_mask]  # (N, D)

        # symmetric layer norm on both sides (prevents collapse, stabilises scale)
        pred_at_targets   = F.layer_norm(pred_at_targets,   pred_at_targets.shape[-1:])
        target_at_targets = F.layer_norm(target_at_targets, target_at_targets.shape[-1:])

        loss = F.mse_loss(pred_at_targets, target_at_targets)

        return {"loss": loss, "context_h": context_h, "target_h": target_h}
