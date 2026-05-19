"""
Transformer encoder for AMP sequences.
Used as both context encoder (f_theta) and target encoder (f_xi, EMA copy).
"""

import math
import torch
import torch.nn as nn

from src.data.tokenizer import VOCAB_SIZE, PAD_ID


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 52,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm, more stable for small data
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) long tensor
        Returns:
            h: (B, L, d_model) hidden states
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        # padding mask: True = ignore
        pad_mask = (input_ids == self.pad_id)  # (B, L)
        h = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.norm(h)  # (B, L, d_model)
