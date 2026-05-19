"""
Supervised heads for JEPA encoder fine-tuning.

MLPHead and TransformerHead, composed into:
  - JEPAClassifier    (AMP binary + optional toxicity dual-head)
  - JEPAMICPredictor  (MIC regression with bacteria conditioning)
    - MLP variant:  FiLM modulates all token representations before pooling
    - Transformer:  bacteria token prepended, TransformerHead attends over it
"""

import torch
import torch.nn as nn

from src.data.tokenizer import PAD_ID
from src.models.generator import Adapter


# ---------------------------------------------------------------------------
# Pooling heads
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """Mean-pool → LayerNorm → 2-layer MLP."""

    def __init__(self, d_model: int, hidden: int, n_out: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, h: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """h: (B, L, D)  →  (B, n_out)"""
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = h.mean(1)
        return self.net(self.norm(pooled))


class TransformerHead(nn.Module):
    """
    Prepend [CLS] token → small TransformerEncoder → CLS output → Linear.
    Bacteria/condition tokens can be passed in as prefix via h directly.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        n_out: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, h: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """h: (B, L, D)  →  (B, n_out)"""
        B = h.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, h], dim=1)
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.head(self.norm(out[:, 0]))


# ---------------------------------------------------------------------------
# AMP + Toxicity classifier
# ---------------------------------------------------------------------------

class JEPAClassifier(nn.Module):
    """
    JEPA encoder (frozen) + Adapter + dual-head for AMP and toxicity classification.

    head_type : "mlp" or "transformer"
    Setting n_tox=0 disables the toxicity head.
    """

    def __init__(
        self,
        encoder,
        d_model: int,
        head_type: str = "mlp",
        hidden: int = 256,
        dropout: float = 0.4,
        adapter_bottleneck: int = 64,
        freeze_encoder: bool = True,
        n_tox: int = 1,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
    ):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self._encoder_needs_grad = any(p.requires_grad for p in self.encoder.parameters())

        self.adapter = Adapter(d_model, bottleneck=adapter_bottleneck)

        def _make_head():
            if head_type == "mlp":
                return MLPHead(d_model, hidden, 1, dropout=dropout)
            return TransformerHead(d_model, nhead, num_layers, dim_feedforward, 1, dropout)

        self.head_amp = _make_head()
        self.head_tox = _make_head() if n_tox > 0 else None

    def _encode(self, input_ids: torch.Tensor):
        padding_mask = (input_ids == PAD_ID)
        ctx = torch.enable_grad() if self._encoder_needs_grad else torch.no_grad()
        with ctx:
            h = self.encoder(input_ids)
        return self.adapter(h), padding_mask

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        h, mask = self._encode(input_ids)
        out = {"amp_logit": self.head_amp(h, mask).squeeze(-1)}
        if self.head_tox is not None:
            out["tox_logit"] = self.head_tox(h, mask).squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# MIC predictor with bacteria conditioning
# ---------------------------------------------------------------------------

class JEPAMICPredictor(nn.Module):
    """
    JEPA encoder (frozen) + Adapter + bacteria conditioning + head for MIC regression.

    MLP variant:
      FiLM modulates all token representations with bacteria (γ, β) before pooling.
      More parameter-efficient, better generalizes to unseen bacteria.

    Transformer variant:
      Bacteria projected to d_model and prepended as an extra token before the head.
      Attention decides how much to use the bacteria signal.

    head_type : "mlp" or "transformer"
    """

    def __init__(
        self,
        encoder,
        d_model: int,
        n_bacteria: int,
        bacteria_dim: int = 64,
        head_type: str = "mlp",
        hidden: int = 256,
        dropout: float = 0.4,
        adapter_bottleneck: int = 64,
        freeze_encoder: bool = True,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
    ):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self._encoder_needs_grad = any(p.requires_grad for p in self.encoder.parameters())
        self._head_type = head_type

        self.adapter = Adapter(d_model, bottleneck=adapter_bottleneck)
        self.bacteria_emb = nn.Embedding(n_bacteria, bacteria_dim)

        if head_type == "mlp":
            # FiLM: zero-init so training starts from unmodulated baseline
            self.film = nn.Linear(bacteria_dim, 2 * d_model)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.head = MLPHead(d_model, hidden, 1, dropout=dropout)
        else:
            self.bact_proj = nn.Linear(bacteria_dim, d_model)
            self.head = TransformerHead(d_model, nhead, num_layers, dim_feedforward, 1, dropout)

    def forward(self, input_ids: torch.Tensor, bacteria_idx: torch.Tensor) -> torch.Tensor:
        """Returns log2(MIC) predictions (B,)."""
        padding_mask = (input_ids == PAD_ID)
        ctx = torch.enable_grad() if self._encoder_needs_grad else torch.no_grad()
        with ctx:
            h = self.encoder(input_ids)
        h = self.adapter(h)  # (B, L, D)
        bact = self.bacteria_emb(bacteria_idx)  # (B, bacteria_dim)

        if self._head_type == "mlp":
            gb = self.film(bact)  # (B, 2D)
            gamma, beta = gb.chunk(2, dim=-1)  # (B, D) each
            h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            return self.head(h, padding_mask).squeeze(-1)
        else:
            bact_tok = self.bact_proj(bact).unsqueeze(1)  # (B, 1, D)
            h_ext = torch.cat([bact_tok, h], dim=1)
            cls_mask = torch.zeros(input_ids.shape[0], 1, dtype=torch.bool, device=input_ids.device)
            mask_ext = torch.cat([cls_mask, padding_mask], dim=1)
            return self.head(h_ext, mask_ext).squeeze(-1)

    @torch.no_grad()
    def mc_predict(self, input_ids: torch.Tensor, bacteria_idx: torch.Tensor,
                   n_samples: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
        """MC-Dropout inference: returns (mean, std) both shape (B,).

        Encoder stays in eval mode (deterministic representations).
        Adapter + head stay in train mode (stochastic via dropout).
        """
        self.encoder.eval()
        # enable dropout only in head layers
        self.adapter.train()
        self.bacteria_emb.train()
        if self._head_type == "mlp":
            self.film.train()
            self.head.train()
        else:
            self.bact_proj.train()
            self.head.train()

        samples = []
        for _ in range(n_samples):
            samples.append(self.forward(input_ids, bacteria_idx))
        stacked = torch.stack(samples, dim=0)  # (n_samples, B)
        return stacked.mean(0), stacked.std(0)
