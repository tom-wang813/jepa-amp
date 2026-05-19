"""
ESM-2 based models for AMP classification and MIC prediction.
Uses fair-esm (Facebook Research) which is compatible with PyTorch 2.3.
"""

import torch
import torch.nn as nn

from src.models.supervised_head import MLPHead, TransformerHead


_ESM2_MODELS = {
    "esm2_t6_8M":    "esm2_t6_8M_UR50D",
    "esm2_t12_35M":  "esm2_t12_35M_UR50D",
    "esm2_t30_150M": "esm2_t30_150M_UR50D",
    "esm2_t33_650M": "esm2_t33_650M_UR50D",
}


def load_esm2(model_key: str):
    """Load ESM-2 model and alphabet. Returns (model, alphabet, d_model)."""
    import esm as esm_lib
    fn_name = _ESM2_MODELS.get(model_key, model_key)
    fn = getattr(esm_lib.pretrained, fn_name)
    model, alphabet = fn()
    return model, alphabet, model.embed_dim


class ESMEncoder(nn.Module):
    """Wrap fair-esm ESM-2 to return last-layer token representations."""

    def __init__(self, model_key: str = "esm2_t12_35M", freeze: bool = False):
        super().__init__()
        self.esm, self.alphabet, d = load_esm2(model_key)
        self.d_model     = d
        self.num_layers  = self.esm.num_layers
        self.padding_idx = self.alphabet.padding_idx
        if freeze:
            for p in self.esm.parameters():
                p.requires_grad_(False)
        self._needs_grad = not freeze

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: (B, L) ESM token ids (include BOS at 0, EOS at end, PAD after)
        Returns:
          h:            (B, L, d_model) — full sequence including BOS/EOS
          padding_mask: (B, L) bool — True where token is PAD
        """
        padding_mask = tokens.eq(self.padding_idx)
        ctx = torch.enable_grad() if self._needs_grad else torch.no_grad()
        with ctx:
            out = self.esm(tokens, repr_layers=[self.num_layers],
                           return_contacts=False)
        h = out["representations"][self.num_layers]  # (B, L, D)
        return h, padding_mask


class ESMClassifier(nn.Module):
    """ESM-2 encoder + MLP or Transformer head for AMP binary classification."""

    def __init__(
        self,
        model_key: str = "esm2_t12_35M",
        head_type: str = "mlp",
        hidden: int = 512,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
    ):
        super().__init__()
        self.encoder = ESMEncoder(model_key, freeze=freeze_encoder)
        d = self.encoder.d_model

        if head_type == "mlp":
            self.head = MLPHead(d, hidden, 1, dropout=dropout)
        else:
            self.head = TransformerHead(d, nhead, num_layers, dim_feedforward, 1, dropout)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        h, padding_mask = self.encoder(tokens)
        return {"amp_logit": self.head(h, padding_mask).squeeze(-1)}


class ESMMICPredictor(nn.Module):
    """ESM-2 encoder + FiLM/Transformer head for MIC regression."""

    def __init__(
        self,
        model_key: str = "esm2_t12_35M",
        n_bacteria: int = 20,
        bacteria_dim: int = 64,
        head_type: str = "mlp",
        hidden: int = 256,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
    ):
        super().__init__()
        self.encoder = ESMEncoder(model_key, freeze=freeze_encoder)
        d = self.encoder.d_model
        self._head_type = head_type

        self.bacteria_emb = nn.Embedding(n_bacteria, bacteria_dim)

        if head_type == "mlp":
            self.film = nn.Linear(bacteria_dim, 2 * d)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.head = MLPHead(d, hidden, 1, dropout=dropout)
        else:
            self.bact_proj = nn.Linear(bacteria_dim, d)
            self.head = TransformerHead(d, nhead, num_layers, dim_feedforward, 1, dropout)

    @torch.no_grad()
    def mc_predict(self, tokens: torch.Tensor, bacteria_idx: torch.Tensor,
                   n_samples: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
        """MC-Dropout inference: returns (mean, std) both shape (B,)."""
        self.encoder.esm.eval()   # encoder deterministic
        self.bacteria_emb.train()
        if self._head_type == "mlp":
            self.film.train(); self.head.train()
        else:
            self.bact_proj.train(); self.head.train()

        samples = [self.forward(tokens, bacteria_idx) for _ in range(n_samples)]
        stacked = torch.stack(samples, dim=0)
        return stacked.mean(0), stacked.std(0)

    def forward(self, tokens: torch.Tensor,
                bacteria_idx: torch.Tensor) -> torch.Tensor:
        h, padding_mask = self.encoder(tokens)
        bact = self.bacteria_emb(bacteria_idx)

        if self._head_type == "mlp":
            gamma, beta = self.film(bact).chunk(2, dim=-1)
            h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            return self.head(h, padding_mask).squeeze(-1)
        else:
            bact_tok = self.bact_proj(bact).unsqueeze(1)
            h_ext = torch.cat([bact_tok, h], dim=1)
            if padding_mask is not None:
                cls_mask = torch.zeros(tokens.shape[0], 1, dtype=torch.bool,
                                       device=tokens.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
            return self.head(h_ext, padding_mask).squeeze(-1)
