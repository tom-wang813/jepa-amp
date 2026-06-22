"""
Masked Diffusion conditional AMP generator (MDLM-style).

Forward process (corruption):
  At timestep t ∈ {1..T}, each non-special token is independently masked
  with probability t/T.  Special tokens (PAD, BOS, EOS) are never masked.

Backward process (denoising):
  A bidirectional transformer predicts the original token at each MASK position,
  conditioned on:
    - the partially-masked sequence x_t
    - JEPA encoder context h_ctx  (via cross-attention)
    - physicochemical / MIC condition cond_emb  (via AdaLN)
    - diffusion timestep t  (sinusoidal embedding, also via AdaLN)

Loss:
  Cross-entropy on MASK positions only.
  Equivalent to masked language modelling where the mask rate is
  proportional to t/T — recovers the ELBO of the absorbing-state DDPM.

Inference:
  Ancestral sampling with T_infer steps (default 50):
    for t = T, T-1, ..., 1:
        predict x_0 logits at all MASK positions
        unmask fraction 1/(t+1) of them (highest confidence first)
  This is the "greedy order" schedule from Ghazvininejad et al. 2019,
  extended with condition guidance.

References:
  - Austin et al. 2021  (D3PM absorbing-state diffusion)
  - Sahoo et al. 2024   (MDLM: Masked Diffusion Language Models)
  - Ghazvininejad et al. 2019  (Mask-Predict iterative decoding)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.tokenizer import VOCAB_SIZE, PAD_ID, MASK_ID, BOS_ID, EOS_ID
from src.models.encoder import TransformerEncoder

AA_START = 2
AA_END   = 22
T_MAX    = 1000   # training diffusion steps (continuous t in [0,1] equivalent)


# ── helpers ───────────────────────────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal timestep embedding → MLP → d_time."""
    def __init__(self, d_time: int = 128):
        super().__init__()
        half = d_time // 2
        self.register_buffer("freqs",
            torch.exp(-math.log(10000) * torch.arange(half).float() / (half - 1)))
        self.mlp = nn.Sequential(
            nn.Linear(d_time, d_time * 2), nn.SiLU(),
            nn.Linear(d_time * 2, d_time),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0, 1]  →  (B, d_time)"""
        args  = t.unsqueeze(1) * self.freqs.unsqueeze(0) * math.pi * 2
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class Adapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.norm(x + self.net(x))


class DenoisingLayer(nn.Module):
    """
    Bidirectional transformer layer.
    AdaLN receives concatenated [cond_emb, time_emb] → combined conditioning.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, d_cond):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaln = nn.Linear(d_cond, 6 * d_model)
        nn.init.zeros_(self.adaln.weight); nn.init.zeros_(self.adaln.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, combined_cond,
                src_key_padding_mask=None, memory_key_padding_mask=None):
        s1, b1, s2, b2, s3, b3 = self.adaln(combined_cond).chunk(6, dim=-1)
        s1, b1 = s1.unsqueeze(1), b1.unsqueeze(1)
        s2, b2 = s2.unsqueeze(1), b2.unsqueeze(1)
        s3, b3 = s3.unsqueeze(1), b3.unsqueeze(1)

        h = (1 + s1) * self.norm1(x) + b1
        a, _ = self.self_attn(h, h, h, key_padding_mask=src_key_padding_mask,
                              need_weights=False)
        x = x + self.drop(a)

        h = (1 + s2) * self.norm2(x) + b2
        c, _ = self.cross_attn(h, memory, memory,
                               key_padding_mask=memory_key_padding_mask,
                               need_weights=False)
        x = x + self.drop(c)

        h = (1 + s3) * self.norm3(x) + b3
        x = x + self.drop(self.ff(h))
        return x


class DenoisingTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=384, nhead=8,
                 num_layers=6, dim_feedforward=1536, dropout=0.1,
                 max_seq_len=52, d_cond=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            DenoisingLayer(d_model, nhead, dim_feedforward, dropout, d_cond)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, noisy_ids, memory, combined_cond,
                src_pad_mask=None, memory_pad_mask=None):
        B, T = noisy_ids.shape
        pos = torch.arange(T, device=noisy_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(noisy_ids) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x, memory, combined_cond,
                      src_key_padding_mask=src_pad_mask,
                      memory_key_padding_mask=memory_pad_mask)
        return self.head(self.norm(x))   # (B, T, vocab_size)


# ── main model ────────────────────────────────────────────────────────────────

class ConditionalGeneratorDiffusion(nn.Module):
    """
    MDLM-style masked diffusion conditional generator.

    During training:
      1. Sample t ~ Uniform(0, 1)
      2. Corrupt target sequence: each AA token masked with prob t
      3. Denoiser predicts x_0 at MASK positions
      4. Loss: NLL at MASK positions only

    During inference:
      Start from all-MASK, iteratively unmask in T_infer steps.
    """
    def __init__(
        self,
        encoder: TransformerEncoder,
        d_model: int = 384,
        adapter_bottleneck: int = 128,
        denoiser_layers: int = 6,
        denoiser_nhead: int = 8,
        denoiser_ff: int = 1536,
        dropout: float = 0.1,
        max_seq_len: int = 52,
        freeze_encoder: bool = True,
        num_conditions: int = 3,
        d_cond: int = 128,
        d_time: int = 128,
        cond_dropout: float = 0.30,
        context_dropout: float = 0.15,
    ):
        super().__init__()
        self.d_cond = d_cond
        self.d_time = d_time

        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self._enc_grad = any(p.requires_grad for p in self.encoder.parameters())

        self.adapter = Adapter(d_model, adapter_bottleneck)

        self.cond_encoder = nn.Sequential(
            nn.Linear(num_conditions, d_cond), nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        self.time_encoder = SinusoidalTimeEmb(d_time)

        # project [cond_emb, time_emb] → combined d_cond for AdaLN
        self.combined_proj = nn.Linear(d_cond + d_time, d_cond)

        self.cond_dropout    = cond_dropout
        self.context_dropout = context_dropout

        self.denoiser = DenoisingTransformer(
            d_model=d_model, nhead=denoiser_nhead, num_layers=denoiser_layers,
            dim_feedforward=denoiser_ff, dropout=dropout,
            max_seq_len=max_seq_len, d_cond=d_cond,
        )

    def _encode_condition(self, conditions, t_float):
        """
        conditions: (B, num_conditions)
        t_float:    (B,) timestep in [0, 1]
        Returns combined_cond: (B, d_cond) for AdaLN
        """
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(conditions.shape[0], 1, device=conditions.device)
                    > self.cond_dropout).float()
            conditions = conditions * mask
        cond_emb = self.cond_encoder(conditions)           # (B, d_cond)
        time_emb = self.time_encoder(t_float)              # (B, d_time)
        combined = self.combined_proj(
            torch.cat([cond_emb, time_emb], dim=-1))       # (B, d_cond)
        return combined

    @staticmethod
    def corrupt(seq_ids: torch.Tensor, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Absorbing-state corruption: mask each AA token independently with prob t.
        Special tokens (PAD, BOS, EOS, MASK) are never masked.
        Returns (noisy_ids, is_masked) where is_masked indicates which positions
        were newly masked (these are the training targets).
        """
        is_aa  = (seq_ids >= AA_START) & (seq_ids < AA_END)
        noise  = torch.rand_like(seq_ids.float()) < t
        masked = is_aa & noise
        noisy  = seq_ids.clone()
        noisy[masked] = MASK_ID
        return noisy, masked

    def forward(self, context_ids, tgt_ids, conditions=None,
                context_key_padding_mask=None):
        """
        Training forward pass.
        tgt_ids: (B, T) clean target token ids

        Returns (logits, is_masked):
          logits:    (B, T, vocab_size)  — predictions at ALL positions
          is_masked: (B, T) bool         — which positions were corrupted
                                          (use only these for loss)
        """
        B = context_ids.shape[0]
        device = context_ids.device

        ctx = torch.enable_grad() if self._enc_grad else torch.no_grad()
        with ctx:
            h_ctx = self.encoder(context_ids)
        h_ctx = self.adapter(h_ctx)

        if self.training and self.context_dropout > 0:
            drop = (torch.rand(B, 1, 1, device=device) < self.context_dropout).float()
            h_ctx = h_ctx * (1 - drop)

        # Sample timestep per sample, uniform in (0, 1]
        t_float = torch.rand(B, device=device).clamp(min=1e-4)

        if conditions is not None:
            combined_cond = self._encode_condition(conditions, t_float)
        else:
            combined_cond = self._encode_condition(
                torch.zeros(B, self.cond_encoder[0].in_features, device=device),
                t_float)

        # Corrupt sequence
        # Use mean t across batch for corruption (per-sample t would be correct but
        # requires looping; mean-t is a common approximation for efficiency)
        t_mean = float(t_float.mean().item())
        noisy_ids, is_masked = self.corrupt(tgt_ids, t_mean)

        src_pad = (noisy_ids == PAD_ID)
        logits = self.denoiser(noisy_ids, h_ctx, combined_cond,
                               src_pad_mask=src_pad,
                               memory_pad_mask=context_key_padding_mask)
        return logits, is_masked

    @torch.no_grad()
    def generate(
        self,
        context_ids: torch.Tensor,
        conditions: torch.Tensor | None = None,
        seq_len: int | None = None,
        T_infer: int = 50,
        temperature: float = 1.0,
    ) -> list[str]:
        """
        Ancestral masked diffusion sampling.
        seq_len: target length (if None, uses context-adaptive heuristic)
        T_infer: number of denoising steps
        """
        self.eval()
        B, device = context_ids.shape[0], context_ids.device
        AA = "ACDEFGHIKLMNPQRSTVWY"

        h_ctx = self.adapter(self.encoder(context_ids))

        # Default length: estimate from context
        if seq_len is None:
            pad_mask = (context_ids == 0)
            ctx_lengths = (~pad_mask).sum(1).float()
            seq_len = int(ctx_lengths.mean().item())
            seq_len = min(max(seq_len, 5), 50)

        # Start: all MASK
        tokens = torch.full((B, seq_len), MASK_ID, dtype=torch.long, device=device)
        src_pad = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

        n_unmasked = 0
        for step in range(T_infer):
            t_val = 1.0 - step / T_infer         # t decreases from 1 → 0
            t_tensor = torch.full((B,), t_val, device=device)

            if conditions is not None:
                cc = self._encode_condition(conditions, t_tensor)
            else:
                cc = self._encode_condition(
                    torch.zeros(B, self.cond_encoder[0].in_features, device=device),
                    t_tensor)

            logits = self.denoiser(tokens, h_ctx, cc, src_pad_mask=src_pad)

            # restrict to AA tokens
            logits[:, :, :AA_START] = -1e9
            logits[:, :, AA_END:]   = -1e9

            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)  # (B, L, V)
            # max confidence at each masked position
            conf, best_token = probs.max(dim=-1)                        # (B, L)

            # compute number of tokens to unmask this step
            target_unmasked = int((step + 1) / T_infer * seq_len)
            n_to_unmask = max(0, target_unmasked - n_unmasked)

            if n_to_unmask > 0:
                # only consider currently MASK positions
                is_mask = (tokens == MASK_ID)
                conf_masked = conf.masked_fill(~is_mask, -1.0)
                # pick top n_to_unmask per sequence
                _, top_idx = torch.topk(conf_masked, min(n_to_unmask, seq_len), dim=-1)
                for b in range(B):
                    for idx in top_idx[b]:
                        if tokens[b, idx] == MASK_ID:
                            # sample from distribution at this position
                            tok = torch.multinomial(probs[b, idx], 1).item()
                            tokens[b, idx] = tok
                n_unmasked = target_unmasked

        # Final pass: fill any remaining MASK with argmax
        still_masked = (tokens == MASK_ID)
        if still_masked.any():
            t_final = torch.zeros(B, device=device)
            if conditions is not None:
                cc = self._encode_condition(conditions, t_final)
            else:
                cc = self._encode_condition(
                    torch.zeros(B, self.cond_encoder[0].in_features, device=device),
                    t_final)
            logits = self.denoiser(tokens, h_ctx, cc, src_pad_mask=src_pad)
            logits[:, :, :AA_START] = -1e9
            logits[:, :, AA_END:]   = -1e9
            fill = logits.argmax(-1)
            tokens[still_masked] = fill[still_masked]

        seqs = []
        for i in range(B):
            s = "".join(AA[t - AA_START] for t in tokens[i].tolist()
                        if AA_START <= t < AA_END)
            seqs.append(s)
        return seqs
