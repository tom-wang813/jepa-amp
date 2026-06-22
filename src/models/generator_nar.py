"""
Non-autoregressive conditional AMP generator (NAR).

Architecture:
  - Frozen JEPA context encoder  →  adapter  →  h_ctx (B, L, d)
  - Condition encoder  →  cond_emb (B, d_cond)
  - Length head: predict target sequence length from [CLS] of h_ctx + cond_emb
  - NAR decoder: bidirectional transformer that attends to h_ctx and outputs
    all token logits in one parallel forward pass
  - Iterative refinement at inference: re-mask lowest-confidence positions
    and re-predict for up to `refine_steps` rounds

Why non-autoregressive?
  - ~L× faster inference (no sequential token loop)
  - Global coherence: every position sees every other position
  - Different inductive bias from AR → complementary diversity
  - Direct comparison to AR on the same JEPA backbone isolates the
    effect of the generation paradigm

Training objective:
  CE on all (non-PAD) target positions  +  MSE on length prediction
  Teacher-forcing not needed: decoder always receives all-MASK input.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.tokenizer import VOCAB_SIZE, BOS_ID, EOS_ID, PAD_ID, MASK_ID
from src.models.encoder import TransformerEncoder

AA_START = 2   # first AA token id
AA_END   = 22  # last AA token id (exclusive)


# ── helpers ───────────────────────────────────────────────────────────────────

class Adapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.norm(x + self.net(x))


class AdaLNLayer(nn.Module):
    """Bidirectional transformer layer with AdaLN condition injection."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, d_cond: int):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaln = nn.Linear(d_cond, 6 * d_model)
        nn.init.zeros_(self.adaln.weight); nn.init.zeros_(self.adaln.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, cond,
                src_key_padding_mask=None, memory_key_padding_mask=None):
        s1, b1, s2, b2, s3, b3 = self.adaln(cond).chunk(6, dim=-1)
        s1, b1 = s1.unsqueeze(1), b1.unsqueeze(1)
        s2, b2 = s2.unsqueeze(1), b2.unsqueeze(1)
        s3, b3 = s3.unsqueeze(1), b3.unsqueeze(1)

        # bidirectional self-attention (no causal mask)
        h = (1 + s1) * self.norm1(x) + b1
        attn, _ = self.self_attn(h, h, h, key_padding_mask=src_key_padding_mask,
                                 need_weights=False)
        x = x + self.drop(attn)

        # cross-attention to JEPA memory
        h = (1 + s2) * self.norm2(x) + b2
        cross, _ = self.cross_attn(h, memory, memory,
                                   key_padding_mask=memory_key_padding_mask,
                                   need_weights=False)
        x = x + self.drop(cross)

        h = (1 + s3) * self.norm3(x) + b3
        x = x + self.drop(self.ff(h))
        return x


class NARDecoder(nn.Module):
    """
    Bidirectional decoder: takes MASK tokens of predicted length,
    attends to JEPA memory and condition, predicts all positions in parallel.
    """
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 384,
                 nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 1536,
                 dropout: float = 0.1, max_seq_len: int = 52, d_cond: int = 128):
        super().__init__()
        self.d_model   = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            AdaLNLayer(d_model, nhead, dim_feedforward, dropout, d_cond)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, tgt_ids, memory, cond,
                tgt_pad_mask=None, memory_key_padding_mask=None):
        """
        tgt_ids: (B, T) — all MASK during inference; mixed during training
        Returns logits (B, T, vocab_size)
        """
        B, T = tgt_ids.shape
        pos = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(tgt_ids) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x, memory, cond,
                      src_key_padding_mask=tgt_pad_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        return self.head(self.norm(x))   # (B, T, vocab_size)


# ── full model ─────────────────────────────────────────────────────────────────

class ConditionalGeneratorNAR(nn.Module):
    """
    Non-autoregressive conditional generator.

    Key differences from v4 (AR):
      - Decoder is bidirectional (all positions attend to all others)
      - Inference: predict all positions in one pass (+ optional refinement)
      - Length prediction head
    """
    def __init__(
        self,
        encoder: TransformerEncoder,
        d_model: int = 384,
        adapter_bottleneck: int = 128,
        decoder_layers: int = 4,
        decoder_nhead: int = 8,
        decoder_ff: int = 1536,
        dropout: float = 0.1,
        max_seq_len: int = 52,
        freeze_encoder: bool = True,
        num_conditions: int = 3,
        d_cond: int = 128,
        cond_dropout: float = 0.30,
        context_dropout: float = 0.15,
        max_len: int = 50,        # max generated length (excl. special tokens)
    ):
        super().__init__()
        self.max_len = max_len

        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self._enc_grad = any(p.requires_grad for p in self.encoder.parameters())

        self.adapter = Adapter(d_model, adapter_bottleneck)

        # Condition encoder
        self.cond_encoder = nn.Sequential(
            nn.Linear(num_conditions, d_cond), nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        self.cond_dropout    = cond_dropout
        self.context_dropout = context_dropout

        # Length prediction head
        # Input: mean-pooled h_ctx (d_model) + cond_emb (d_cond)
        self.length_head = nn.Sequential(
            nn.Linear(d_model + d_cond, 256), nn.GELU(),
            nn.Linear(256, max_len),   # logits over lengths 1..max_len
        )

        self.decoder = NARDecoder(
            d_model=d_model, nhead=decoder_nhead, num_layers=decoder_layers,
            dim_feedforward=decoder_ff, dropout=dropout,
            max_seq_len=max_seq_len, d_cond=d_cond,
        )

    def _encode_condition(self, conditions):
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(conditions.shape[0], 1, device=conditions.device)
                    > self.cond_dropout).float()
            conditions = conditions * mask
        return self.cond_encoder(conditions)

    def forward(self, context_ids, tgt_ids, conditions=None,
                context_key_padding_mask=None):
        """
        Training forward pass.
        tgt_ids: (B, T) ground-truth suffix token ids (teacher-forced input)
                 — replaced with all-MASK for pure NAR training
        Returns: (logits, length_logits)
          logits:        (B, T, vocab_size)
          length_logits: (B, max_len)   — predicts target sequence length
        """
        ctx = torch.enable_grad() if self._enc_grad else torch.no_grad()
        with ctx:
            h_ctx = self.encoder(context_ids)
        h_ctx = self.adapter(h_ctx)

        # Context dropout
        if self.training and self.context_dropout > 0:
            drop = (torch.rand(h_ctx.shape[0], 1, 1, device=h_ctx.device)
                    < self.context_dropout).float()
            h_ctx = h_ctx * (1 - drop)

        if conditions is not None:
            cond_emb = self._encode_condition(conditions)
        else:
            B = context_ids.shape[0]
            cond_emb = torch.zeros(B, self.cond_encoder[0].in_features,
                                   device=context_ids.device)

        # Length prediction from mean-pooled context + condition
        pad_mask = (context_ids == 0)
        h_mean = h_ctx.masked_fill(pad_mask.unsqueeze(-1), 0).sum(1)
        h_mean = h_mean / (~pad_mask).sum(1, keepdim=True).float().clamp(min=1)
        length_logits = self.length_head(torch.cat([h_mean, cond_emb], dim=-1))

        # Build NAR input: replace all real tokens with MASK
        tgt_masked = torch.full_like(tgt_ids, MASK_ID)
        # Keep PAD positions as PAD
        tgt_masked = tgt_masked.masked_fill(tgt_ids == PAD_ID, PAD_ID)
        tgt_pad = (tgt_masked == PAD_ID)

        logits = self.decoder(tgt_masked, h_ctx, cond_emb,
                              tgt_pad_mask=tgt_pad,
                              memory_key_padding_mask=context_key_padding_mask)
        return logits, length_logits

    @torch.no_grad()
    def generate(
        self,
        context_ids: torch.Tensor,
        conditions: torch.Tensor | None = None,
        temperature: float = 1.0,
        refine_steps: int = 3,
        refine_mask_ratio: float = 0.5,
    ) -> list[str]:
        """
        Generate sequences.
        refine_steps: number of iterative refinement rounds
        refine_mask_ratio: fraction of low-confidence positions to re-mask per round
        Returns list of amino acid strings.
        """
        from src.data.tokenizer import decode
        self.eval()
        B, device = context_ids.shape[0], context_ids.device

        h_ctx = self.adapter(self.encoder(context_ids))

        if conditions is not None:
            cond_emb = self.cond_encoder(conditions)
        else:
            cond_emb = torch.zeros(B, self.cond_encoder[0].in_features, device=device)

        # Predict lengths
        pad_mask = (context_ids == 0)
        h_mean = h_ctx.masked_fill(pad_mask.unsqueeze(-1), 0).sum(1)
        h_mean = h_mean / (~pad_mask).sum(1, keepdim=True).float().clamp(min=1)
        length_logits = self.length_head(torch.cat([h_mean, cond_emb], dim=-1))
        lengths = (length_logits / max(temperature, 0.1)).argmax(-1) + 1  # 1..max_len
        max_L = int(lengths.max().item())

        # Initial all-MASK input
        tgt = torch.full((B, max_L), MASK_ID, dtype=torch.long, device=device)
        tgt_pad = torch.zeros(B, max_L, dtype=torch.bool, device=device)
        for i, l in enumerate(lengths.tolist()):
            tgt_pad[i, l:] = True
            tgt[i, l:] = PAD_ID

        # Initial parallel prediction
        logits = self.decoder(tgt, h_ctx, cond_emb,
                              tgt_pad_mask=tgt_pad)  # (B, L, V)
        # mask non-AA logits (keep only AA tokens 2..21)
        logits[:, :, :AA_START]  = -1e9
        logits[:, :, AA_END:]    = -1e9
        probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
        tokens = torch.multinomial(probs.view(-1, VOCAB_SIZE),
                                   num_samples=1).view(B, max_L)
        tokens.masked_fill_(tgt_pad, PAD_ID)

        # Iterative refinement
        for step in range(refine_steps):
            # compute per-position confidence
            conf = probs.gather(-1, tokens.clamp(0, VOCAB_SIZE-1).unsqueeze(-1)).squeeze(-1)
            conf.masked_fill_(tgt_pad, 1.0)   # don't re-mask padding

            # re-mask the least confident positions
            n_mask = max(1, int(refine_mask_ratio * max_L * (1 - step / refine_steps)))
            _, low_idx = torch.topk(conf, n_mask, dim=-1, largest=False)
            tgt_refine = tokens.clone()
            for b in range(B):
                tgt_refine[b, low_idx[b]] = MASK_ID

            logits = self.decoder(tgt_refine, h_ctx, cond_emb, tgt_pad_mask=tgt_pad)
            logits[:, :, :AA_START] = -1e9
            logits[:, :, AA_END:]   = -1e9
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            new_tokens = torch.multinomial(probs.view(-1, VOCAB_SIZE),
                                           num_samples=1).view(B, max_L)
            # only update re-masked positions
            mask_positions = (tgt_refine == MASK_ID)
            tokens[mask_positions] = new_tokens[mask_positions]

        # decode to strings
        AA = "ACDEFGHIKLMNPQRSTVWY"
        seqs = []
        for i in range(B):
            L = int(lengths[i].item())
            toks = tokens[i, :L].tolist()
            s = "".join(AA[t - AA_START] for t in toks
                        if AA_START <= t < AA_END)
            seqs.append(s)
        return seqs
