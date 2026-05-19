"""
Conditional AMP generator.

Given a context sequence x_c, generate the target region x_t autoregressively.

Architecture:
  - Frozen (or partially frozen) pre-trained context encoder f_theta
  - Lightweight adapter A_psi on top of encoder output
  - Autoregressive decoder that conditions on adapter output via cross-attention

For short sequences (≤50 AA), a small decoder with cross-attention is sufficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.tokenizer import VOCAB_SIZE, BOS_ID, EOS_ID, PAD_ID
from src.models.encoder import TransformerEncoder

N_BACTERIA = 20  # number of bacteria in GRAMPA; mirrors supervised_dataset constant

# AA token indices 2–21 in vocab order "ACDEFGHIKLMNPQRSTVWY"
_CHARGE_VEC = torch.tensor(
    [0,0,-1,-1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0], dtype=torch.float32)
_KD_VEC = torch.tensor(
    [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3],
    dtype=torch.float32)


class Adapter(nn.Module):
    """Bottleneck adapter: d_model -> bottleneck -> d_model"""
    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        # zero-init so adapter starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ConditionProjector(nn.Module):
    def __init__(self, num_conditions: int = 3, d_model: int = 384,
                 cond_dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_conditions, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.cond_dropout = cond_dropout

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        """conditions: (B, num_conditions) → (B, 1, d_model)
        During training, randomly drop entire condition vectors (CFG-style)
        so the model learns both conditional and unconditional generation.
        """
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(conditions.shape[0], 1, device=conditions.device)
                    > self.cond_dropout).float()
            conditions = conditions * mask
        return self.norm(self.proj(conditions)).unsqueeze(1)


class ARDecoder(nn.Module):
    """
    Autoregressive decoder with cross-attention to encoder memory.
    Generates one token at a time; during training uses teacher forcing.
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 52,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # tie weights with embedding
        self.head.weight = self.token_emb.weight

    def forward(
        self,
        tgt_ids: torch.Tensor,    # (B, T) decoder input tokens
        memory: torch.Tensor,     # (B, L, d_model) encoder output
        memory_key_padding_mask: torch.Tensor | None = None,  # (B, L) True=ignore
    ) -> torch.Tensor:
        """Returns logits (B, T, vocab_size)."""
        B, T = tgt_ids.shape
        positions = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(tgt_ids) + self.pos_emb(positions))

        # causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)

        h = self.transformer(
            x, memory,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        h = self.norm(h)
        return self.head(h)  # (B, T, vocab_size)


class AdaLNDecoderLayer(nn.Module):
    """
    Transformer decoder layer with per-sub-block AdaLN condition injection.
    At init the adaln linear is zero-init'd, so training starts as vanilla transformer.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, d_cond: int):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        # elementwise_affine=False: scale/shift come from condition
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        # 6 outputs: (scale1, shift1, scale2, shift2, scale3, shift3) for 3 sub-blocks
        self.adaln = nn.Linear(d_cond, 6 * d_model)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                                      # (B, T, D)
        memory: torch.Tensor,                                  # (B, L, D)
        cond: torch.Tensor,                                    # (B, d_cond)
        tgt_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s1, b1, s2, b2, s3, b3 = self.adaln(cond).chunk(6, dim=-1)
        # unsqueeze to (B, 1, D) for broadcasting over sequence
        s1, b1 = s1.unsqueeze(1), b1.unsqueeze(1)
        s2, b2 = s2.unsqueeze(1), b2.unsqueeze(1)
        s3, b3 = s3.unsqueeze(1), b3.unsqueeze(1)

        # self-attention sub-block (pre-norm + AdaLN)
        h_norm = (1 + s1) * self.norm1(x) + b1
        attn_out, _ = self.self_attn(h_norm, h_norm, h_norm,
                                     attn_mask=tgt_mask, is_causal=True, need_weights=False)
        x = x + self.drop(attn_out)

        # cross-attention sub-block
        h_norm = (1 + s2) * self.norm2(x) + b2
        cross_out, _ = self.cross_attn(h_norm, memory, memory,
                                       key_padding_mask=memory_key_padding_mask,
                                       need_weights=False)
        x = x + self.drop(cross_out)

        # feed-forward sub-block
        h_norm = (1 + s3) * self.norm3(x) + b3
        x = x + self.drop(self.ff(h_norm))
        return x


class AdaLNDecoder(nn.Module):
    """Autoregressive decoder with per-layer AdaLN condition injection."""
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 52,
        d_cond: int = 128,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            AdaLNDecoderLayer(d_model, nhead, dim_feedforward, dropout, d_cond)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(
        self,
        tgt_ids: torch.Tensor,    # (B, T)
        memory: torch.Tensor,     # (B, L, D)
        cond: torch.Tensor,       # (B, d_cond)
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = tgt_ids.shape
        positions = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(tgt_ids) + self.pos_emb(positions))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)
        for layer in self.layers:
            x = layer(x, memory, cond,
                      tgt_mask=causal_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        return self.head(self.norm(x))  # (B, T, vocab_size)


class ConditionalGeneratorV3(nn.Module):
    """
    AdaLN conditional generator.
    Condition is injected into EVERY decoder layer via adaptive LayerNorm (γ, β),
    instead of a single condition token in cross-attention memory.
    This gives much stronger and more direct control over generation.
    """
    def __init__(
        self,
        encoder: TransformerEncoder,
        d_model: int = 256,
        adapter_bottleneck: int = 64,
        decoder_layers: int = 4,
        decoder_nhead: int = 8,
        decoder_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 52,
        freeze_encoder: bool = True,
        num_conditions: int = 3,
        d_cond: int = 128,
        cond_dropout: float = 0.15,
    ):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self._encoder_needs_grad = any(p.requires_grad for p in self.encoder.parameters())

        self.adapter = Adapter(d_model, bottleneck=adapter_bottleneck)

        # Condition encoder: raw scalars → d_cond latent
        self.cond_encoder = nn.Sequential(
            nn.Linear(num_conditions, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        self.cond_dropout = cond_dropout

        self.decoder = AdaLNDecoder(
            d_model=d_model,
            nhead=decoder_nhead,
            num_layers=decoder_layers,
            dim_feedforward=decoder_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            d_cond=d_cond,
        )

    def _encode_condition(self, conditions: torch.Tensor) -> torch.Tensor:
        """conditions: (B, num_conditions) → (B, d_cond), with CFG dropout."""
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(conditions.shape[0], 1, device=conditions.device)
                    > self.cond_dropout).float()
            conditions = conditions * mask
        return self.cond_encoder(conditions)  # (B, d_cond)

    def forward(
        self,
        context_ids: torch.Tensor,                   # (B, L)
        tgt_ids: torch.Tensor,                        # (B, T)
        conditions: torch.Tensor | None = None,       # (B, num_conditions)
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx = torch.enable_grad() if self._encoder_needs_grad else torch.no_grad()
        with ctx:
            h_ctx = self.encoder(context_ids)
        h_ctx = self.adapter(h_ctx)

        if conditions is not None:
            cond_emb = self._encode_condition(conditions)  # (B, d_cond)
        else:
            B = context_ids.shape[0]
            cond_emb = torch.zeros(B, self.decoder.layers[0].adaln.in_features,
                                   device=context_ids.device)

        return self.decoder(tgt_ids, memory=h_ctx, cond=cond_emb,
                            memory_key_padding_mask=context_key_padding_mask)

    @torch.no_grad()
    def generate(
        self,
        context_ids: torch.Tensor,
        conditions: torch.Tensor | None = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        B = context_ids.shape[0]
        device = context_ids.device

        h_ctx = self.encoder(context_ids)
        h_ctx = self.adapter(h_ctx)

        if conditions is not None:
            cond_emb = self.cond_encoder(conditions)
        else:
            d_cond = self.decoder.layers[0].adaln.in_features
            cond_emb = torch.zeros(B, d_cond, device=device)

        null_cond = torch.zeros_like(cond_emb)

        generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decoder(generated, h_ctx, cond_emb)[:, -1, :]

            if cfg_scale > 0:
                logits_u = self.decoder(generated, h_ctx, null_cond)[:, -1, :]
                logits = logits_u + cfg_scale * (logits - logits_u)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

            logits = logits / temperature
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token[finished] = PAD_ID
            generated = torch.cat([generated, next_token], dim=1)
            finished |= (next_token.squeeze(-1) == EOS_ID)
            if finished.all():
                break

        return generated


class ConditionalGenerator(nn.Module):
    """
    Full conditional generation model:
      1. Encode context with (frozen) pre-trained encoder
      2. Adapt encoder output with lightweight adapter
      3. Decode target sequence autoregressively
    """
    def __init__(
        self,
        encoder: TransformerEncoder,
        d_model: int = 256,
        adapter_bottleneck: int = 64,
        decoder_layers: int = 4,
        decoder_nhead: int = 8,
        decoder_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 52,
        freeze_encoder: bool = True,
        num_conditions: int = 3,
        cond_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self._encoder_needs_grad = any(p.requires_grad for p in self.encoder.parameters())

        self.adapter = Adapter(d_model, bottleneck=adapter_bottleneck)
        self.condition_proj = ConditionProjector(num_conditions=num_conditions, d_model=d_model,
                                                 cond_dropout=cond_dropout)
        self.decoder = ARDecoder(
            d_model=d_model,
            num_layers=decoder_layers,
            nhead=decoder_nhead,
            dim_feedforward=decoder_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

    def forward(
        self,
        context_ids: torch.Tensor,                  # (B, L)
        tgt_ids: torch.Tensor,                       # (B, T)
        conditions: torch.Tensor | None = None,      # (B, num_conditions)
        context_key_padding_mask: torch.Tensor | None = None,  # (B, L)
    ) -> torch.Tensor:
        """Returns logits (B, T, vocab_size)."""
        ctx = torch.enable_grad() if self._encoder_needs_grad else torch.no_grad()
        with ctx:
            h_ctx = self.encoder(context_ids)  # (B, L, D)

        h_ctx = self.adapter(h_ctx)  # (B, L, D)

        if conditions is not None:
            cond_token = self.condition_proj(conditions)  # (B, 1, D)
            memory = torch.cat([cond_token, h_ctx], dim=1)  # (B, L+1, D)
            if context_key_padding_mask is not None:
                cond_mask = torch.zeros(conditions.shape[0], 1, dtype=torch.bool, device=conditions.device)
                memory_mask = torch.cat([cond_mask, context_key_padding_mask], dim=1)
            else:
                memory_mask = None
        else:
            memory = h_ctx
            memory_mask = context_key_padding_mask

        logits = self.decoder(tgt_ids, memory=memory, memory_key_padding_mask=memory_mask)
        return logits  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        context_ids: torch.Tensor,              # (B, L)
        conditions: torch.Tensor | None = None,  # (B, num_conditions)
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive sampling. Returns generated token ids (B, T).
        Stops when EOS is produced or max_new_tokens reached.
        """
        self.eval()
        B = context_ids.shape[0]
        device = context_ids.device

        h_ctx = self.encoder(context_ids)
        h_ctx = self.adapter(h_ctx)

        if conditions is not None:
            cond_token = self.condition_proj(conditions)
            h_ctx = torch.cat([cond_token, h_ctx], dim=1)

        # start with BOS
        # note: generate() does not take a padding mask, so no mask extension needed
        generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decoder(generated, memory=h_ctx)  # (B, t, V)
            next_logits = logits[:, -1, :] / temperature     # (B, V)

            # top-k
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            # top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # once finished, keep padding
            next_token[finished] = PAD_ID
            generated = torch.cat([generated, next_token], dim=1)

            finished |= (next_token.squeeze(-1) == EOS_ID)
            if finished.all():
                break

        return generated  # (B, T)


class ConditionalGeneratorV4(nn.Module):
    """
    v4: dual-pathway conditioning (AdaLN per layer + condition cross-attention token)
    + context dropout to break context dominance.

    Key fixes over v3:
      1. Condition injected BOTH as a prepended memory token (cross-attn) AND via
         AdaLN scale/shift in every decoder layer — two independent gradient paths.
      2. context_dropout: zeroes out encoder output with probability p during training,
         forcing the model to rely on the condition when context is absent.
      3. cond_dropout raised to 0.30 for stronger CFG signal.
    """
    def __init__(
        self,
        encoder: TransformerEncoder,
        d_model: int = 256,
        adapter_bottleneck: int = 64,
        decoder_layers: int = 4,
        decoder_nhead: int = 8,
        decoder_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 52,
        freeze_encoder: bool = True,
        num_conditions: int = 3,
        d_cond: int = 128,
        cond_dropout: float = 0.30,
        context_dropout: float = 0.15,
    ):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self._encoder_needs_grad = any(p.requires_grad for p in self.encoder.parameters())

        self.adapter = Adapter(d_model, bottleneck=adapter_bottleneck)

        # Condition encoder: 3 scalars → d_cond
        self.cond_encoder = nn.Sequential(
            nn.Linear(num_conditions, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        # Project condition to a cross-attention memory token (belt)
        self.cond_token_proj = nn.Linear(d_cond, d_model)
        nn.init.zeros_(self.cond_token_proj.weight)
        nn.init.zeros_(self.cond_token_proj.bias)

        self.cond_dropout    = cond_dropout
        self.context_dropout = context_dropout

        # AdaLN decoder (suspenders)
        self.decoder = AdaLNDecoder(
            d_model=d_model,
            nhead=decoder_nhead,
            num_layers=decoder_layers,
            dim_feedforward=decoder_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            d_cond=d_cond,
        )

    def _encode_condition(self, conditions: torch.Tensor) -> torch.Tensor:
        """Apply CFG-style dropout then encode. Returns (B, d_cond)."""
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(conditions.shape[0], 1, device=conditions.device)
                    > self.cond_dropout).float()
            conditions = conditions * mask
        return self.cond_encoder(conditions)

    def forward(
        self,
        context_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        conditions: torch.Tensor | None = None,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx = torch.enable_grad() if self._encoder_needs_grad else torch.no_grad()
        with ctx:
            h_ctx = self.encoder(context_ids)
        h_ctx = self.adapter(h_ctx)

        # Context dropout: zero the entire context with probability p
        if self.training and self.context_dropout > 0:
            drop_mask = (torch.rand(h_ctx.shape[0], 1, 1, device=h_ctx.device)
                         < self.context_dropout).float()
            h_ctx = h_ctx * (1 - drop_mask)

        if conditions is not None:
            cond_emb = self._encode_condition(conditions)          # (B, d_cond)
        else:
            B = context_ids.shape[0]
            d_cond = self.cond_encoder[0].in_features
            cond_emb = torch.zeros(B, d_cond, device=context_ids.device)

        # Prepend condition as a cross-attention memory token
        cond_tok = self.cond_token_proj(cond_emb).unsqueeze(1)     # (B, 1, d_model)
        memory = torch.cat([cond_tok, h_ctx], dim=1)               # (B, L+1, D)
        if context_key_padding_mask is not None:
            cond_mask = torch.zeros(context_ids.shape[0], 1,
                                    dtype=torch.bool, device=context_ids.device)
            memory_mask = torch.cat([cond_mask, context_key_padding_mask], dim=1)
        else:
            memory_mask = None

        return self.decoder(tgt_ids, memory=memory, cond=cond_emb,
                            memory_key_padding_mask=memory_mask)

    @torch.no_grad()
    def generate(
        self,
        context_ids: torch.Tensor,
        conditions: torch.Tensor | None = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        B = context_ids.shape[0]
        device = context_ids.device

        h_ctx = self.adapter(self.encoder(context_ids))

        if conditions is not None:
            cond_emb = self.cond_encoder(conditions)
        else:
            d_cond = self.cond_encoder[0].in_features
            cond_emb = torch.zeros(B, d_cond, device=device)

        null_cond = torch.zeros_like(cond_emb)
        cond_tok  = self.cond_token_proj(cond_emb).unsqueeze(1)
        null_tok  = self.cond_token_proj(null_cond).unsqueeze(1)
        memory      = torch.cat([cond_tok, h_ctx], dim=1)
        memory_null = torch.cat([null_tok, h_ctx], dim=1)

        generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decoder(generated, memory, cond_emb)[:, -1, :]

            if cfg_scale > 0:
                logits_u = self.decoder(generated, memory_null, null_cond)[:, -1, :]
                logits = logits_u + cfg_scale * (logits - logits_u)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

            logits = logits / temperature
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token[finished] = PAD_ID
            generated = torch.cat([generated, next_token], dim=1)
            finished |= (next_token.squeeze(-1) == EOS_ID)
            if finished.all():
                break

        return generated


class ConditionalGeneratorV5(ConditionalGeneratorV4):
    """
    MIC-conditioned generator.

    Extends v4 with per-bacteria MIC conditioning.
    Condition vector layout (43 dims):
      [0:3]   physchem  — [len/50, tanh(charge/5), tanh(GRAVY)]
      [3:23]  mic_vals  — log2(MIC) for each of 20 bacteria (0 if masked)
      [23:43] mic_mask  — binary: 1 = this bacterium's MIC is specified

    At generation time, set mic_mask[i]=1 and mic_vals[i]=target_log2_mic for
    bacteria you want to constrain; leave others at 0/0.
    """

    def __init__(
        self,
        encoder: TransformerEncoder,
        d_model: int = 256,
        adapter_bottleneck: int = 64,
        decoder_layers: int = 4,
        decoder_nhead: int = 8,
        decoder_ff: int = 1536,
        dropout: float = 0.1,
        max_seq_len: int = 52,
        freeze_encoder: bool = True,
        d_cond: int = 128,
        cond_dropout: float = 0.30,
        context_dropout: float = 0.15,
        n_bacteria: int = N_BACTERIA,
    ):
        # num_conditions = 3 physchem + 20 MIC values + 20 mask bits = 43
        super().__init__(
            encoder=encoder,
            d_model=d_model,
            adapter_bottleneck=adapter_bottleneck,
            decoder_layers=decoder_layers,
            decoder_nhead=decoder_nhead,
            decoder_ff=decoder_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            freeze_encoder=freeze_encoder,
            num_conditions=3 + 2 * n_bacteria,
            d_cond=d_cond,
            cond_dropout=0.0,   # handle CFG dropout manually in v5
            context_dropout=context_dropout,
        )
        self.n_bacteria = n_bacteria
        self._cond_dropout_p = cond_dropout

    def _encode_condition(self, conditions: torch.Tensor) -> torch.Tensor:
        """
        conditions: (B, 43)
          [0:3]   physchem
          [3:23]  mic_vals (0 if not specified)
          [23:43] mic_mask (0/1)

        CFG dropout strategy:
          - With p_physchem=0.30: zero out physchem dims
          - With p_mic=0.50 per bacterium: additionally zero out individual MIC entries
        """
        if self.training and self._cond_dropout_p > 0:
            B = conditions.shape[0]
            # Drop entire physchem block
            phys_drop = (torch.rand(B, 1, device=conditions.device) >
                         self._cond_dropout_p).float()
            # Drop individual bacteria MIC entries
            mic_drop = (torch.rand(B, self.n_bacteria, device=conditions.device) > 0.4).float()

            cond = conditions.clone()
            cond[:, :3] = cond[:, :3] * phys_drop
            # zero mic_vals AND mask for dropped bacteria
            cond[:, 3:3+self.n_bacteria] = cond[:, 3:3+self.n_bacteria] * mic_drop
            cond[:, 3+self.n_bacteria:]  = cond[:, 3+self.n_bacteria:]  * mic_drop
        else:
            cond = conditions

        return self.cond_encoder(cond)  # (B, d_cond)

    @torch.no_grad()
    def generate_mic_targeted(
        self,
        context_ids: torch.Tensor,
        physchem: torch.Tensor | None,          # (B, 3) or None
        target_mic: dict[int, float],           # {bacteria_idx: log2_MIC_target}
        max_new_tokens: int = 50,
        temperature: float = 0.9,
        top_p: float = 0.9,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """
        High-level generation interface.
        target_mic: e.g. {0: -1.0, 1: 3.0} means
          E. coli log2_MIC=-1 (very potent), S. aureus log2_MIC=3 (weak)
        Unspecified bacteria are masked out (don't care).
        """
        B = context_ids.shape[0]
        device = context_ids.device

        if physchem is None:
            physchem = torch.zeros(B, 3, device=device)
        else:
            physchem = physchem.to(device)

        mic_vals = torch.zeros(B, self.n_bacteria, device=device)
        mic_mask = torch.zeros(B, self.n_bacteria, device=device)
        for bact_idx, log2_mic in target_mic.items():
            mic_vals[:, bact_idx] = log2_mic
            mic_mask[:, bact_idx] = 1.0

        conditions = torch.cat([physchem, mic_vals, mic_mask], dim=1)  # (B, 43)
        return self.generate(context_ids, conditions=conditions,
                             max_new_tokens=max_new_tokens,
                             temperature=temperature, top_p=top_p,
                             cfg_scale=cfg_scale)
