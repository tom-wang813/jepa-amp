"""
Fine-tuning the conditional generator on top of a pre-trained JEPA encoder.

Usage:
    uv run python -m src.train.finetune --config configs/finetune.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import (
    build_seq2seq_datasets, build_seq2seq_datasets_v5,
    build_seq2seq_datasets_grampa_v5, build_seq2seq_datasets_v6,
    build_seq2seq_datasets_v7,
)
from src.models.pretrain_utils import load_pretrained_encoder
from src.models.generator import (
    ConditionalGenerator, ConditionalGeneratorV3, ConditionalGeneratorV4,
    ConditionalGeneratorV5,
    _CHARGE_VEC, _KD_VEC,
)


def load_mic_oracle(cfg: dict, device: torch.device):
    """Load frozen MIC predictor as differentiable oracle for aux loss."""
    import yaml
    from src.models.pretrain_utils import load_pretrained_encoder
    from src.models.supervised_head import JEPAMICPredictor
    from src.data.supervised_dataset import N_BACTERIA

    mic_cfg_path = cfg.get("mic_oracle_cfg")
    mic_ckpt_path = cfg.get("mic_oracle_ckpt")
    if not (mic_cfg_path and mic_ckpt_path):
        return None

    project_root = Path(cfg.get("_project_root", Path(__file__).resolve().parents[2]))
    with open(project_root / mic_cfg_path) as f:
        mc = yaml.safe_load(f)

    enc, pt_cfg = load_pretrained_encoder(
        str(project_root / mc["pretrain_checkpoint"]), device)

    head_cfg  = mc["head"].copy()
    head_type = head_cfg.pop("head_type", "transformer")
    model = JEPAMICPredictor(
        encoder=enc,
        d_model=pt_cfg["model"]["d_model"],
        n_bacteria=N_BACTERIA,
        head_type=head_type,
        freeze_encoder=True,
        **head_cfg,
    ).to(device)

    ckpt = torch.load(project_root / mic_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    print(f"MIC oracle loaded (frozen): {mic_ckpt_path}")
    return model


@torch.no_grad()
def mic_oracle_loss_batch(
    mic_oracle,
    logits: "torch.Tensor",
    conditions: "torch.Tensor",
    tgt_labels: "torch.Tensor",
) -> "torch.Tensor":
    """
    Non-differentiable MIC oracle loss: decode argmax sequences, run through
    frozen MIC predictor, compute MSE vs target MIC where mask=1.

    Used for logging / reward signal only (no backward).
    conditions: (B, 43) = [physchem(3) | mic_vals(20) | mic_mask(20)]
    """
    import numpy as np
    from src.data.tokenizer import PAD_ID

    AA = "ACDEFGHIKLMNPQRSTVWY"
    device = logits.device
    B = logits.shape[0]
    N_BACT = 20

    mic_vals = conditions[:, 3:23]   # (B, 20)
    mic_mask = conditions[:, 23:43]  # (B, 20)

    # Only compute loss for samples that have at least one MIC target
    has_target = mic_mask.sum(1) > 0  # (B,)
    if not has_target.any():
        return torch.tensor(0.0, device=device)

    # Decode argmax sequences
    token_ids = logits.argmax(-1)  # (B, T)
    label_mask = (tgt_labels != -100)

    # Build padded token sequences for oracle input
    max_l = label_mask.sum(1).max().item() + 2  # +BOS+EOS
    max_l = max(int(max_l), 5)
    padded = torch.full((B, max_l), PAD_ID, dtype=torch.long, device=device)
    padded[:, 0] = 0  # BOS
    for b in range(B):
        toks = token_ids[b][label_mask[b]]
        end = min(len(toks), max_l - 2)
        padded[b, 1:end + 1] = toks[:end]
        padded[b, end + 1] = 1  # EOS

    total_loss = torch.tensor(0.0, device=device)
    n_active = has_target.sum().item()

    for bi in range(N_BACT):
        col_mask = (mic_mask[:, bi] > 0.5) & has_target
        if not col_mask.any():
            continue
        bact_ids = torch.full((B,), bi, dtype=torch.long, device=device)
        preds = mic_oracle(padded, bact_ids)  # (B,)
        tgt   = mic_vals[:, bi]
        loss  = ((preds - tgt) ** 2 * col_mask.float()).sum() / col_mask.float().sum()
        total_loss = total_loss + loss

    return total_loss


def physchem_aux_loss(
    logits: torch.Tensor,
    conditions: torch.Tensor,
    tgt_labels: torch.Tensor,
) -> torch.Tensor:
    """Legacy combined physicochemical loss (charge + GRAVY, equal weights)."""
    mask   = (tgt_labels != -100).float()
    n_real = mask.sum(dim=1).clamp(min=1)
    probs_aa = F.softmax(logits, dim=-1)[:, :, 2:22]
    charge_vec = _CHARGE_VEC.to(logits.device)
    kd_vec     = _KD_VEC.to(logits.device)
    E_charge = ((probs_aa * charge_vec).sum(-1) * mask).sum(1)
    E_gravy  = ((probs_aa * kd_vec).sum(-1) * mask).sum(1) / n_real
    tgt_charge = torch.atanh(conditions[:, 1].clamp(-0.9999, 0.9999)) * 5.0
    tgt_gravy  = torch.atanh(conditions[:, 2].clamp(-0.9999, 0.9999))
    return F.mse_loss(E_charge, tgt_charge) + F.mse_loss(E_gravy, tgt_gravy)


def physchem_aux_loss_v7(
    logits: torch.Tensor,
    conditions: torch.Tensor,
    tgt_labels: torch.Tensor,
    charge_weight: float = 0.5,
    gravy_weight: float = 2.5,
) -> torch.Tensor:
    """
    V7 physicochemical aux loss with separate charge and GRAVY weights.
    Higher gravy_weight overcomes CE-loss dominance that caused GRAVY R²≈0.
    Fully differentiable via soft token probabilities.
    """
    mask   = (tgt_labels != -100).float()          # (B, T)
    n_real = mask.sum(dim=1).clamp(min=1)           # (B,)
    probs_aa = F.softmax(logits, dim=-1)[:, :, 2:22]   # (B, T, 20)
    charge_vec = _CHARGE_VEC.to(logits.device)
    kd_vec     = _KD_VEC.to(logits.device)
    E_charge = ((probs_aa * charge_vec).sum(-1) * mask).sum(1)           # (B,)
    E_gravy  = ((probs_aa * kd_vec).sum(-1) * mask).sum(1) / n_real      # (B,)
    tgt_charge = torch.atanh(conditions[:, 1].clamp(-0.9999, 0.9999)) * 5.0
    tgt_gravy  = torch.atanh(conditions[:, 2].clamp(-0.9999, 0.9999))
    return (charge_weight * F.mse_loss(E_charge, tgt_charge) +
            gravy_weight  * F.mse_loss(E_gravy,  tgt_gravy))


def load_hc50_oracle(cfg: dict, device: torch.device):
    """Load frozen HC50 predictor (MeanPoolRegressor from qmap script)."""
    import torch.nn as nn

    hc50_ckpt_path = cfg.get("hc50_oracle_ckpt")
    if not hc50_ckpt_path:
        return None

    project_root = Path(cfg.get("_project_root", Path(__file__).resolve().parents[2]))
    ckpt_path = project_root / hc50_ckpt_path
    if not ckpt_path.exists():
        print(f"WARNING: HC50 oracle checkpoint not found at {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    pretrain_ckpt = args.get("checkpoint", "checkpoints/jepa_pretrain_868k/last_jepa.pt")
    enc, pt_cfg = load_pretrained_encoder(str(project_root / pretrain_ckpt), device)
    d_model = pt_cfg["model"]["d_model"]

    class _HC50Oracle(nn.Module):
        def __init__(self, encoder, d_model, hidden, dropout):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden),  nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        def forward(self, input_ids, lengths):
            h = self.encoder(input_ids)  # (B, L, D)
            pooled = torch.stack([h[i, 1:lengths[i]-1].mean(0) for i in range(len(lengths))])
            return self.head(pooled).squeeze(-1)

    model = _HC50Oracle(enc, d_model, hidden=args.get("hidden", 512),
                        dropout=args.get("dropout", 0.25)).to(device)
    model.load_state_dict(ckpt["model_state"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    print(f"HC50 oracle loaded (frozen): {hc50_ckpt_path}")
    return model


@torch.no_grad()
def hc50_oracle_loss_batch(
    hc50_oracle,
    logits: torch.Tensor,
    conditions: torch.Tensor,
    tgt_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Non-differentiable HC50 oracle loss.
    conditions[:, 3] = tanh(log10_HC50 / 3).
    Decodes argmax sequences, runs through frozen HC50 predictor.
    """
    from src.data.tokenizer import PAD_ID, EOS_ID
    device = logits.device
    B = logits.shape[0]

    token_ids  = logits.argmax(-1)      # (B, T)
    label_mask = (tgt_labels != -100)   # (B, T)

    max_l = int(label_mask.sum(1).max().item()) + 2
    max_l = max(max_l, 5)
    padded  = torch.full((B, max_l), PAD_ID, dtype=torch.long, device=device)
    lengths = torch.zeros(B, dtype=torch.long, device=device)
    padded[:, 0] = 0  # BOS
    for b in range(B):
        toks = token_ids[b][label_mask[b]]
        end  = min(len(toks), max_l - 2)
        padded[b, 1:end + 1] = toks[:end]
        padded[b, end + 1]   = EOS_ID
        lengths[b] = end + 2

    preds = hc50_oracle(padded, lengths)                          # (B,) log10 HC50
    tgt_hc50 = torch.atanh(conditions[:, 3].clamp(-0.9999, 0.9999)) * 3.0  # decode
    return F.mse_loss(preds, tgt_hc50)


def build_generator_batch(batch: dict, device: torch.device):
    """Move a seq2seq batch to device."""
    context_ids = batch["context_ids"].to(device)
    tgt_ids = batch["target_ids"].to(device)
    tgt_labels = batch["target_labels"].to(device)
    conditions = batch["conditions"].to(device) if "conditions" in batch else None
    return context_ids, tgt_ids, tgt_labels, conditions


def train(cfg: dict, gpu: int = 0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    use_fp16 = device.type == "cuda" and cfg["train"].get("fp16", True)
    print(f"Device: {device}  fp16={use_fp16}")

    # --- load pretrained encoder (JEPA or MLM) ---
    encoder, pretrain_cfg = load_pretrained_encoder(cfg["pretrain_checkpoint"], device)

    # --- build generator ---
    gen_version = cfg.get("generator_version", "v2")
    if gen_version in ("v5", "grampa_v5"):
        GenClass = ConditionalGeneratorV5
    elif gen_version in ("v4", "v6", "v7"):
        GenClass = ConditionalGeneratorV4
    elif gen_version == "v3":
        GenClass = ConditionalGeneratorV3
    else:
        GenClass = ConditionalGenerator
    gen = GenClass(
        encoder=encoder,
        d_model=pretrain_cfg["model"]["d_model"],
        freeze_encoder=cfg["train"]["freeze_encoder"],
        **cfg["generator"],
    ).to(device)

    trainable = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # --- data ---
    data_cfg = cfg["data"]
    ds_kwargs = dict(
        max_len=data_cfg["max_len"],
        val_ratio=data_cfg["val_ratio"],
        prefix_ratio=data_cfg.get("prefix_ratio", 0.5),
        min_prefix_len=data_cfg.get("min_prefix_len", 3),
        max_seq_len=cfg["generator"]["max_seq_len"],
    )
    if gen_version == "v7":
        train_ds, val_ds = build_seq2seq_datasets_v7(
            data_cfg["fasta_paths"],
            hc50_cache_path=data_cfg.get("hc50_cache"),
            **ds_kwargs,
        )
    elif gen_version == "v6":
        train_ds, val_ds = build_seq2seq_datasets_v6(
            data_cfg["fasta_paths"],
            amp_score_cache_path=data_cfg.get("amp_score_cache"),
            **ds_kwargs,
        )
    elif gen_version in ("v5", "grampa_v5"):
        from pathlib import Path as _Path
        if gen_version == "grampa_v5":
            train_ds, val_ds = build_seq2seq_datasets_grampa_v5(
                grampa_csv=_Path(data_cfg["grampa_csv"]),
                fasta_paths=[_Path(p) for p in data_cfg.get("fasta_paths", [])] or None,
                mic_pseudolabel_npy=_Path(data_cfg["mic_pseudolabel_npy"]) if data_cfg.get("mic_pseudolabel_npy") else None,
                mic_pseudolabel_seqs=_Path(data_cfg["mic_pseudolabel_seqs"]) if data_cfg.get("mic_pseudolabel_seqs") else None,
                pseudo_mic_mask_prob=data_cfg.get("pseudo_mic_mask_prob", 0.30),
                random_drop_prob=data_cfg.get("random_drop_prob", 0.10),
                n_repeats=data_cfg.get("n_repeats", 10),
                **ds_kwargs,
            )
        else:
            train_ds, val_ds = build_seq2seq_datasets_v5(
                data_cfg["fasta_paths"],
                mic_pseudolabel_npy=_Path(data_cfg["mic_pseudolabel_npy"]),
                mic_pseudolabel_seqs=_Path(data_cfg["mic_pseudolabel_seqs"]),
                mic_mask_prob=data_cfg.get("mic_mask_prob", 0.30),
                **ds_kwargs,
            )
    else:
        train_ds, val_ds = build_seq2seq_datasets(
            data_cfg["fasta_paths"], **ds_kwargs
        )
    nw = cfg["train"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,  num_workers=nw,          pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=min(nw, 2),  pin_memory=True)

    # --- optimizer: only adapter + decoder parameters ---
    optimizer = torch.optim.AdamW(
        [p for p in gen.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        import wandb
        wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"], config=cfg)

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    patience           = cfg["train"].get("patience", 10)
    save_every         = cfg["train"].get("save_every", 5)
    physchem_weight    = cfg["train"].get("physchem_loss_weight", 0.0)
    mic_oracle_weight  = cfg["train"].get("mic_oracle_loss_weight", 0.0)
    hc50_oracle_weight = cfg["train"].get("hc50_oracle_loss_weight", 0.0)
    charge_weight      = cfg["train"].get("charge_loss_weight", 0.5)
    gravy_weight       = cfg["train"].get("gravy_loss_weight", 2.5)
    use_v7_loss        = gen_version == "v7"

    project_root = Path(__file__).resolve().parents[2]
    cfg["train"]["_project_root"] = str(project_root)
    mic_oracle  = load_mic_oracle(cfg["train"], device) if mic_oracle_weight > 0 else None
    hc50_oracle = load_hc50_oracle(cfg["train"], device) if hc50_oracle_weight > 0 else None

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(cfg["train"]["epochs"]):
        gen.train()
        train_loss = 0.0

        for batch in train_loader:
            context_ids, tgt_ids, tgt_labels, conditions = build_generator_batch(batch, device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                logits = gen(context_ids, tgt_ids, conditions=conditions)  # (B, T, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt_labels.reshape(-1),
                    ignore_index=-100,
                )
                if conditions is not None:
                    if use_v7_loss:
                        loss = loss + physchem_aux_loss_v7(
                            logits, conditions, tgt_labels,
                            charge_weight=charge_weight,
                            gravy_weight=gravy_weight)
                    elif physchem_weight > 0:
                        loss = loss + physchem_weight * physchem_aux_loss(
                            logits, conditions, tgt_labels)
            if conditions is not None:
                if mic_oracle_weight > 0 and mic_oracle is not None:
                    with torch.cuda.amp.autocast(enabled=False):
                        mic_loss = mic_oracle_loss_batch(
                            mic_oracle, logits.float().detach(), conditions, tgt_labels)
                    loss = loss + mic_oracle_weight * mic_loss
                if hc50_oracle_weight > 0 and hc50_oracle is not None:
                    with torch.cuda.amp.autocast(enabled=False):
                        hc50_loss = hc50_oracle_loss_batch(
                            hc50_oracle, logits.float().detach(), conditions, tgt_labels)
                    loss = loss + hc50_oracle_weight * hc50_loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # --- validation ---
        gen.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                context_ids, tgt_ids, tgt_labels, conditions = build_generator_batch(batch, device)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    logits = gen(context_ids, tgt_ids, conditions=conditions)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        tgt_labels.reshape(-1),
                        ignore_index=-100,
                    )
                val_loss += loss.item()
        val_loss /= len(val_loader)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d} | train={train_loss:.4f} | val={val_loss:.4f} | lr={lr:.2e}")

        if use_wandb:
            wandb.log({"gen/train_loss": train_loss, "gen/val_loss": val_loss, "epoch": epoch + 1})

        ckpt = {
            "epoch": epoch + 1,
            "model_state": gen.state_dict(),
            "val_loss": val_loss,
            "cfg": cfg,
            "pretrain_cfg": pretrain_cfg,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(ckpt, save_dir / "best_generator.pt")
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            no_improve += 1

        if (epoch + 1) % save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch+1:03d}.pt")
            print(f"  -> Saved periodic checkpoint epoch_{epoch+1:03d}.pt")

        if no_improve >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    print("Fine-tuning done. Best val_loss:", best_val_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, gpu=args.gpu)


if __name__ == "__main__":
    main()
