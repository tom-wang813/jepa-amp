"""
Training script for NAR and Masked-Diffusion conditional generators.

Usage:
  # Non-autoregressive
  uv run python -m src.train.finetune_nar_diffusion --config configs/finetune_868k_nar.yaml --gpu 0
  # Masked diffusion
  uv run python -m src.train.finetune_nar_diffusion --config configs/finetune_868k_diffusion.yaml --gpu 0
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import build_seq2seq_datasets
from src.models.pretrain_utils import load_pretrained_encoder
from src.models.generator import _CHARGE_VEC, _KD_VEC


def physchem_aux_loss(logits, conditions, tgt_labels):
    """Differentiable physicochemical aux loss (shared with AR trainer)."""
    mask    = (tgt_labels != -100).float()
    n_real  = mask.sum(dim=1).clamp(min=1)
    probs   = F.softmax(logits, dim=-1)[:, :, 2:22]
    charge_vec = _CHARGE_VEC.to(logits.device)
    kd_vec     = _KD_VEC.to(logits.device)
    E_charge   = ((probs * charge_vec).sum(-1) * mask).sum(1)
    E_gravy    = ((probs * kd_vec).sum(-1) * mask).sum(1) / n_real
    tgt_charge = torch.atanh(conditions[:, 1].clamp(-0.9999, 0.9999)) * 5.0
    tgt_gravy  = torch.atanh(conditions[:, 2].clamp(-0.9999, 0.9999))
    return F.mse_loss(E_charge, tgt_charge) + F.mse_loss(E_gravy, tgt_gravy)


def build_batch(batch, device):
    ctx  = batch["context_ids"].to(device)
    tgt  = batch["target_ids"].to(device)
    lbls = batch["target_labels"].to(device)
    cond = batch["conditions"].to(device) if "conditions" in batch else None
    return ctx, tgt, lbls, cond


def train(cfg: dict, gpu: int = 0):
    device   = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and cfg["train"].get("fp16", True)
    gen_version = cfg.get("generator_version", "nar")
    print(f"Device: {device}  fp16={use_fp16}  model={gen_version}")

    # Load pre-trained encoder (JEPA or MLM)
    encoder, pretrain_cfg = load_pretrained_encoder(cfg["pretrain_checkpoint"], device)

    # Build model
    gen_cfg = cfg["generator"]
    if gen_version == "nar":
        from src.models.generator_nar import ConditionalGeneratorNAR
        model = ConditionalGeneratorNAR(
            encoder=encoder,
            d_model=pretrain_cfg["model"]["d_model"],
            freeze_encoder=cfg["train"]["freeze_encoder"],
            **gen_cfg,
        ).to(device)
    elif gen_version == "diffusion":
        from src.models.generator_diffusion import ConditionalGeneratorDiffusion
        model = ConditionalGeneratorDiffusion(
            encoder=encoder,
            d_model=pretrain_cfg["model"]["d_model"],
            freeze_encoder=cfg["train"]["freeze_encoder"],
            **gen_cfg,
        ).to(device)
    else:
        raise ValueError(f"Unknown generator_version: {gen_version}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Data
    data_cfg = cfg["data"]
    train_ds, val_ds = build_seq2seq_datasets(
        data_cfg["fasta_paths"],
        max_len=data_cfg["max_len"],
        val_ratio=data_cfg["val_ratio"],
        prefix_ratio=data_cfg.get("prefix_ratio", 0.5),
        min_prefix_len=data_cfg.get("min_prefix_len", 3),
        max_seq_len=cfg["generator"]["max_seq_len"],
    )
    nw = cfg["train"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=False, num_workers=min(nw, 2), pin_memory=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    physchem_w = cfg["train"].get("physchem_loss_weight", 0.0)
    patience   = cfg["train"].get("patience", 20)
    save_every = cfg["train"].get("save_every", 5)
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            ctx, tgt, lbls, cond = build_batch(batch, device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                if gen_version == "nar":
                    logits, length_logits = model(ctx, tgt, conditions=cond)
                    # CE on token positions
                    ce_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        lbls.reshape(-1), ignore_index=-100,
                    )
                    # Length prediction loss
                    # Ground-truth length = number of real tokens in target
                    tgt_lengths = (lbls != -100).sum(dim=1).clamp(max=model.max_len) - 1
                    len_loss = F.cross_entropy(length_logits, tgt_lengths)
                    loss = ce_loss + 0.1 * len_loss

                else:  # diffusion
                    logits, is_masked = model(ctx, tgt, conditions=cond)
                    # Only compute loss on masked positions
                    if is_masked.any():
                        # tgt (original clean sequence) at masked positions
                        tgt_flat   = tgt.reshape(-1)
                        log_flat   = logits.reshape(-1, logits.shape[-1])
                        mask_flat  = is_masked.reshape(-1)
                        # also exclude PAD
                        valid = mask_flat & (tgt_flat != 0)
                        if valid.any():
                            loss = F.cross_entropy(log_flat[valid], tgt_flat[valid])
                        else:
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                    else:
                        loss = torch.tensor(0.0, device=device, requires_grad=True)

                if physchem_w > 0 and cond is not None:
                    loss = loss + physchem_w * physchem_aux_loss(logits, cond, lbls)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ctx, tgt, lbls, cond = build_batch(batch, device)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    if gen_version == "nar":
                        logits, length_logits = model(ctx, tgt, conditions=cond)
                        tgt_lengths = (lbls != -100).sum(dim=1).clamp(max=model.max_len) - 1
                        loss = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                lbls.reshape(-1), ignore_index=-100)
                                + 0.1 * F.cross_entropy(length_logits, tgt_lengths))
                    else:
                        logits, is_masked = model(ctx, tgt, conditions=cond)
                        tgt_flat  = tgt.reshape(-1)
                        log_flat  = logits.reshape(-1, logits.shape[-1])
                        mask_flat = is_masked.reshape(-1)
                        valid     = mask_flat & (tgt_flat != 0)
                        loss = (F.cross_entropy(log_flat[valid], tgt_flat[valid])
                                if valid.any() else torch.tensor(0.0))
                val_loss += loss.item()
        val_loss /= len(val_loader)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d} | train={train_loss:.4f} | val={val_loss:.4f} | lr={lr:.2e}")

        ckpt_out = {
            "epoch": epoch + 1, "model_state": model.state_dict(),
            "val_loss": val_loss, "cfg": cfg, "pretrain_cfg": pretrain_cfg,
        }
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            torch.save(ckpt_out, save_dir / "best_generator.pt")
            print(f"  -> Saved best (val={val_loss:.4f})")
        else:
            no_improve += 1

        if (epoch + 1) % save_every == 0:
            torch.save(ckpt_out, save_dir / f"epoch_{epoch+1:03d}.pt")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}."); break

    print(f"Done. Best val_loss: {best_val:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu",    type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, gpu=args.gpu)


if __name__ == "__main__":
    main()
