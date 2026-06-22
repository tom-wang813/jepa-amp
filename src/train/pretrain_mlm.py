"""
MLM pre-training — ablation baseline for JEPA.

Identical masking strategy (block masking), identical encoder architecture,
identical training hyperparameters. Only difference from JEPA: the model
predicts discrete token IDs (cross-entropy) instead of masked latent
representations (MSE + EMA target encoder).

Usage:
    uv run python -m src.train.pretrain_mlm --config configs/mlm_pretrain_868k.yaml
    uv run python -m src.train.pretrain_mlm --config configs/mlm_pretrain_868k_smoke.yaml
"""

import argparse
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import build_datasets
from src.models.mlm import MLMModel


def cosine_schedule(step: int, total_steps: int, base: float, final: float) -> float:
    progress = min(step / max(total_steps, 1), 1.0)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * progress))


def train(cfg: dict, gpu: int = 0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and cfg["train"].get("fp16", True)
    print(f"Device: {device}  fp16={use_fp16}")

    # --- data: reuse JEPA block-masking dataset ---
    train_ds, val_ds = build_datasets(
        cfg["data"]["fasta_paths"],
        max_len=cfg["data"]["max_len"],
        val_ratio=cfg["data"]["val_ratio"],
        block_size=cfg["data"]["block_size"],
        num_target_blocks=cfg["data"]["num_target_blocks"],
    )
    nw = cfg["train"].get("num_workers", 4)
    loader_kw = dict(pin_memory=True)
    if nw > 0:
        loader_kw.update(persistent_workers=True,
                         prefetch_factor=cfg["train"].get("prefetch_factor", 4))
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True,  num_workers=nw, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"],
                              shuffle=False, num_workers=max(1, nw // 2), **loader_kw)

    # --- model ---
    model = MLMModel(**cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    total_steps  = cfg["train"]["epochs"] * len(train_loader)
    warmup_steps = cfg["train"]["warmup_steps"]

    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        import wandb
        wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"], config=cfg)

    save_dir   = Path(cfg["train"]["save_dir"])
    save_every = cfg["train"].get("save_every", 10)
    save_dir.mkdir(parents=True, exist_ok=True)

    scaler       = torch.cuda.amp.GradScaler(enabled=use_fp16)
    global_step  = 0
    best_val_loss = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        t0         = time.time()
        train_loss = 0.0

        for batch in train_loader:
            input_ids   = batch["input_ids"].to(device)
            target_mask = batch["target_mask"].to(device)

            # LR: linear warmup + cosine decay (identical schedule to JEPA)
            if global_step < warmup_steps:
                lr = cfg["train"]["lr"] * global_step / max(warmup_steps, 1)
            else:
                lr = cosine_schedule(global_step - warmup_steps,
                                     total_steps - warmup_steps,
                                     cfg["train"]["lr"],
                                     cfg["train"]["lr"] * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                out = model(input_ids, target_mask)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += out["loss"].item()
            global_step += 1

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids   = batch["input_ids"].to(device)
                target_mask = batch["target_mask"].to(device)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    out = model(input_ids, target_mask)
                val_loss += out["loss"].item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:03d} | train={train_loss:.4f} | val={val_loss:.4f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        if use_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                       "lr": lr, "epoch": epoch + 1})

        ckpt = {
            "epoch":           epoch + 1,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss":        val_loss,
            "cfg":             cfg,
            "pretrain_type":   "mlm",   # consumed by load_pretrained_encoder()
        }
        torch.save(ckpt, save_dir / "last_mlm.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, save_dir / "best_mlm.pt")
            print(f"  -> best checkpoint (val_loss={val_loss:.4f})")

        if (epoch + 1) % save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch+1:03d}.pt")

    print(f"Pre-training done. Best val_loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlm_pretrain_868k.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, gpu=args.gpu)


if __name__ == "__main__":
    main()
