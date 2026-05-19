"""
JEPA pre-training loop.

Usage:
    uv run python -m src.train.pretrain --config configs/jepa_pretrain.yaml
"""

import argparse
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import build_datasets
from src.models.jepa import JEPA


def cosine_schedule(step: int, total_steps: int, base: float, final: float) -> float:
    """Cosine annealing from base to final over total_steps."""
    progress = min(step / max(total_steps, 1), 1.0)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * progress))


def train(cfg: dict, gpu: int = 0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    use_fp16 = device.type == "cuda" and cfg["train"].get("fp16", True)
    print(f"Device: {device}  fp16={use_fp16}")

    # --- data ---
    fasta_paths = cfg["data"]["fasta_paths"]
    train_ds, val_ds = build_datasets(
        fasta_paths,
        max_len=cfg["data"]["max_len"],
        val_ratio=cfg["data"]["val_ratio"],
        block_size=cfg["data"]["block_size"],
        num_target_blocks=cfg["data"]["num_target_blocks"],
    )
    train_num_workers = cfg["train"].get("num_workers", 4)
    val_num_workers = cfg["train"].get("val_num_workers", max(1, train_num_workers // 2))
    persistent_workers = cfg["train"].get("persistent_workers", train_num_workers > 0)
    prefetch_factor = cfg["train"].get("prefetch_factor", 4)

    train_loader_kwargs = dict(
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
    )
    val_loader_kwargs = dict(
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=True,
    )
    if train_num_workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
    if val_num_workers > 0:
        val_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    # --- model ---
    model = JEPA(**cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- optimizer ---
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    total_steps = cfg["train"]["epochs"] * len(train_loader)
    warmup_steps = cfg["train"]["warmup_steps"]

    # --- optional: wandb ---
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        import wandb
        wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"], config=cfg)

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    save_every = cfg["train"].get("save_every", 10)

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for batch in train_loader:
            input_ids    = batch["input_ids"].to(device)
            context_mask = batch["context_mask"].to(device)
            target_mask  = batch["target_mask"].to(device)

            # LR schedule: linear warmup + cosine decay
            if global_step < warmup_steps:
                lr = cfg["train"]["lr"] * global_step / max(warmup_steps, 1)
            else:
                lr = cosine_schedule(
                    global_step - warmup_steps,
                    total_steps - warmup_steps,
                    cfg["train"]["lr"],
                    cfg["train"]["lr"] * 0.1,
                )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # EMA decay schedule: ramp up from ema_start to ema_end
            ema_decay = cosine_schedule(global_step, total_steps, cfg["model"]["ema_decay"], 1.0)
            model.ema_decay = ema_decay

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                out = model(input_ids, context_mask, target_mask)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            model.update_target_encoder()

            train_loss += out["loss"].item()
            global_step += 1

        train_loss /= len(train_loader)

        # --- validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids    = batch["input_ids"].to(device)
                context_mask = batch["context_mask"].to(device)
                target_mask  = batch["target_mask"].to(device)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    out = model(input_ids, context_mask, target_mask)
                val_loss += out["loss"].item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.2e} | ema={ema_decay:.4f} | {elapsed:.1f}s")

        if use_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": lr, "epoch": epoch + 1})

        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "cfg": cfg,
        }

        # always keep the latest weights (used for downstream tasks)
        torch.save(ckpt, save_dir / "last_jepa.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, save_dir / "best_jepa.pt")
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

        if (epoch + 1) % save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch+1:03d}.pt")
            print(f"  -> Saved periodic checkpoint epoch_{epoch+1:03d}.pt")

    print("Pre-training done. Best val_loss:", best_val_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jepa_pretrain.yaml")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, gpu=args.gpu)


if __name__ == "__main__":
    main()
