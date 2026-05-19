"""
Supervised fine-tuning on top of a pre-trained JEPA encoder.

Supports three modes (set via config `task`):
  amp        – binary AMP classification (BCE)
  amp+tox    – AMP + toxicity dual-head (BCE x2, toxicity loss masked for unknown labels)
  mic        – MIC regression (Huber loss), bacteria conditioning via FiLM or Transformer token
  multitask  – AMP classification + MIC regression jointly (interleaved batches)

Usage:
  uv run python -m src.train.finetune_supervised --config configs/amp_classifier_868k.yaml
  uv run python -m src.train.finetune_supervised --config configs/mic_868k.yaml
"""

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

from src.data.tokenizer import PAD_ID
from src.data.supervised_dataset import (
    AMPClassificationDataset,
    load_fasta_sequences,
    load_grampa,
    collate_supervised,
    BACTERIA_TO_IDX,
    N_BACTERIA,
)
from src.models.jepa import JEPA
from src.models.encoder import TransformerEncoder
from src.models.supervised_head import JEPAClassifier, JEPAMICPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_encoder(pretrain_ckpt: str, device: torch.device) -> tuple[TransformerEncoder, dict]:
    ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    jepa = JEPA(**cfg["model"])
    jepa.load_state_dict(ckpt["model_state"])
    print(f"Loaded JEPA encoder (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return jepa.context_encoder, cfg


def _build_neg_sequences(neg_cfg: dict, max_len: int) -> list[str]:
    """Load negative AMP sequences from one or multiple fastas, or UniProt download."""
    seqs = []
    if "fastas" in neg_cfg:
        for path in neg_cfg["fastas"]:
            seqs.extend(load_fasta_sequences(path, max_len=max_len))
        return list(dict.fromkeys(seqs))  # dedup, preserve order
    if "fasta" in neg_cfg:
        return load_fasta_sequences(neg_cfg["fasta"], max_len=max_len)
    from src.eval.amp_classifier import _fetch_non_amp_sequences
    return _fetch_non_amp_sequences(max_seqs=neg_cfg.get("max_seqs", 50_000), max_len=max_len)


def _amp_loss(logits: torch.Tensor, labels: torch.Tensor,
              pos_weight: float | None = None,
              label_smoothing: float = 0.0) -> torch.Tensor:
    if label_smoothing > 0:
        labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
    pw = torch.tensor([pos_weight], device=logits.device) if pos_weight is not None else None
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)


def _tox_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # mask out unknown labels (-1)
    mask = labels >= 0
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def _mic_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(preds, targets, delta=1.0)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm.std() * ym.std()).clamp(min=1e-8)
    return float((xm * ym).mean() / denom)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_classifier(cfg: dict, gpu: int = 0) -> None:
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and cfg["train"].get("fp16", True)

    encoder, pretrain_cfg = _load_encoder(cfg["pretrain_checkpoint"], device)
    d_model = pretrain_cfg["model"]["d_model"]
    max_len = pretrain_cfg["model"].get("max_seq_len", 52)

    task = cfg.get("task", "amp")
    n_tox = 1 if "tox" in task else 0

    model = JEPAClassifier(
        encoder=encoder,
        d_model=d_model,
        freeze_encoder=cfg["train"]["freeze_encoder"],
        n_tox=n_tox,
        **cfg["head"],
    ).to(device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- data ---
    data_cfg = cfg["data"]
    pos_seqs = load_fasta_sequences(data_cfg["pos_fasta"], max_len=max_len - 2)
    neg_seqs = _build_neg_sequences(data_cfg["neg"], max_len=max_len - 2)

    # balance: subsample majority class to match minority class
    if data_cfg.get("balance", False):
        import random as _rng
        _rng.seed(42)
        n = min(len(pos_seqs), len(neg_seqs))
        if len(pos_seqs) > n:
            pos_seqs = _rng.sample(pos_seqs, n)
        if len(neg_seqs) > n:
            neg_seqs = _rng.sample(neg_seqs, n)

    # pos_weight: keep all samples, weight positive loss contribution to counter imbalance
    pw = None
    if data_cfg.get("use_pos_weight", False) and len(pos_seqs) > 0:
        pw = len(neg_seqs) / len(pos_seqs)
        print(f"pos_weight = {pw:.4f} ({len(neg_seqs)} neg / {len(pos_seqs)} pos)")

    print(f"Positive: {len(pos_seqs)}  Negative: {len(neg_seqs)}")
    dataset = AMPClassificationDataset(pos_seqs, neg_seqs)
    val_n = int(len(dataset) * data_cfg.get("val_ratio", 0.05))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )
    nw = cfg["train"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=True, collate_fn=collate_supervised)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=min(nw, 2), pin_memory=True, collate_fn=collate_supervised)

    ls = cfg["train"].get("label_smoothing", 0.0)
    _run_training(model, train_loader, val_loader, cfg, device, use_fp16,
                  loss_fn=_clf_loss_fn(task, pos_weight=pw, label_smoothing=ls),
                  metric_fn=_clf_metric_fn())


def train_mic(cfg: dict, gpu: int = 0) -> None:
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and cfg["train"].get("fp16", True)

    encoder, pretrain_cfg = _load_encoder(cfg["pretrain_checkpoint"], device)
    d_model = pretrain_cfg["model"]["d_model"]
    max_len = pretrain_cfg["model"].get("max_seq_len", 52)

    head_cfg = cfg["head"].copy()
    head_type = head_cfg.pop("head_type", "mlp")

    model = JEPAMICPredictor(
        encoder=encoder,
        d_model=d_model,
        n_bacteria=N_BACTERIA,
        head_type=head_type,
        freeze_encoder=cfg["train"]["freeze_encoder"],
        **head_cfg,
    ).to(device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    data_cfg = cfg["data"]
    train_ds, val_ds, test_ds = load_grampa(
        data_cfg["grampa_csv"],
        max_len=max_len - 2,
        val_ratio=data_cfg.get("val_ratio", 0.1),
        test_ratio=data_cfg.get("test_ratio", 0.1),
        label_noise_std=data_cfg.get("label_noise_std", 0.3),
    )
    print(f"MIC train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    nw = cfg["train"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=True, collate_fn=collate_supervised)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=min(nw, 2), pin_memory=True, collate_fn=collate_supervised)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=min(nw, 2), pin_memory=True, collate_fn=collate_supervised)

    save_dir = Path(cfg["train"]["save_dir"])
    _run_training(model, train_loader, val_loader, cfg, device, use_fp16,
                  loss_fn=_mic_loss_fn(), metric_fn=_mic_metric_fn(),
                  is_mic=True)

    # --- test evaluation after training ---
    best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    test_loss, test_metrics = _evaluate(model, test_loader, device, use_fp16,
                                        _mic_loss_fn(), _mic_metric_fn(), is_mic=True)
    print(f"\nTest | loss={test_loss:.4f} | " + " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


def train_multitask(cfg: dict, gpu: int = 0) -> None:
    """Joint AMP classification + MIC regression."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and cfg["train"].get("fp16", True)

    encoder, pretrain_cfg = _load_encoder(cfg["pretrain_checkpoint"], device)
    d_model = pretrain_cfg["model"]["d_model"]
    max_len = pretrain_cfg["model"].get("max_seq_len", 52)

    head_cfg = cfg["head"].copy()
    head_type = head_cfg.pop("head_type", "mlp")

    clf = JEPAClassifier(encoder=encoder, d_model=d_model,
                         freeze_encoder=cfg["train"]["freeze_encoder"],
                         n_tox=0, **head_cfg).to(device)
    mic = JEPAMICPredictor(encoder=encoder, d_model=d_model, n_bacteria=N_BACTERIA,
                           head_type=head_type,
                           freeze_encoder=cfg["train"]["freeze_encoder"],
                           **head_cfg).to(device)

    # shared adapter params are the same object (encoder is shared), train jointly
    params = list(clf.parameters()) + [p for p in mic.parameters()
                                        if not any(p is q for q in clf.parameters())]
    trainable = sum(p.numel() for p in params if p.requires_grad)
    print(f"Trainable params (joint): {trainable:,}")

    data_cfg = cfg["data"]
    pos_seqs = load_fasta_sequences(data_cfg["pos_fasta"], max_len=max_len - 2)
    neg_seqs = _build_neg_sequences(data_cfg["neg"], max_len=max_len - 2)
    amp_ds = AMPClassificationDataset(pos_seqs, neg_seqs)
    val_n = int(len(amp_ds) * data_cfg.get("val_ratio", 0.05))
    amp_train, amp_val = random_split(amp_ds, [len(amp_ds) - val_n, val_n],
                                      generator=torch.Generator().manual_seed(42))

    mic_train, mic_val, mic_test = load_grampa(
        data_cfg["grampa_csv"], max_len=max_len - 2,
        label_noise_std=data_cfg.get("label_noise_std", 0.3),
    )

    nw = cfg["train"].get("num_workers", 4)
    bs = cfg["train"]["batch_size"]
    amp_train_l = DataLoader(amp_train, batch_size=bs, shuffle=True, num_workers=nw,
                             pin_memory=True, collate_fn=collate_supervised)
    amp_val_l   = DataLoader(amp_val,   batch_size=bs, shuffle=False, num_workers=min(nw,2),
                             pin_memory=True, collate_fn=collate_supervised)
    mic_train_l = DataLoader(mic_train, batch_size=bs, shuffle=True, num_workers=nw,
                             pin_memory=True, collate_fn=collate_supervised)
    mic_val_l   = DataLoader(mic_val,   batch_size=bs, shuffle=False, num_workers=min(nw,2),
                             pin_memory=True, collate_fn=collate_supervised)

    lam_amp = cfg["train"].get("lambda_amp", 1.0)
    lam_mic = cfg["train"].get("lambda_mic", 1.0)

    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    patience   = cfg["train"].get("patience", 10)
    save_every = cfg["train"].get("save_every", 5)
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(cfg["train"]["epochs"]):
        clf.train(); mic.train()
        total_loss = 0.0
        n_batches = 0

        for amp_batch, mic_batch in itertools.zip_longest(amp_train_l, mic_train_l):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)

            if amp_batch is not None:
                ids = amp_batch["input_ids"].to(device)
                labels = amp_batch["amp_label"].to(device)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    out = clf(ids)
                    loss = loss + lam_amp * _amp_loss(out["amp_logit"], labels)

            if mic_batch is not None:
                ids = mic_batch["input_ids"].to(device)
                bidx = mic_batch["bacteria_idx"].to(device)
                targets = mic_batch["log2_mic"].to(device)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    preds = mic(ids, bidx)
                    loss = loss + lam_mic * _mic_loss(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # validation: average of normalised AMP val loss and MIC val loss
        clf.eval(); mic.eval()
        amp_val_loss = _evaluate(clf, amp_val_l, device, use_fp16,
                                 _clf_loss_fn("amp"), _clf_metric_fn())[0]
        mic_val_loss, mic_metrics = _evaluate(mic, mic_val_l, device, use_fp16,
                                              _mic_loss_fn(), _mic_metric_fn(), is_mic=True)
        val_loss = lam_amp * amp_val_loss + lam_mic * mic_val_loss

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d} | train={avg_loss:.4f} | amp_val={amp_val_loss:.4f} | "
              f"mic_val={mic_val_loss:.4f} | pearson={mic_metrics.get('pearson', float('nan')):.4f} | lr={lr:.2e}")

        ckpt = {"epoch": epoch+1, "clf_state": clf.state_dict(),
                "mic_state": mic.state_dict(), "val_loss": val_loss, "cfg": cfg}
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(ckpt, save_dir / "best_model.pt")
            print(f"  -> Saved best checkpoint")
        else:
            no_improve += 1
        if (epoch + 1) % save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch+1:03d}.pt")
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    print("Multi-task fine-tuning done.")


# ---------------------------------------------------------------------------
# Generic training engine
# ---------------------------------------------------------------------------

def _clf_loss_fn(task: str, pos_weight: float | None = None, label_smoothing: float = 0.0):
    def fn(model, batch, device, use_fp16):
        ids = batch["input_ids"].to(device)
        amp_labels = batch["amp_label"].to(device)
        with torch.cuda.amp.autocast(enabled=use_fp16):
            out = model(ids)
            loss = _amp_loss(out["amp_logit"], amp_labels,
                             pos_weight=pos_weight, label_smoothing=label_smoothing)
            if "tox" in task and "tox_logit" in out:
                loss = loss + _tox_loss(out["tox_logit"], batch["tox_label"].to(device))
        return loss, out
    return fn


def _clf_metric_fn():
    def fn(all_logits, all_labels):
        probs = torch.sigmoid(torch.tensor(all_logits))
        preds = (probs > 0.5).float()
        labels = torch.tensor(all_labels)
        acc = (preds == labels).float().mean().item()
        # ROC-AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels.numpy(), probs.numpy())
        except Exception:
            auc = float("nan")
        return {"acc": acc, "roc_auc": auc}
    return fn


def _mic_loss_fn():
    def fn(model, batch, device, use_fp16):
        ids = batch["input_ids"].to(device)
        bidx = batch["bacteria_idx"].to(device)
        targets = batch["log2_mic"].to(device)
        with torch.cuda.amp.autocast(enabled=use_fp16):
            preds = model(ids, bidx)
            loss = _mic_loss(preds, targets)
        return loss, preds
    return fn


def _mic_metric_fn():
    def fn(all_preds, all_targets):
        p = torch.tensor(all_preds)
        t = torch.tensor(all_targets)
        rmse = ((p - t) ** 2).mean().sqrt().item()
        pearson = _pearson(p, t)
        return {"rmse": rmse, "pearson": pearson}
    return fn


def _evaluate(model, loader, device, use_fp16, loss_fn, metric_fn, is_mic: bool = False):
    model.eval()
    total_loss = 0.0
    all_out, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            loss, out = loss_fn(model, batch, device, use_fp16)
            total_loss += loss.item()
            if is_mic:
                all_out.extend(out.cpu().float().tolist())
                all_labels.extend(batch["log2_mic"].tolist())
            else:
                all_out.extend(out["amp_logit"].cpu().float().tolist())
                all_labels.extend(batch["amp_label"].tolist())
    metrics = metric_fn(all_out, all_labels)
    return total_loss / max(len(loader), 1), metrics


def _run_training(model, train_loader, val_loader, cfg, device, use_fp16,
                  loss_fn, metric_fn, is_mic: bool = False):
    lr_encoder = cfg["train"].get("lr_encoder", None)
    if lr_encoder is not None and hasattr(model, "encoder"):
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids],
             "lr": cfg["train"]["lr"]},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad],
             "lr": lr_encoder},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["train"]["weight_decay"])
        print(f"Differential LR: head={cfg['train']['lr']:.1e}  encoder={lr_encoder:.1e}")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    patience   = cfg["train"].get("patience", 10)
    save_every = cfg["train"].get("save_every", 5)
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = loss_fn(model, batch, device, use_fp16)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        val_loss, val_metrics = _evaluate(model, val_loader, device, use_fp16,
                                          loss_fn, metric_fn, is_mic=is_mic)
        lr = optimizer.param_groups[0]["lr"]
        metric_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        print(f"Epoch {epoch+1:03d} | train={train_loss:.4f} | val={val_loss:.4f} | {metric_str} | lr={lr:.2e}")

        ckpt = {"epoch": epoch+1, "model_state": model.state_dict(),
                "val_loss": val_loss, "cfg": cfg}
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(ckpt, save_dir / "best_model.pt")
            print(f"  -> Saved best checkpoint (val={val_loss:.4f})")
        else:
            no_improve += 1
        if (epoch + 1) % save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch+1:03d}.pt")
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    print(f"Training done. Best val_loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task = cfg.get("task", "amp")
    if task in ("amp", "amp+tox"):
        train_classifier(cfg, gpu=args.gpu)
    elif task == "mic":
        train_mic(cfg, gpu=args.gpu)
    elif task == "multitask":
        train_multitask(cfg, gpu=args.gpu)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
