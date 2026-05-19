"""
Fine-tuning ESM-2 for AMP classification and MIC regression.

Uses fair-esm (compatible with PyTorch 2.3) as encoder + lightweight head.

Usage:
  uv run python -m src.train.train_esm_supervised --config configs/esm2_amp_amplify_identical.yaml --gpu 1
  uv run python -m src.train.train_esm_supervised --config configs/esm2_mic.yaml --gpu 2
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ESMAMPDataset(Dataset):
    def __init__(self, pos_seqs: list[str], neg_seqs: list[str]):
        self.seqs   = pos_seqs + neg_seqs
        self.labels = [1.0] * len(pos_seqs) + [0.0] * len(neg_seqs)
    def __len__(self):  return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labels[i]


class ESMMICDataset(Dataset):
    def __init__(self, seqs: list[str], bacteria_idxs: list[int], log2_mics: list[float]):
        self.seqs = seqs
        self.bacteria_idxs = bacteria_idxs
        self.log2_mics = log2_mics
    def __len__(self):  return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.bacteria_idxs[i], self.log2_mics[i]


def _make_amp_collate(batch_converter):
    def collate(batch):
        seqs, labels = zip(*batch)
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = batch_converter(data)
        return {
            "tokens":    tokens,
            "amp_label": torch.tensor(labels, dtype=torch.float32),
        }
    return collate


def _make_mic_collate(batch_converter):
    def collate(batch):
        seqs, bact_idxs, mics = zip(*batch)
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = batch_converter(data)
        return {
            "tokens":       tokens,
            "bacteria_idx": torch.tensor(bact_idxs, dtype=torch.long),
            "log2_mic":     torch.tensor(mics, dtype=torch.float32),
        }
    return collate


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_fasta(path: str | Path, max_len: int | None = None) -> list[str]:
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    s = "".join(cur).upper()
                    if all(c in VALID_AA for c in s) and (max_len is None or len(s) <= max_len):
                        seqs.append(s)
                cur = []
            else:
                cur.append(line)
    if cur:
        s = "".join(cur).upper()
        if all(c in VALID_AA for c in s) and (max_len is None or len(s) <= max_len):
            seqs.append(s)
    return seqs


def _build_neg_sequences(neg_cfg: dict, max_len: int | None) -> list[str]:
    seqs = []
    if "fastas" in neg_cfg:
        for p in neg_cfg["fastas"]:
            seqs.extend(_load_fasta(PROJECT_ROOT / p, max_len))
        return list(dict.fromkeys(seqs))
    return _load_fasta(PROJECT_ROOT / neg_cfg["fasta"], max_len)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    return float((xm * ym).mean() / (xm.std() * ym.std()).clamp(min=1e-8))


# ---------------------------------------------------------------------------
# Generic training engine
# ---------------------------------------------------------------------------

def _run_training(model, train_loader, val_loader, cfg, device, use_fp16,
                  loss_fn, metric_fn, is_mic: bool = False):
    lr_encoder = cfg["train"].get("lr_encoder", None)
    if lr_encoder is not None and hasattr(model, "encoder"):
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        param_groups = [
            {"params": [p for p in model.parameters()
                        if p.requires_grad and id(p) not in encoder_ids],
             "lr": cfg["train"]["lr"]},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad],
             "lr": lr_encoder},
        ]
        optimizer = torch.optim.AdamW(param_groups,
                                      weight_decay=cfg["train"]["weight_decay"])
        print(f"Differential LR: head={cfg['train']['lr']:.1e}  encoder={lr_encoder:.1e}")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    save_dir = PROJECT_ROOT / cfg["train"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    patience   = cfg["train"].get("patience", 10)
    save_every = cfg["train"].get("save_every", 5)
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                loss, _ = loss_fn(model, batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss, all_out, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    loss, out = loss_fn(model, batch, device)
                val_loss += loss.item()
                if is_mic:
                    all_out.extend(out.cpu().float().tolist())
                    all_labels.extend(batch["log2_mic"].tolist())
                else:
                    all_out.extend(out["amp_logit"].cpu().float().tolist())
                    all_labels.extend(batch["amp_label"].tolist())
        val_loss /= len(val_loader)
        metrics = metric_fn(all_out, all_labels)

        lr = optimizer.param_groups[0]["lr"]
        metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch+1:03d} | train={train_loss:.4f} | val={val_loss:.4f} | "
              f"{metric_str} | lr={lr:.2e}")

        ckpt = {"epoch": epoch+1, "model_state": model.state_dict(),
                "val_loss": val_loss, "cfg": cfg}
        if val_loss < best_val:
            best_val = val_loss
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

    print(f"Training done. Best val_loss: {best_val:.4f}")


# ---------------------------------------------------------------------------
# Task: AMP classification
# ---------------------------------------------------------------------------

def train_amp(cfg: dict, gpu: int = 0):
    from sklearn.metrics import roc_auc_score
    from src.models.esm_head import ESMClassifier, load_esm2

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and cfg["train"].get("fp16", True)

    model_key = cfg.get("esm_model", "esm2_t12_35M")
    _, alphabet, _ = load_esm2(model_key)
    batch_converter = alphabet.get_batch_converter()

    head_cfg = cfg["head"].copy()
    model = ESMClassifier(
        model_key=model_key,
        freeze_encoder=cfg["train"]["freeze_encoder"],
        **head_cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ESM model: {model_key}  d_model={model.encoder.d_model}")
    print(f"Trainable params: {n_params:,}")

    data_cfg = cfg["data"]
    max_len  = data_cfg.get("max_len", None)
    pos_seqs = _load_fasta(PROJECT_ROOT / data_cfg["pos_fasta"], max_len)
    neg_seqs = _build_neg_sequences(data_cfg["neg"], max_len)

    if data_cfg.get("balance", False):
        import random as _rng
        _rng.seed(42)
        n = min(len(pos_seqs), len(neg_seqs))
        pos_seqs = _rng.sample(pos_seqs, n) if len(pos_seqs) > n else pos_seqs
        neg_seqs = _rng.sample(neg_seqs, n) if len(neg_seqs) > n else neg_seqs

    print(f"Positive: {len(pos_seqs)}  Negative: {len(neg_seqs)}")
    dataset = ESMAMPDataset(pos_seqs, neg_seqs)
    val_n = int(len(dataset) * data_cfg.get("val_ratio", 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n],
                                    generator=torch.Generator().manual_seed(42))

    collate = _make_amp_collate(batch_converter)
    nw = cfg["train"].get("num_workers", 4)
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=min(nw, 2), pin_memory=True, collate_fn=collate)

    ls = cfg["train"].get("label_smoothing", 0.0)

    def loss_fn(model, batch, device):
        tokens = batch["tokens"].to(device)
        labs   = batch["amp_label"].to(device)
        out    = model(tokens)
        if ls > 0:
            labs = labs * (1 - ls) + 0.5 * ls
        return F.binary_cross_entropy_with_logits(out["amp_logit"], labs), out

    def metric_fn(logits, labels):
        probs = torch.sigmoid(torch.tensor(logits))
        acc   = ((probs > 0.5).float() == torch.tensor(labels)).float().mean().item()
        try:
            auc = roc_auc_score(labels, probs.numpy())
        except Exception:
            auc = float("nan")
        return {"acc": acc, "roc_auc": auc}

    _run_training(model, train_loader, val_loader, cfg, device, use_fp16,
                  loss_fn, metric_fn, is_mic=False)


# ---------------------------------------------------------------------------
# Task: MIC regression
# ---------------------------------------------------------------------------

def train_mic(cfg: dict, gpu: int = 0):
    from src.models.esm_head import ESMMICPredictor, load_esm2
    from src.data.supervised_dataset import load_grampa, N_BACTERIA

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and cfg["train"].get("fp16", True)

    model_key = cfg.get("esm_model", "esm2_t12_35M")
    _, alphabet, _ = load_esm2(model_key)
    batch_converter = alphabet.get_batch_converter()

    head_cfg = cfg["head"].copy()
    model = ESMMICPredictor(
        model_key=model_key,
        n_bacteria=N_BACTERIA,
        freeze_encoder=cfg["train"]["freeze_encoder"],
        **head_cfg,
    ).to(device)
    print(f"ESM model: {model_key}  d_model={model.encoder.d_model}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    data_cfg = cfg["data"]
    max_len  = data_cfg.get("max_len", 48)
    train_ds_jepa, val_ds_jepa, test_ds_jepa = load_grampa(
        PROJECT_ROOT / data_cfg["grampa_csv"],
        max_len=max_len,
        val_ratio=data_cfg.get("val_ratio", 0.1),
        test_ratio=data_cfg.get("test_ratio", 0.1),
        label_noise_std=data_cfg.get("label_noise_std", 0.3),
    )

    def jepa_to_esm(ds):
        aa = "ACDEFGHIKLMNPQRSTVWY"
        seqs, bact_idxs, mics = [], [], []
        for item in ds:
            ids = item["input_ids"].tolist()
            seq = "".join(aa[i - 2] for i in ids if 2 <= i <= 21)
            seqs.append(seq)
            bact_idxs.append(int(item["bacteria_idx"]))
            mics.append(float(item["log2_mic"]))
        return ESMMICDataset(seqs, bact_idxs, mics)

    train_ds = jepa_to_esm(train_ds_jepa)
    val_ds   = jepa_to_esm(val_ds_jepa)
    test_ds  = jepa_to_esm(test_ds_jepa)
    print(f"MIC train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    collate = _make_mic_collate(batch_converter)
    nw = cfg["train"].get("num_workers", 4)
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=min(nw, 2), pin_memory=True, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=min(nw, 2), pin_memory=True, collate_fn=collate)

    def loss_fn(model, batch, device):
        tokens = batch["tokens"].to(device)
        bidx   = batch["bacteria_idx"].to(device)
        tgt    = batch["log2_mic"].to(device)
        preds  = model(tokens, bidx)
        return F.huber_loss(preds, tgt, delta=1.0), preds

    def metric_fn(preds, targets):
        p, t = torch.tensor(preds), torch.tensor(targets)
        return {"rmse": ((p - t)**2).mean().sqrt().item(), "pearson": _pearson(p, t)}

    _run_training(model, train_loader, val_loader, cfg, device, use_fp16,
                  loss_fn, metric_fn, is_mic=True)

    # test evaluation
    save_dir = PROJECT_ROOT / cfg["train"]["save_dir"]
    best = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch["tokens"].to(device)
            bidx   = batch["bacteria_idx"].to(device)
            p = model(tokens, bidx).cpu().float().tolist()
            preds.extend(p)
            targets.extend(batch["log2_mic"].tolist())
    p, t = torch.tensor(preds), torch.tensor(targets)
    print(f"\nTest | Pearson={_pearson(p, t):.4f}  RMSE={((p-t)**2).mean().sqrt():.4f}  n={len(preds)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    task = cfg.get("task", "amp")
    if task == "amp":
        train_amp(cfg, gpu=args.gpu)
    elif task == "mic":
        train_mic(cfg, gpu=args.gpu)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
