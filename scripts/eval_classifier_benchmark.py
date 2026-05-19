"""
Evaluate our JEPA AMP classifier on the AMPlify held-out test set.
Reports ROC-AUC, accuracy, precision, recall, F1.

Usage:
  uv run python scripts/eval_classifier_benchmark.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR    = PROJECT_ROOT / "data/benchmarks"

MODELS = {
    "JEPA-v2 (balanced 3k)": {
        "ckpt": PROJECT_ROOT / "checkpoints/amp_classifier_868k/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/amp_classifier_868k.yaml",
    },
    "JEPA-v3 (pos_weight 868k+3k neg)": {
        "ckpt": PROJECT_ROOT / "checkpoints/amp_classifier_868k_v3/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/amp_classifier_868k_v3.yaml",
    },
    "JEPA-v4 (pos_weight 868k+8.7k neg)": {
        "ckpt": PROJECT_ROOT / "checkpoints/amp_classifier_868k_v4/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/amp_classifier_868k_v4.yaml",
    },
    "JEPA-v5 (pos_weight 868k+50k shuffled)": {
        "ckpt": PROJECT_ROOT / "checkpoints/amp_classifier_868k_v5/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/amp_classifier_868k_v5.yaml",
    },
    "JEPA-v6 (868k balanced, noleak)": {
        "ckpt": PROJECT_ROOT / "checkpoints/amp_classifier_v6/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/amp_classifier_v6.yaml",
    },
    "JEPA-v7 (UniProt+curated neg, noleak)": {
        "ckpt": PROJECT_ROOT / "checkpoints/amp_classifier_v7/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/amp_classifier_v7.yaml",
    },
    "JEPA-AMPlify-identical (3k, apple-to-apple)": {
        "ckpt": PROJECT_ROOT / "checkpoints/amp_classifier_amplify_identical/best_model.pt",
        "cfg":  PROJECT_ROOT / "configs/amp_classifier_amplify_identical.yaml",
    },
}

ESM_MODELS = {
    "ESM2-AMPlify-identical (3k)": {
        "ckpt":      PROJECT_ROOT / "checkpoints/esm2_amp_amplify_identical/best_model.pt",
        "model_key": "esm2_t12_35M",
    },
    "ESM2-v6 (868k, noleak)": {
        "ckpt":      PROJECT_ROOT / "checkpoints/esm2_amp_v6/best_model.pt",
        "model_key": "esm2_t12_35M",
    },
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_fasta(path: Path) -> list[str]:
    seqs = []
    cur = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    s = "".join(cur).upper()
                    if all(c in VALID_AA for c in s):
                        seqs.append(s)
                cur = []
            else:
                cur.append(line)
    if cur:
        s = "".join(cur).upper()
        if all(c in VALID_AA for c in s):
            seqs.append(s)
    return seqs


def load_esm_model(ckpt_path: Path, model_key: str, device: torch.device):
    from src.models.esm_head import ESMClassifier, load_esm2

    _, alphabet, _ = load_esm2(model_key)
    batch_converter = alphabet.get_batch_converter()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    head_cfg = cfg["head"].copy()
    model = ESMClassifier(model_key=model_key, **head_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, batch_converter


@torch.no_grad()
def score_esm_sequences(model, batch_converter, seqs: list[str], device: torch.device,
                         batch_size: int = 128) -> np.ndarray:
    scores = []
    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i:i+batch_size]
        data = [(f"s{j}", s) for j, s in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model(tokens)
        probs = torch.sigmoid(out["amp_logit"]).cpu().float().numpy()
        scores.extend(probs.tolist())
    return np.array(scores)


def load_model(ckpt_path: Path, cfg_path: Path, device: torch.device):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.jepa import JEPA
    from src.models.supervised_head import JEPAClassifier

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    pretrain_ckpt = torch.load(
        PROJECT_ROOT / cfg["pretrain_checkpoint"], map_location=device, weights_only=False
    )
    pretrain_cfg = pretrain_ckpt["cfg"]
    jepa = JEPA(**pretrain_cfg["model"])
    jepa.load_state_dict(pretrain_ckpt["model_state"])
    encoder = jepa.context_encoder

    model = JEPAClassifier(
        encoder=encoder,
        d_model=pretrain_cfg["model"]["d_model"],
        freeze_encoder=True,
        n_tox=0,
        **cfg["head"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, pretrain_cfg["model"].get("max_seq_len", 52)


@torch.no_grad()
def score_sequences(model, seqs: list[str], max_seq_len: int,
                    device: torch.device, batch_size: int = 256) -> np.ndarray:
    from src.data.tokenizer import encode
    from src.data.supervised_dataset import collate_supervised

    scores = []
    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i:i+batch_size]
        # filter by length
        max_aa = max_seq_len - 2
        batch_seqs = [s[:max_aa] for s in batch_seqs]
        items = [{"input_ids": torch.tensor(encode(s, add_special_tokens=True), dtype=torch.long),
                  "amp_label": torch.tensor(0.0)} for s in batch_seqs]
        batch = collate_supervised(items)
        ids = batch["input_ids"].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model(ids)
        probs = torch.sigmoid(out["amp_logit"]).cpu().float().numpy()
        scores.extend(probs.tolist())
    return np.array(scores)


def evaluate(scores: np.ndarray, labels: np.ndarray) -> dict:
    preds = (scores >= 0.5).astype(int)
    return {
        "ROC-AUC":   round(roc_auc_score(labels, scores), 4),
        "Accuracy":  round(accuracy_score(labels, preds), 4),
        "Precision": round(precision_score(labels, preds, zero_division=0), 4),
        "Recall":    round(recall_score(labels, preds, zero_division=0), 4),
        "F1":        round(f1_score(labels, preds, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(labels, preds), 4),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load AMPlify benchmark test set
    pos_seqs = load_fasta(BENCH_DIR / "amplify_test_pos.fasta")
    neg_seqs = load_fasta(BENCH_DIR / "amplify_test_neg.fasta")
    print(f"AMPlify test set: {len(pos_seqs)} AMP, {len(neg_seqs)} non-AMP\n")

    all_seqs   = pos_seqs + neg_seqs
    all_labels = np.array([1]*len(pos_seqs) + [0]*len(neg_seqs))

    results = {}
    for name, spec in MODELS.items():
        if not spec["ckpt"].exists():
            print(f"[SKIP] {name} — checkpoint not found")
            continue
        print(f"Evaluating: {name} …")
        model, max_seq_len = load_model(spec["ckpt"], spec["cfg"], device)
        scores = score_sequences(model, all_seqs, max_seq_len, device)
        metrics = evaluate(scores, all_labels)
        results[name] = metrics
        print("  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))

    # Published baselines (from AMPlify paper Table 1, same test set)
    published = {
        "AMPlify ensemble":          {"ROC-AUC": 0.9837, "Accuracy": 0.9371, "Precision": None, "Recall": 0.9293, "F1": 0.9366, "MCC": None},
        "AMPlify single (best)":     {"ROC-AUC": 0.9798, "Accuracy": 0.9257, "Precision": None, "Recall": 0.9257, "F1": 0.9257, "MCC": None},
        "AMPlify single (worst)":    {"ROC-AUC": 0.9727, "Accuracy": 0.9210, "Precision": None, "Recall": 0.9090, "F1": 0.9200, "MCC": None},
        "AMP Scanner (re-trained)":  {"ROC-AUC": 0.9740, "Accuracy": 0.9066, "Precision": None, "Recall": 0.9114, "F1": 0.9070, "MCC": None},
        "iAMPpred (original)":       {"ROC-AUC": 0.8070, "Accuracy": 0.7401, "Precision": None, "Recall": 0.8790, "F1": 0.7718, "MCC": None},
    }

    # Print comparison table
    all_results = {**published, **results}
    metrics_keys = ["ROC-AUC", "Accuracy", "F1", "MCC"]

    print("\n" + "=" * 70)
    print("AMP CLASSIFICATION BENCHMARK (AMPlify test set)")
    print("=" * 70)
    header = f"{'Model':<35}" + "".join(f"{k:>10}" for k in metrics_keys)
    print(header)
    print("-" * 70)
    for name, m in all_results.items():
        row = f"{'* ' + name if name in results else name:<35}"
        for k in metrics_keys:
            v = m.get(k)
            row += f"{'N/A':>10}" if v is None else f"{v:>10.4f}"
        marker = " ← OURS" if name in results else ""
        print(row + marker)
    print("-" * 70)
    print("* = our models  |  Published numbers from respective papers")

    # ESM-2 models
    esm_results = {}
    for name, spec in ESM_MODELS.items():
        if not spec["ckpt"].exists():
            print(f"[SKIP] {name} — checkpoint not found")
            continue
        print(f"Evaluating: {name} …")
        esm_model, batch_converter = load_esm_model(spec["ckpt"], spec["model_key"], device)
        scores = score_esm_sequences(esm_model, batch_converter, all_seqs, device)
        metrics = evaluate(scores, all_labels)
        esm_results[name] = metrics
        print("  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))

    if esm_results:
        print("\n" + "=" * 70)
        print("ESM-2 MODELS — AMPlify test set")
        print("=" * 70)
        print(header)
        print("-" * 70)
        for name, m in esm_results.items():
            row = f"{'* ' + name:<35}"
            for k in metrics_keys:
                v = m.get(k)
                row += f"{'N/A':>10}" if v is None else f"{v:>10.4f}"
            print(row + " ← ESM-2")
        print("-" * 70)

    # APD3 independent test (sequences not in any AMPlify split)
    apd3_path = BENCH_DIR / "apd3_independent_test.fasta"
    if apd3_path.exists():
        apd3_pos = load_fasta(apd3_path)
        apd3_neg = neg_seqs  # reuse AMPlify negatives as control
        print(f"\nAPD3 independent test: {len(apd3_pos)} AMP (not in AMPlify), {len(apd3_neg)} neg")
        apd3_seqs   = apd3_pos + apd3_neg
        apd3_labels = np.array([1]*len(apd3_pos) + [0]*len(apd3_neg))

        apd3_results = {}
        for name, spec in MODELS.items():
            if not spec["ckpt"].exists():
                continue
            model, max_seq_len = load_model(spec["ckpt"], spec["cfg"], device)
            scores = score_sequences(model, apd3_seqs, max_seq_len, device)
            metrics = evaluate(scores, apd3_labels)
            apd3_results[name] = metrics

        print("\n" + "=" * 70)
        print("AMP CLASSIFICATION — APD3 INDEPENDENT TEST (cross-dataset generalisation)")
        print("=" * 70)
        print(header)
        print("-" * 70)
        for name, m in apd3_results.items():
            row = f"{'* ' + name:<35}"
            for k in metrics_keys:
                v = m.get(k)
                row += f"{'N/A':>10}" if v is None else f"{v:>10.4f}"
            print(row)
        print("-" * 70)
    else:
        apd3_results = {}
        print(f"\n[SKIP] APD3 independent test — {apd3_path} not found")

    # Save
    out = {
        "amplify_test": {"ours": results, "esm": esm_results,
                         "published": {k: {kk: vv for kk, vv in v.items() if vv is not None}
                                        for k, v in published.items()}},
        "apd3_independent": apd3_results,
    }
    with open(PROJECT_ROOT / "eval_results/classifier_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to eval_results/classifier_benchmark.json")


if __name__ == "__main__":
    main()
