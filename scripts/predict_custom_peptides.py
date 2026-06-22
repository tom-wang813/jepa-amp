"""
Predict AMP scores and MIC values for custom peptides using JEPA, MLM, and ESM-2 models.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import encode, PAD_ID
from src.data.supervised_dataset import BACTERIA_TO_IDX, GRAMPA_TOP20, N_BACTERIA

PEPTIDES = {
    # AMP_01: no valid mutant found
    "AMP_01_orig": "ACYCRIPACIAGERRYGTCIYQGRLWAFCC",
    # AMP_02: score 0.702 → 0.720
    "AMP_02_orig": "LPVFSTLPFAYCNIHQVCH",
    "AMP_02_opt":  "LRVFSTLPRAYCNIHQVCK",
    # AMP_03: no valid mutant found
    "AMP_03_orig": "DCYCRIPACIAGERRYGTCIYQGRLWAFCC",
    # AMP_04: no valid mutant found
    "AMP_04_orig": "GIINTLQKYYCRVRGGRCAVLSCLPKEEQIGKCSTRGRKCCRRKK",
    # AMP_05: score 0.667 → 0.719
    "AMP_05_orig": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "AMP_05_opt":  "LLGDFFRKSKEKIGKAFKRIVQRIKDFLRCCVPRTES",
    # AMP_06: score 0.739 → 0.774
    "AMP_06_orig": "DHYNCVSSGGQCLYSACPIFTKIQGTCYRGKAKCCK",
    "AMP_06_opt":  "DFYNCVSSGGQCLKSACPIFTKIWGTCYRGKAKCCK",
    # AMP_07: no valid mutant found
    "AMP_07_orig": "QPWSQCSATCGDGVRERRR",
    # AMP_08: score 0.674 → 0.822
    "AMP_08_orig": "LRRFSTMPFMFCNINNVCNF",
    "AMP_08_opt":  "LRRFCTMPSMFCNINNVCNR",
    # AMP_09 = CKS1: score 0.639 → 0.717
    "AMP_09_orig (CKS1)": "NGRKACLNPASPIVKKIIEKMLNS",
    "AMP_09_opt  (CKS1)": "RGRKACLRPASPIVKKIIEKILNS",
    # AMP_10 = CKS3: score 0.404 → 0.667
    "AMP_10_orig (CKS3)": "NGKKACLNPASPMVQKIIEKIL",
    "AMP_10_opt  (CKS3)": "RGKKACLNPASKMVQKIIKKIL",
}

TARGET_BACTERIA = ["E. coli", "S. aureus", "P. aeruginosa"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_physicochemical(seq: str) -> dict:
    charge_map = {"K": 1, "R": 1, "H": 0.1, "D": -1, "E": -1}
    charge = sum(charge_map.get(aa, 0) for aa in seq)
    kd = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }
    gravy = np.mean([kd.get(aa, 0) for aa in seq])
    return {"length": len(seq), "charge": charge, "GRAVY": round(gravy, 3)}


# ── JEPA AMP classifier ──
def load_jepa_classifier():
    from src.models.pretrain_utils import load_pretrained_encoder
    from src.models.supervised_head import JEPAClassifier

    cfg_path = PROJECT_ROOT / "configs/amp_classifier_amplify_identical.yaml"
    ckpt_path = PROJECT_ROOT / "checkpoints/amp_classifier_amplify_identical/best_model.pt"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    encoder, pretrain_cfg = load_pretrained_encoder(
        str(PROJECT_ROOT / cfg["pretrain_checkpoint"]), device
    )
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
    max_seq_len = pretrain_cfg["model"].get("max_seq_len", 52)
    return model, max_seq_len


# ── ESM-2 AMP classifier ──
def load_esm_classifier():
    from src.models.esm_head import ESMClassifier, load_esm2

    ckpt_path = PROJECT_ROOT / "checkpoints/esm2_amp_amplify_identical/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model_key = "esm2_t12_35M"
    _, alphabet, _ = load_esm2(model_key)
    batch_converter = alphabet.get_batch_converter()

    head_cfg = cfg["head"].copy()
    model = ESMClassifier(model_key=model_key, **head_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, batch_converter


# ── JEPA MIC (Transformer head) ──
def load_jepa_mic():
    from src.models.pretrain_utils import load_pretrained_encoder
    from src.models.supervised_head import JEPAMICPredictor

    cfg_path = PROJECT_ROOT / "checkpoints/formal_mic_868k_transformer/config_resolved.yaml"
    ckpt_path = PROJECT_ROOT / "checkpoints/formal_mic_868k_transformer/best_model.pt"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    encoder, pretrain_cfg = load_pretrained_encoder(
        str(PROJECT_ROOT / cfg["pretrain_checkpoint"]), device
    )
    model = JEPAMICPredictor(
        encoder=encoder,
        d_model=pretrain_cfg["model"]["d_model"],
        n_bacteria=N_BACTERIA,
        freeze_encoder=True,
        **cfg["head"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    max_seq_len = pretrain_cfg["model"].get("max_seq_len", 52)
    return model, max_seq_len


# ── MLM MIC (Transformer head) ──
def load_mlm_mic():
    from src.models.pretrain_utils import load_pretrained_encoder
    from src.models.supervised_head import JEPAMICPredictor

    cfg_path = PROJECT_ROOT / "checkpoints/formal_mic_mlm_transformer/config_resolved.yaml"
    ckpt_path = PROJECT_ROOT / "checkpoints/formal_mic_mlm_transformer/best_model.pt"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    encoder, pretrain_cfg = load_pretrained_encoder(
        str(PROJECT_ROOT / cfg["pretrain_checkpoint"]), device
    )
    model = JEPAMICPredictor(
        encoder=encoder,
        d_model=pretrain_cfg["model"]["d_model"],
        n_bacteria=N_BACTERIA,
        freeze_encoder=True,
        **cfg["head"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    max_seq_len = pretrain_cfg["model"].get("max_seq_len", 52)
    return model, max_seq_len


# ── ESM-2 MIC ──
def load_esm_mic():
    from src.models.esm_head import ESMMICPredictor, load_esm2

    ckpt_path = PROJECT_ROOT / "checkpoints/formal_esm2_mic/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model_key = "esm2_t12_35M"
    _, alphabet, _ = load_esm2(model_key)
    batch_converter = alphabet.get_batch_converter()

    head_cfg = cfg["head"].copy()
    model = ESMMICPredictor(model_key=model_key, n_bacteria=N_BACTERIA, **head_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, batch_converter


@torch.no_grad()
def predict_jepa_amp(model, seqs, max_seq_len):
    max_aa = max_seq_len - 2
    items = [torch.tensor(encode(s[:max_aa], add_special_tokens=True), dtype=torch.long) for s in seqs]
    max_len = max(t.shape[0] for t in items)
    padded = torch.full((len(items), max_len), PAD_ID, dtype=torch.long)
    for i, t in enumerate(items):
        padded[i, :t.shape[0]] = t
    padded = padded.to(device)
    out = model(padded)
    return torch.sigmoid(out["amp_logit"]).cpu().numpy()


@torch.no_grad()
def predict_esm_amp(model, batch_converter, seqs):
    data = [(f"s{i}", s) for i, s in enumerate(seqs)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    out = model(tokens)
    return torch.sigmoid(out["amp_logit"]).cpu().numpy()


@torch.no_grad()
def predict_jepa_mic(model, seqs, max_seq_len, bacteria_names):
    max_aa = max_seq_len - 2
    results = {}
    for bact in bacteria_names:
        bact_idx = BACTERIA_TO_IDX[bact]
        items = [torch.tensor(encode(s[:max_aa], add_special_tokens=True), dtype=torch.long) for s in seqs]
        max_len = max(t.shape[0] for t in items)
        padded = torch.full((len(items), max_len), PAD_ID, dtype=torch.long)
        for i, t in enumerate(items):
            padded[i, :t.shape[0]] = t
        padded = padded.to(device)
        bact_tensor = torch.full((len(seqs),), bact_idx, dtype=torch.long, device=device)
        preds = model(padded, bact_tensor).cpu().numpy()
        results[bact] = preds
    return results


@torch.no_grad()
def predict_esm_mic(model, batch_converter, seqs, bacteria_names):
    results = {}
    for bact in bacteria_names:
        bact_idx = BACTERIA_TO_IDX[bact]
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        bact_tensor = torch.full((len(seqs),), bact_idx, dtype=torch.long, device=device)
        preds = model(tokens, bact_tensor).cpu().numpy()
        results[bact] = preds
    return results


def main():
    names = list(PEPTIDES.keys())
    seqs = list(PEPTIDES.values())

    print("=" * 80)
    print("PEPTIDE PROPERTIES")
    print("=" * 80)
    for name, seq in zip(names, seqs):
        props = compute_physicochemical(seq)
        print(f"{name}")
        print(f"  Sequence: {seq}")
        print(f"  Length={props['length']}, Charge={props['charge']:+.1f}, GRAVY={props['GRAVY']:.3f}")
        print()

    # ── AMP Classification ──
    print("=" * 80)
    print("AMP CLASSIFICATION SCORES (probability of being AMP)")
    print("=" * 80)

    print("\n[JEPA-AMP classifier]")
    jepa_clf, max_len = load_jepa_classifier()
    jepa_scores = predict_jepa_amp(jepa_clf, seqs, max_len)
    for name, score in zip(names, jepa_scores):
        print(f"  {name}: {score:.4f}")
    del jepa_clf

    print("\n[ESM-2 classifier]")
    esm_clf, esm_bc = load_esm_classifier()
    esm_scores = predict_esm_amp(esm_clf, esm_bc, seqs)
    for name, score in zip(names, esm_scores):
        print(f"  {name}: {score:.4f}")
    del esm_clf

    # ── MIC Prediction ──
    print("\n" + "=" * 80)
    print("MIC PREDICTIONS (log2 MIC, lower = more potent)")
    print("=" * 80)

    # JEPA MIC
    print("\n[JEPA-AMP MIC (Transformer head)]")
    jepa_mic, max_len_mic = load_jepa_mic()
    jepa_mic_results = predict_jepa_mic(jepa_mic, seqs, max_len_mic, TARGET_BACTERIA)
    for bact in TARGET_BACTERIA:
        print(f"  {bact}:")
        for name, val in zip(names, jepa_mic_results[bact]):
            mic_ug = 2 ** val
            print(f"    {name}: log2={val:.3f} (≈{mic_ug:.1f} µg/mL)")
    del jepa_mic

    # MLM MIC
    print("\n[MLM MIC (Transformer head)]")
    mlm_mic, max_len_mlm = load_mlm_mic()
    mlm_mic_results = predict_jepa_mic(mlm_mic, seqs, max_len_mlm, TARGET_BACTERIA)
    for bact in TARGET_BACTERIA:
        print(f"  {bact}:")
        for name, val in zip(names, mlm_mic_results[bact]):
            mic_ug = 2 ** val
            print(f"    {name}: log2={val:.3f} (≈{mic_ug:.1f} µg/mL)")
    del mlm_mic

    # ESM-2 MIC
    print("\n[ESM-2 MIC (FiLM-MLP head)]")
    esm_mic, esm_mic_bc = load_esm_mic()
    esm_mic_results = predict_esm_mic(esm_mic, esm_mic_bc, seqs, TARGET_BACTERIA)
    for bact in TARGET_BACTERIA:
        print(f"  {bact}:")
        for name, val in zip(names, esm_mic_results[bact]):
            mic_ug = 2 ** val
            print(f"    {name}: log2={val:.3f} (≈{mic_ug:.1f} µg/mL)")
    del esm_mic

    # ── Summary Table ──
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Peptide':<45} {'JEPA AMP':>9} {'ESM2 AMP':>9}"
    for bact in TARGET_BACTERIA:
        header += f" | JEPA {bact[:4]:>5} MLM {bact[:4]:>5} ESM2 {bact[:4]:>5}"
    print(header)
    print("-" * len(header))
    for i, (name, seq) in enumerate(zip(names, seqs)):
        row = f"{name:<45} {jepa_scores[i]:>9.4f} {esm_scores[i]:>9.4f}"
        for bact in TARGET_BACTERIA:
            jv = jepa_mic_results[bact][i]
            mv = mlm_mic_results[bact][i]
            ev = esm_mic_results[bact][i]
            row += f" | {jv:>10.2f} {mv:>9.2f} {ev:>10.2f}"
        print(row)

    # Save JSON
    out = {}
    for i, (name, seq) in enumerate(zip(names, seqs)):
        props = compute_physicochemical(seq)
        entry = {
            "sequence": seq,
            "physicochemical": props,
            "amp_score_jepa": round(float(jepa_scores[i]), 4),
            "amp_score_esm2": round(float(esm_scores[i]), 4),
            "mic_predictions": {},
        }
        for bact in TARGET_BACTERIA:
            entry["mic_predictions"][bact] = {
                "jepa_log2mic": round(float(jepa_mic_results[bact][i]), 3),
                "mlm_log2mic": round(float(mlm_mic_results[bact][i]), 3),
                "esm2_log2mic": round(float(esm_mic_results[bact][i]), 3),
                "jepa_mic_ugml": round(float(2 ** jepa_mic_results[bact][i]), 1),
                "mlm_mic_ugml": round(float(2 ** mlm_mic_results[bact][i]), 1),
                "esm2_mic_ugml": round(float(2 ** esm_mic_results[bact][i]), 1),
            }
        out[name] = entry

    out_path = PROJECT_ROOT / "eval_results/amp_panel_predictions.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
