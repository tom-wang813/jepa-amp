"""
External zero-leakage evaluation on eLife 2025 (doi:10.7554/eLife.97330) Table 5.

28 peptides with wet-lab E. coli ATCC 25922 MIC values (μM).
None of these sequences appear in GRAMPA (0/28 overlap verified).

MIC unit conversion: μM → log2(µg/mL)
  log2_mic = log2(mic_uM * MW_g_per_mol / 1000)
where MW is computed from the amino acid sequence.

Compares:
  - JEPA-AMP MIC (Transformer head) — our model
  - ESM-2 MIC (FiLM-MLP head)      — baseline
  - AMPredictor (eLife paper)        — original paper's model, values from Table 5
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.data.tokenizer import encode, PAD_ID
from src.data.supervised_dataset import BACTERIA_TO_IDX, N_BACTERIA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Amino acid molecular weights (monoisotopic residue masses, g/mol)
AA_MW = {
    "A": 71.04, "R": 156.10, "N": 114.04, "D": 115.03, "C": 103.01,
    "Q": 128.06, "E": 129.04, "G": 57.02,  "H": 137.06, "I": 113.08,
    "L": 113.08, "K": 128.09, "M": 131.04, "F": 147.07, "P": 97.05,
    "S": 87.03,  "T": 101.05, "W": 186.08, "Y": 163.06, "V": 99.07,
}
WATER_MW = 18.01


def peptide_mw(seq: str) -> float:
    """Compute peptide molecular weight (g/mol)."""
    return sum(AA_MW.get(aa, 111.1) for aa in seq.upper()) + WATER_MW


def mic_uM_to_log2_ugml(mic_uM: float, seq: str) -> float:
    """Convert MIC in μM to log2(MIC in µg/mL)."""
    mw = peptide_mw(seq)
    mic_ugml = mic_uM * mw / 1000.0
    return np.log2(mic_ugml)


def load_jepa_mic():
    from src.models.pretrain_utils import load_pretrained_encoder
    from src.models.supervised_head import JEPAMICPredictor
    import yaml

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


def load_esm_mic():
    from src.models.esm_head import ESMMICPredictor, load_esm2

    ckpt_path = PROJECT_ROOT / "checkpoints/formal_esm2_mic/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    _, alphabet, _ = load_esm2("esm2_t12_35M")
    batch_converter = alphabet.get_batch_converter()
    model = ESMMICPredictor(model_key="esm2_t12_35M", n_bacteria=N_BACTERIA, **cfg["head"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, batch_converter


@torch.no_grad()
def predict_jepa(model, seqs, max_seq_len, bact_name):
    max_aa = max_seq_len - 2
    bact_idx = BACTERIA_TO_IDX[bact_name]
    items = [torch.tensor(encode(s[:max_aa], add_special_tokens=True), dtype=torch.long) for s in seqs]
    max_len = max(t.shape[0] for t in items)
    padded = torch.full((len(items), max_len), PAD_ID, dtype=torch.long)
    for i, t in enumerate(items):
        padded[i, :t.shape[0]] = t
    padded = padded.to(device)
    bact_tensor = torch.full((len(seqs),), bact_idx, dtype=torch.long, device=device)
    return model(padded, bact_tensor).cpu().numpy().flatten()


@torch.no_grad()
def predict_esm(model, batch_converter, seqs, bact_name):
    bact_idx = BACTERIA_TO_IDX[bact_name]
    data = [(f"s{i}", s) for i, s in enumerate(seqs)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    bact_tensor = torch.full((len(seqs),), bact_idx, dtype=torch.long, device=device)
    return model(tokens, bact_tensor).cpu().numpy().flatten()


def main():
    csv_path = PROJECT_ROOT / "data/external/elife97330_external_test.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} external sequences (eLife 2025, doi:10.7554/eLife.97330)")
    print(f"Bacterium: E. coli ATCC 25922 | MIC range: {df['mic_um'].min():.1f}–{df['mic_um'].max():.1f} μM\n")

    seqs = df["Sequence"].tolist()
    mic_uM = df["mic_um"].values

    # Convert to log2(µg/mL) — same units as GRAMPA training target
    log2_mic_true = np.array([mic_uM_to_log2_ugml(m, s) for m, s in zip(mic_uM, seqs)])
    log10_mic_true = np.log10(mic_uM)  # also keep log10(μM) for reference

    print(f"log2(MIC µg/mL) range: {log2_mic_true.min():.2f} – {log2_mic_true.max():.2f}")
    print(f"log10(MIC μM)   range: {log10_mic_true.min():.2f} – {log10_mic_true.max():.2f}\n")

    results = {}

    # AMPredictor baseline (from Table 5, predicted MIC in μM)
    ampredictor_pred_uM = df["logMIC_predicted_ampredictor"].values
    ampredictor_log2 = np.array([np.log2(m * peptide_mw(s) / 1000) for m, s in zip(ampredictor_pred_uM, seqs)])
    r_amp, _ = pearsonr(log2_mic_true, ampredictor_log2)
    rho_amp, _ = spearmanr(log2_mic_true, ampredictor_log2)
    results["AMPredictor (eLife 2025)"] = {"pearson": r_amp, "spearman": rho_amp}
    print(f"AMPredictor (paper baseline): Pearson={r_amp:.3f}, Spearman={rho_amp:.3f}")

    # JEPA-AMP MIC
    print("Loading JEPA-AMP MIC model...")
    jepa_mic, max_len = load_jepa_mic()
    jepa_preds = predict_jepa(jepa_mic, seqs, max_len, "E. coli")
    del jepa_mic
    r_jepa, _ = pearsonr(log2_mic_true, jepa_preds)
    rho_jepa, _ = spearmanr(log2_mic_true, jepa_preds)
    results["JEPA-AMP (ours)"] = {"pearson": r_jepa, "spearman": rho_jepa}
    print(f"JEPA-AMP MIC:                Pearson={r_jepa:.3f}, Spearman={rho_jepa:.3f}")

    # ESM-2 MIC
    print("Loading ESM-2 MIC model...")
    esm_mic, esm_bc = load_esm_mic()
    esm_preds = predict_esm(esm_mic, esm_bc, seqs, "E. coli")
    del esm_mic
    r_esm, _ = pearsonr(log2_mic_true, esm_preds)
    rho_esm, _ = spearmanr(log2_mic_true, esm_preds)
    results["ESM-2 (baseline)"] = {"pearson": r_esm, "spearman": rho_esm}
    print(f"ESM-2 MIC:                   Pearson={r_esm:.3f}, Spearman={rho_esm:.3f}")

    # Per-sequence output
    print("\n=== Per-sequence predictions ===")
    header = f"{'Peptide':<20} {'MIC(μM)':>8} {'logTrue':>8} {'JEPA':>7} {'ESM2':>7} {'AMPred':>8}"
    print(header)
    print("-" * len(header))
    for i, row in df.iterrows():
        print(f"{row['Peptide']:<20} {row['mic_um']:>8.1f} {log2_mic_true[i]:>8.2f}"
              f" {jepa_preds[i]:>7.2f} {esm_preds[i]:>7.2f} {ampredictor_log2[i]:>8.2f}")

    # Save results
    out = {
        "source": "eLife 2025 doi:10.7554/eLife.97330 Table 5",
        "bacterium": "E. coli ATCC 25922",
        "n_sequences": len(df),
        "grampa_overlap": 0,
        "note": "zero-leakage external test set; all sequences absent from GRAMPA training data",
        "metrics": results,
        "per_sequence": [
            {
                "peptide": df.iloc[i]["Peptide"],
                "sequence": df.iloc[i]["Sequence"],
                "mic_uM": float(mic_uM[i]),
                "log2_mic_true": float(log2_mic_true[i]),
                "jepa_pred": float(jepa_preds[i]),
                "esm2_pred": float(esm_preds[i]),
                "ampredictor_pred": float(ampredictor_log2[i]),
            }
            for i in range(len(df))
        ],
    }
    out_path = PROJECT_ROOT / "eval_results/external_elife2025_mic.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== Summary ===")
    print(f"{'Model':<30} {'Pearson':>8} {'Spearman':>9}")
    print("-" * 50)
    for name, m in results.items():
        print(f"{name:<30} {m['pearson']:>8.3f} {m['spearman']:>9.3f}")
    print(f"\nn={len(df)}, source: eLife 2025 (doi:10.7554/eLife.97330), zero GRAMPA overlap")


if __name__ == "__main__":
    main()
