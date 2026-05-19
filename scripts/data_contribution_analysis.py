"""
Data Contribution Analysis for Failed AMP Predictions
======================================================
For 5 misclassified validation samples, find 10 nearest neighbors
using THREE independent methods:
  1. Sequence Identity  — simple global identity (proxy for BLAST)
  2. ESM-2 embeddings   — facebook/esm2_t6_8M_UR50D mean-pool representation
  3. JEPA embeddings    — our trained context encoder mean-pool representation

Then compute three contribution scores for each method's neighbors:
  A. Gradient Matching  — cosine similarity of loss gradients
  B. Influence Function — gradient dot-product / curvature approximation
  C. Data Shapley       — exact Shapley via 2^N subsets, KNN utility

Outputs are saved to:
  eval_results/contribution_analysis/
    summary.json          — all numeric results
    case_{i}_method_{m}.csv — per-case per-method neighbor table
    neighbor_overlap.csv  — pairwise Jaccard between 3 methods per case

Usage:
  uv run python scripts/data_contribution_analysis.py
"""

import csv
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import encode, PAD_ID
from src.data.supervised_dataset import (
    load_fasta_sequences,
    AMPClassificationDataset,
)
from src.models.jepa import JEPA
from src.models.supervised_head import JEPAClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CKPT_DIR   = Path("checkpoints/amp_classifier_v7")
CONFIG     = Path("configs/amp_classifier_v7.yaml")
OUT_DIR    = Path("eval_results/contribution_analysis")
N_FAILED   = 5
N_NEIGHBORS = 10
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: dict) -> JEPAClassifier:
    pretrain_ckpt = torch.load(cfg["pretrain_checkpoint"], map_location=DEVICE, weights_only=False)
    pretrain_cfg  = pretrain_ckpt["cfg"]
    jepa = JEPA(**pretrain_cfg["model"])
    jepa.load_state_dict(pretrain_ckpt["model_state"])
    encoder = jepa.context_encoder

    ckpt = torch.load(CKPT_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    model = JEPAClassifier(
        encoder=encoder,
        d_model=pretrain_cfg["model"]["d_model"],
        freeze_encoder=False,
        n_tox=0,
        **cfg["head"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(f"Loaded v7 checkpoint: epoch={ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.4f}")
    return model


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenise(seqs: list[str], max_len: int = 54) -> torch.Tensor:
    ids_list = [encode(s, add_special_tokens=True) for s in seqs]
    L = min(max(len(x) for x in ids_list), max_len)
    out = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        ids = ids[:L]
        out[i, :len(ids)] = torch.tensor(ids)
    return out


# ---------------------------------------------------------------------------
# Embedding methods
# ---------------------------------------------------------------------------

@torch.no_grad()
def jepa_embed(model: JEPAClassifier, seqs: list[str], batch_size: int = 256) -> np.ndarray:
    all_embs = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        ids = tokenise(chunk).to(DEVICE)
        pad_mask = (ids == PAD_ID)
        h = model.encoder(ids)
        h = model.adapter(h)
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        all_embs.append(pooled.cpu().float().numpy())
    return np.vstack(all_embs)


def esm_embed(seqs: list[str], batch_size: int = 64) -> np.ndarray:
    """ESM-2 6-layer 8M via fair-esm (not HuggingFace)."""
    import esm as fair_esm
    print("    Loading ESM-2 (fair-esm)…")
    model_esm, alphabet = fair_esm.pretrained.esm2_t6_8M_UR50D()
    model_esm.to(DEVICE).eval()
    converter = alphabet.get_batch_converter()

    all_embs = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        data = [(f"s{j}", s) for j, s in enumerate(chunk)]
        _, _, tokens = converter(data)
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            out = model_esm(tokens, repr_layers=[6], return_contacts=False)
        reps = out["representations"][6]          # (B, L+2, D)
        for j, seq in enumerate(chunk):
            L = len(seq)
            emb = reps[j, 1:L+1, :].mean(0).cpu().float().numpy()
            all_embs.append(emb)
    return np.vstack(all_embs)


def seq_identity_row(query: str, corpus: list[str]) -> np.ndarray:
    """Returns identity scores of query against every corpus sequence."""
    scores = np.zeros(len(corpus), dtype=np.float32)
    for j, c in enumerate(corpus):
        n = max(len(query), len(c))
        if n == 0:
            scores[j] = 1.0
        else:
            scores[j] = sum(a == b for a, b in zip(query, c)) / n
    return scores


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def compute_gradient(model: JEPAClassifier, seq: str, label: float) -> torch.Tensor:
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad()
    ids = tokenise([seq]).to(DEVICE)
    lbl = torch.tensor([label], device=DEVICE)
    out = model(ids)
    loss = F.binary_cross_entropy_with_logits(out["amp_logit"], lbl)
    loss.backward()
    grads = []
    for p in model.parameters():
        grads.append(p.grad.detach().view(-1) if p.grad is not None
                     else torch.zeros(p.numel(), device=DEVICE))
    model.zero_grad()
    return torch.cat(grads)


def gradient_matching(g_test: torch.Tensor, g_train: torch.Tensor) -> float:
    return F.cosine_similarity(g_test.unsqueeze(0), g_train.unsqueeze(0)).item()


def influence(g_test: torch.Tensor, g_train: torch.Tensor) -> float:
    """
    Sign convention: positive = neighbor supports correct prediction.
    H^{-1} approximated as 1/mean(g_test^2) (isotropic curvature).
    """
    eps = 1e-8
    c = g_test.pow(2).mean().clamp(min=eps)
    dot = (g_test * g_train).sum()
    norm = g_test.norm().clamp(min=eps) * g_train.norm().clamp(min=eps)
    return (dot / (c * norm)).item()


# ---------------------------------------------------------------------------
# Data Shapley — exact 2^N
# ---------------------------------------------------------------------------

def data_shapley(
    test_emb: np.ndarray,
    train_embs: np.ndarray,
    train_labels: np.ndarray,
    true_label: float,
    n_neighbors: int = 10,
) -> np.ndarray:
    """
    Positive Shapley = neighbor supports predicting the CORRECT class.
    Utility = P(correct class) under weighted KNN using subset.
    """
    N = n_neighbors
    diff = train_embs - test_emb[None, :]
    dists = np.linalg.norm(diff, axis=1)
    w_all = np.exp(-dists / (dists.mean() + 1e-8))

    def utility(idx: list[int]) -> float:
        if not idx:
            return 0.0
        w = w_all[idx]; lbl = train_labels[idx]
        p_amp = float((w * lbl).sum() / (w.sum() + 1e-12))
        return p_amp if true_label == 1 else (1.0 - p_amp)

    factorials = [1] * (N + 1)
    for i in range(1, N + 1):
        factorials[i] = factorials[i - 1] * i

    sv = np.zeros(N)
    for i in range(N):
        others = [j for j in range(N) if j != i]
        for k in range(N):
            for subset in itertools.combinations(others, k):
                s = len(subset)
                weight = factorials[s] * factorials[N - s - 1] / factorials[N]
                sv[i] += weight * (utility(list(subset) + [i]) - utility(list(subset)))
    return sv


# ---------------------------------------------------------------------------
# Jaccard overlap between two neighbor sets
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Contribution scores for one neighbor set
# ---------------------------------------------------------------------------

def compute_contributions(
    model: JEPAClassifier,
    test_seq: str,
    true_label: float,
    neighbor_seqs: list[str],
    neighbor_labels: list[float],
    test_emb_jepa: np.ndarray,
    train_embs_jepa: np.ndarray,
) -> dict:
    """Returns dict with lists of grad_match, influence, shapley scores."""
    # Gradients
    g_test = compute_gradient(model, test_seq, true_label)
    gm_scores, inf_scores = [], []
    for seq, lbl in zip(neighbor_seqs, neighbor_labels):
        g_nb = compute_gradient(model, seq, lbl)
        gm_scores.append(gradient_matching(g_test, g_nb))
        inf_scores.append(influence(g_test, g_nb))

    # Shapley (using JEPA embeddings for the local distance metric)
    nb_embs = train_embs_jepa  # (N, D), already filtered to these neighbors
    shap = data_shapley(
        test_emb=test_emb_jepa,
        train_embs=nb_embs,
        train_labels=np.array(neighbor_labels, dtype=np.float32),
        true_label=true_label,
        n_neighbors=len(neighbor_seqs),
    )
    return {"grad_match": gm_scores, "influence": inf_scores, "shapley": shap.tolist()}


# ---------------------------------------------------------------------------
# Pretty-print one neighbor table
# ---------------------------------------------------------------------------

def print_neighbor_table(
    method: str,
    neighbor_seqs: list[str],
    neighbor_labels: list[float],
    sim_scores: list[float],
    scores: dict,
    true_label: float,
):
    correct_lbl = "AMP" if true_label == 1 else "non-AMP"
    print(f"\n  ── Method: {method}  (true_label={correct_lbl}) ──")
    print(f"  {'#':>2}  {'Sequence':<24}  {'Lbl':>4}  {'Sim':>6}  "
          f"{'GradMatch':>9}  {'Influence':>9}  {'Shapley':>7}  {'3-way'}")
    print(f"  {'--':>2}  {'-'*24}  {'----':>4}  {'------':>6}  "
          f"{'-'*9}  {'-'*9}  {'-'*7}  {'-----'}")
    for k in range(len(neighbor_seqs)):
        seq   = neighbor_seqs[k]
        lbl   = neighbor_labels[k]
        sim   = sim_scores[k]
        gm    = scores["grad_match"][k]
        inf   = scores["influence"][k]
        shp   = scores["shapley"][k]
        lbl_str = "AMP" if lbl == 1 else "non"
        signs = [np.sign(gm), np.sign(inf), np.sign(shp)]
        agree = "✓" if len(set(signs)) == 1 else "✗"
        print(f"  {k+1:>2}  {seq[:24]:<24}  {lbl_str:>4}  {sim:>6.3f}  "
              f"{gm:>+9.4f}  {inf:>+9.4f}  {shp:>+7.4f}  {agree}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    model = load_model(cfg)

    # --- load and split data (same as training) ---
    data_cfg = cfg["data"]
    pos_seqs = load_fasta_sequences(data_cfg["pos_fasta"], max_len=50)
    neg_seqs = []
    if "fastas" in data_cfg["neg"]:
        for fa in data_cfg["neg"]["fastas"]:
            neg_seqs.extend(load_fasta_sequences(fa, max_len=50))
    else:
        neg_seqs = load_fasta_sequences(data_cfg["neg"]["fasta"], max_len=50)

    rng = random.Random(42)
    n = min(len(pos_seqs), len(neg_seqs))
    if len(pos_seqs) > n: pos_seqs = rng.sample(pos_seqs, n)
    if len(neg_seqs) > n: neg_seqs = rng.sample(neg_seqs, n)

    all_seqs   = pos_seqs + neg_seqs
    all_labels = [1.0] * len(pos_seqs) + [0.0] * len(neg_seqs)
    print(f"\nDataset: {len(pos_seqs)} pos + {len(neg_seqs)} neg = {len(all_seqs)} total")

    dataset = AMPClassificationDataset(pos_seqs, neg_seqs)
    val_n   = int(len(dataset) * data_cfg.get("val_ratio", 0.05))
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=g).tolist()
    train_idx = indices[: len(dataset) - val_n]
    val_idx   = indices[len(dataset) - val_n :]

    train_seqs   = [all_seqs[i]   for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_seqs     = [all_seqs[i]   for i in val_idx]
    val_labels   = [all_labels[i] for i in val_idx]
    print(f"Train: {len(train_seqs)}  Val: {len(val_seqs)}")

    # --- find failures ---
    print("\nRunning validation predictions…")
    failed = []
    bs = 256
    with torch.no_grad():
        for start in range(0, len(val_seqs), bs):
            chunk_s = val_seqs[start : start + bs]
            chunk_l = val_labels[start : start + bs]
            ids  = tokenise(chunk_s).to(DEVICE)
            out  = model(ids)
            prob = torch.sigmoid(out["amp_logit"]).cpu().numpy()
            pred = (prob > 0.5).astype(float)
            for seq, lbl, pr, pb in zip(chunk_s, chunk_l, pred, prob):
                if pr != lbl:
                    failed.append({"seq": seq, "true_label": lbl,
                                   "pred_label": pr, "prob": float(pb)})

    fp = [x for x in failed if x["true_label"] == 0]
    fn = [x for x in failed if x["true_label"] == 1]
    print(f"Failures — FP: {len(fp)}, FN: {len(fn)}, total: {len(failed)}")

    rng2 = random.Random(SEED)
    sel_fp = rng2.sample(fp, min(len(fp), (N_FAILED + 1) // 2))
    sel_fn = rng2.sample(fn, min(len(fn), N_FAILED - len(sel_fp)))
    selected = (sel_fp + sel_fn)[:N_FAILED]
    print(f"Selected {len(selected)} failures  "
          f"({sum(x['true_label']==0 for x in selected)} FP, "
          f"{sum(x['true_label']==1 for x in selected)} FN)")

    # --- pre-compute embeddings ---
    print("\n[1/3] Computing JEPA embeddings for all training data…")
    train_embs_jepa = jepa_embed(model, train_seqs)
    train_embs_jepa_norm = l2norm(train_embs_jepa)
    print(f"  JEPA shape: {train_embs_jepa.shape}")

    print("[2/3] Computing ESM-2 embeddings for all training data…")
    train_embs_esm = esm_embed(train_seqs)
    train_embs_esm_norm = l2norm(train_embs_esm)
    print(f"  ESM shape: {train_embs_esm.shape}")

    train_labels_np = np.array(train_labels, dtype=np.float32)

    METHOD_NAMES = ["SeqIdentity", "ESM", "JEPA"]
    all_results  = []
    overlap_rows = []

    for fi, failure in enumerate(selected):
        test_seq   = failure["seq"]
        true_label = failure["true_label"]
        error_type = "FP" if true_label == 0 else "FN"
        print(f"\n{'='*72}")
        print(f"[{fi+1}/{len(selected)}] {error_type}  |  "
              f"seq={test_seq[:35]}{'…' if len(test_seq)>35 else ''}  "
              f"len={len(test_seq)}")
        print(f"  true={'AMP' if true_label==1 else 'non-AMP'}  "
              f"pred={'AMP' if failure['pred_label']==1 else 'non-AMP'}  "
              f"prob={failure['prob']:.3f}")

        # ---- find neighbors by each method ----
        # SeqIdentity: search in top-500 JEPA candidates first (speed)
        test_emb_jepa = jepa_embed(model, [test_seq])
        test_emb_jepa_norm = l2norm(test_emb_jepa)
        cos_jepa = (test_emb_jepa_norm @ train_embs_jepa_norm.T)[0]
        top500_idx = np.argsort(cos_jepa)[::-1][:500]

        print("\n  [SeqIdentity] scanning top-500 JEPA candidates…")
        cand_seqs = [train_seqs[i] for i in top500_idx]
        seq_id_scores_500 = seq_identity_row(test_seq, cand_seqs)
        top_si_local = np.argsort(seq_id_scores_500)[::-1][:N_NEIGHBORS]
        top_si_idx   = top500_idx[top_si_local]
        si_scores    = seq_id_scores_500[top_si_local]

        test_emb_esm = esm_embed([test_seq])
        test_emb_esm_norm = l2norm(test_emb_esm)
        cos_esm = (test_emb_esm_norm @ train_embs_esm_norm.T)[0]
        top_esm_idx  = np.argsort(cos_esm)[::-1][:N_NEIGHBORS]
        esm_scores   = cos_esm[top_esm_idx]

        top_jepa_idx = np.argsort(cos_jepa)[::-1][:N_NEIGHBORS]
        jepa_scores  = cos_jepa[top_jepa_idx]

        neighbor_sets = {
            "SeqIdentity": (top_si_idx,  si_scores),
            "ESM":         (top_esm_idx, esm_scores),
            "JEPA":        (top_jepa_idx, jepa_scores),
        }

        # ---- Jaccard overlap between the 3 methods ----
        sets_by_method = {m: set(top_si_idx.tolist()) if m == "SeqIdentity"
                          else set(top_esm_idx.tolist()) if m == "ESM"
                          else set(top_jepa_idx.tolist())
                          for m in METHOD_NAMES}
        print("\n  Neighbor overlap (Jaccard):")
        for ma, mb in itertools.combinations(METHOD_NAMES, 2):
            j = jaccard(sets_by_method[ma], sets_by_method[mb])
            n_common = len(sets_by_method[ma] & sets_by_method[mb])
            print(f"    {ma} ∩ {mb}: {n_common}/{N_NEIGHBORS} shared, "
                  f"Jaccard={j:.3f}")
            overlap_rows.append({
                "case": fi + 1,
                "error_type": error_type,
                "test_seq": test_seq[:30],
                "method_A": ma,
                "method_B": mb,
                "n_shared": n_common,
                "jaccard": round(j, 4),
            })

        # ---- contribution analysis per method ----
        case_results = {"failure": failure, "methods": {}}

        for method_name, (nb_idx, sim_sc) in neighbor_sets.items():
            nb_seqs   = [train_seqs[i]   for i in nb_idx]
            nb_labels = [train_labels[i] for i in nb_idx]
            nb_embs   = train_embs_jepa[nb_idx]  # JEPA emb used for Shapley dist

            print(f"\n  [{method_name}] computing contributions…")
            scores = compute_contributions(
                model      = model,
                test_seq   = test_seq,
                true_label = true_label,
                neighbor_seqs   = nb_seqs,
                neighbor_labels = nb_labels,
                test_emb_jepa   = test_emb_jepa[0],
                train_embs_jepa = nb_embs,
            )
            print_neighbor_table(method_name, nb_seqs, nb_labels,
                                 sim_sc.tolist(), scores, true_label)

            # aggregate
            n_pos = sum(v > 0 for v in scores["grad_match"])
            lbl_same = sum(l == true_label for l in nb_labels)
            case_results["methods"][method_name] = {
                "neighbors": [
                    {"seq": s, "label": l, "sim": float(sim),
                     "grad_match": float(gm), "influence": float(inf),
                     "shapley": float(shp)}
                    for s, l, sim, gm, inf, shp in zip(
                        nb_seqs, nb_labels, sim_sc,
                        scores["grad_match"], scores["influence"], scores["shapley"])
                ],
                "n_support":    n_pos,
                "n_conflict":   N_NEIGHBORS - n_pos,
                "n_same_label": lbl_same,
                "pct_same_label": round(100 * lbl_same / N_NEIGHBORS, 1),
                "mean_grad_match": round(float(np.mean(scores["grad_match"])), 4),
                "mean_shapley":    round(float(np.mean(scores["shapley"])), 4),
            }

            # save per-case per-method CSV
            csv_path = OUT_DIR / f"case_{fi+1}_{method_name}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "rank", "sequence", "label", "similarity",
                    "grad_match", "influence", "shapley", "3way_agree"])
                w.writeheader()
                for k, row in enumerate(case_results["methods"][method_name]["neighbors"]):
                    signs = [np.sign(row["grad_match"]),
                             np.sign(row["influence"]),
                             np.sign(row["shapley"])]
                    w.writerow({
                        "rank": k + 1,
                        "sequence": row["seq"],
                        "label": "AMP" if row["label"] == 1 else "non-AMP",
                        "similarity": f"{row['sim']:.4f}",
                        "grad_match": f"{row['grad_match']:+.4f}",
                        "influence": f"{row['influence']:+.4f}",
                        "shapley": f"{row['shapley']:+.4f}",
                        "3way_agree": "Y" if len(set(signs)) == 1 else "N",
                    })

        all_results.append(case_results)

    # -----------------------------------------------------------------------
    # Global summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("GLOBAL SUMMARY")
    print(f"{'='*72}")

    for method_name in METHOD_NAMES:
        print(f"\n  ── {method_name} ──")
        for stat in ["mean_grad_match", "mean_shapley",
                     "n_support", "pct_same_label"]:
            vals = [r["methods"][method_name][stat]
                    for r in all_results if method_name in r["methods"]]
            print(f"    {stat:20s}: {vals}")

    # Three-way metric agreement per method
    print("\n  Three-way agreement (GradMatch, Influence, Shapley sign equal):")
    for method_name in METHOD_NAMES:
        agree = 0; total = 0
        for r in all_results:
            if method_name not in r["methods"]:
                continue
            for nb in r["methods"][method_name]["neighbors"]:
                signs = [np.sign(nb["grad_match"]),
                         np.sign(nb["influence"]),
                         np.sign(nb["shapley"])]
                agree += 1 if len(set(signs)) == 1 else 0
                total += 1
        print(f"    {method_name:12s}: {agree}/{total} ({100*agree/total:.1f}%)")

    # Overlap summary
    print("\n  Average Jaccard between methods across all cases:")
    for ma, mb in itertools.combinations(METHOD_NAMES, 2):
        vals = [row["jaccard"] for row in overlap_rows
                if row["method_A"] == ma and row["method_B"] == mb]
        print(f"    {ma} ∩ {mb}: mean={np.mean(vals):.3f}  "
              f"range=[{min(vals):.3f}, {max(vals):.3f}]")

    # Observation about same-label neighbors and Shapley
    print("\n  Same-label → positive Shapley?  (per method)")
    for method_name in METHOD_NAMES:
        same_lbl_pos  = 0; same_lbl_total  = 0
        diff_lbl_pos  = 0; diff_lbl_total  = 0
        for r in all_results:
            if method_name not in r["methods"]:
                continue
            tl = r["failure"]["true_label"]
            for nb in r["methods"][method_name]["neighbors"]:
                shp = nb["shapley"]
                if nb["label"] == tl:
                    same_lbl_total += 1
                    if shp > 0: same_lbl_pos += 1
                else:
                    diff_lbl_total += 1
                    if shp > 0: diff_lbl_pos += 1
        print(f"    {method_name:12s}: "
              f"same-label→pos Shapley={same_lbl_pos}/{same_lbl_total}, "
              f"diff-label→pos Shapley={diff_lbl_pos}/{diff_lbl_total}")

    # -----------------------------------------------------------------------
    # Save JSON summary
    # -----------------------------------------------------------------------
    summary_out = []
    for fi, r in enumerate(all_results):
        row = {
            "case": fi + 1,
            "error_type": "FP" if r["failure"]["true_label"] == 0 else "FN",
            "test_seq":   r["failure"]["seq"],
            "true_label": int(r["failure"]["true_label"]),
            "pred_prob":  round(r["failure"]["prob"], 4),
        }
        for method_name in METHOD_NAMES:
            if method_name in r["methods"]:
                m = r["methods"][method_name]
                row[f"{method_name}_n_support"]    = m["n_support"]
                row[f"{method_name}_pct_same_lbl"] = m["pct_same_label"]
                row[f"{method_name}_mean_gradmatch"]= m["mean_grad_match"]
                row[f"{method_name}_mean_shapley"]  = m["mean_shapley"]
        summary_out.append(row)

    json_path = OUT_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump({"cases": summary_out, "overlap": overlap_rows,
                   "raw_results": [
                       {**r, "failure": r["failure"],
                        "methods": {
                            mn: {k: v for k, v in md.items() if k != "neighbors"}
                            for mn, md in r["methods"].items()
                        }}
                       for r in all_results]}, f, indent=2)

    overlap_csv = OUT_DIR / "neighbor_overlap.csv"
    with open(overlap_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(overlap_rows[0].keys()))
        w.writeheader(); w.writerows(overlap_rows)

    print(f"\nResults saved to {OUT_DIR}/")
    print(f"  {json_path.name}")
    print(f"  {overlap_csv.name}")
    print(f"  case_{{1-{len(selected)}}}_{{method}}.csv")


if __name__ == "__main__":
    main()
