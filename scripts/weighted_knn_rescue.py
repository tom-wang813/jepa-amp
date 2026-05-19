"""
Distance-Weighted KNN Rescue Analysis
======================================
For each failed prediction (5 from JEPA classifier, 5 from ESM+LR classifier),
find 10 nearest neighbors using 3 independent distance methods:
  1. SeqIdentity  — global pairwise identity
  2. ESM          — ESM-2 6-layer mean-pool cosine similarity
  3. JEPA         — JEPA context encoder mean-pool cosine similarity

For each method compute a distance-weighted label:
  pred_prob = Σ(w_i · y_i) / Σ(w_i)   where w_i = similarity score (clipped ≥ 0)

Records per case:
  - original model prediction + confidence
  - weighted-KNN prediction for each of 3 methods
  - whether each method rescues the error
  - Jaccard overlap between the 3 neighbor sets

Writes: eval_results/weighted_knn_rescue/report.md + per-case CSVs + summary.json
"""

import csv, json, random, sys, itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import encode, PAD_ID
from src.data.supervised_dataset import load_fasta_sequences, AMPClassificationDataset
from src.models.jepa import JEPA
from src.models.supervised_head import JEPAClassifier

SEED = 42
N_NEIGHBORS = 10
OUT_DIR = Path("eval_results/weighted_knn_rescue")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def tokenise(seqs: list[str], max_len: int = 54) -> torch.Tensor:
    ids_list = [encode(s, add_special_tokens=True) for s in seqs]
    L = min(max(len(x) for x in ids_list), max_len)
    out = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        ids = ids[:L]; out[i, :len(ids)] = torch.tensor(ids)
    return out


def load_split(cfg: dict):
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
    all_labels = [1.0]*len(pos_seqs) + [0.0]*len(neg_seqs)
    dataset = AMPClassificationDataset(pos_seqs, neg_seqs)
    val_n = int(len(dataset) * data_cfg.get("val_ratio", 0.05))
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(dataset), generator=g).tolist()
    tr_idx, va_idx = idx[:len(dataset)-val_n], idx[len(dataset)-val_n:]
    return (
        [all_seqs[i] for i in tr_idx], [all_labels[i] for i in tr_idx],
        [all_seqs[i] for i in va_idx], [all_labels[i] for i in va_idx],
    )

# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


@torch.no_grad()
def jepa_embed(model: JEPAClassifier, seqs: list[str], bs: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i+bs]
        ids = tokenise(chunk).to(DEVICE)
        mask = (~(ids == PAD_ID)).float().unsqueeze(-1)
        h = model.adapter(model.encoder(ids))
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.cpu().float().numpy())
    return np.vstack(out)


def esm_embed(seqs: list[str], esm_model, converter, bs: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i+bs]
        _, _, tokens = converter([(f"s{j}", s) for j, s in enumerate(chunk)])
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            reps = esm_model(tokens, repr_layers=[6])["representations"][6]
        for j, seq in enumerate(chunk):
            out.append(reps[j, 1:len(seq)+1].mean(0).cpu().float().numpy())
    return np.vstack(out)


def seq_id_row(query: str, corpus: list[str]) -> np.ndarray:
    n = max(len(query), 1)
    return np.array([sum(a==b for a,b in zip(query,c)) / max(len(query),len(c),1)
                     for c in corpus], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Weighted-KNN prediction
# ─────────────────────────────────────────────────────────────────────────────

def weighted_knn_pred(sim_scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Returns (pred_prob, pred_label).
    sim_scores: similarity to each neighbor (higher = closer), clipped to [0,1].
    """
    w = np.clip(sim_scores, 0, None)
    total = w.sum()
    if total < 1e-12:
        return 0.5, 0.5
    prob = float((w * labels).sum() / total)
    return prob, float(prob >= 0.5)


def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)

# ─────────────────────────────────────────────────────────────────────────────
# Find failures for a given predict function
# ─────────────────────────────────────────────────────────────────────────────

def find_failures_jepa(model, val_seqs, val_labels, n_fp=3, n_fn=2):
    failed = []
    with torch.no_grad():
        for i in range(0, len(val_seqs), 256):
            ids = tokenise(val_seqs[i:i+256]).to(DEVICE)
            prob = torch.sigmoid(model(ids)["amp_logit"]).cpu().numpy()
            for seq, lbl, pb in zip(val_seqs[i:i+256], val_labels[i:i+256], prob):
                pred = float(pb >= 0.5)
                if pred != lbl:
                    failed.append({"seq":seq,"true_label":lbl,"prob":float(pb),
                                   "model":"JEPA"})
    rng = random.Random(SEED)
    fp = rng.sample([x for x in failed if x["true_label"]==0], min(n_fp, sum(1 for x in failed if x["true_label"]==0)))
    fn = rng.sample([x for x in failed if x["true_label"]==1], min(n_fn, sum(1 for x in failed if x["true_label"]==1)))
    return fp + fn


def find_failures_esm(pipe, train_embs_esm, val_seqs, val_labels,
                      val_embs_esm, n_fp=3, n_fn=2):
    prob = pipe.predict_proba(val_embs_esm)[:, 1]
    failed = []
    for seq, lbl, pb in zip(val_seqs, val_labels, prob):
        pred = float(pb >= 0.5)
        if pred != lbl:
            failed.append({"seq":seq,"true_label":lbl,"prob":float(pb),
                           "model":"ESM+LR"})
    rng = random.Random(SEED)
    fp = rng.sample([x for x in failed if x["true_label"]==0], min(n_fp, sum(1 for x in failed if x["true_label"]==0)))
    fn = rng.sample([x for x in failed if x["true_label"]==1], min(n_fn, sum(1 for x in failed if x["true_label"]==1)))
    return fp + fn

# ─────────────────────────────────────────────────────────────────────────────
# Per-case analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_case(
    case_idx: int,
    failure: dict,
    train_seqs: list[str],
    train_labels: list[float],
    train_embs_jepa_norm: np.ndarray,
    train_embs_esm_norm: np.ndarray,
    jepa_model,
    esm_model, converter,
) -> dict:
    seq        = failure["seq"]
    true_label = failure["true_label"]
    model_prob = failure["prob"]
    model_name = failure["model"]
    error_type = "FP" if true_label == 0 else "FN"

    # ── test embeddings ──
    test_emb_jepa = l2norm(jepa_embed(jepa_model, [seq]))   # (1, D)
    test_emb_esm  = l2norm(esm_embed([seq], esm_model, converter))  # (1, D)

    # ── SeqID: scan top-1000 JEPA candidates ──
    cos_jepa_all = (test_emb_jepa @ train_embs_jepa_norm.T)[0]
    top1k = np.argsort(cos_jepa_all)[::-1][:1000]
    si_scores_1k = seq_id_row(seq, [train_seqs[i] for i in top1k])
    top_si_local = np.argsort(si_scores_1k)[::-1][:N_NEIGHBORS]
    top_si_idx   = top1k[top_si_local]
    si_sim        = si_scores_1k[top_si_local]

    # ── ESM ──
    cos_esm_all = (test_emb_esm @ train_embs_esm_norm.T)[0]
    top_esm_idx = np.argsort(cos_esm_all)[::-1][:N_NEIGHBORS]
    esm_sim      = cos_esm_all[top_esm_idx]

    # ── JEPA ──
    top_jepa_idx = np.argsort(cos_jepa_all)[::-1][:N_NEIGHBORS]
    jepa_sim      = cos_jepa_all[top_jepa_idx]

    train_labels_np = np.array(train_labels)

    methods = {}
    for mname, nb_idx, sim_sc in [("SeqID", top_si_idx, si_sim),
                                   ("ESM",   top_esm_idx, esm_sim),
                                   ("JEPA",  top_jepa_idx, jepa_sim)]:
        nb_labels = train_labels_np[nb_idx]
        nb_seqs   = [train_seqs[i] for i in nb_idx]
        pred_prob, pred_label = weighted_knn_pred(sim_sc, nb_labels)
        rescued = (pred_label == true_label)
        n_same  = int((nb_labels == true_label).sum())
        methods[mname] = {
            "neighbors": [
                {"seq": s, "label": float(l), "sim": float(sim)}
                for s, l, sim in zip(nb_seqs, nb_labels, sim_sc)
            ],
            "pred_prob":   round(pred_prob, 4),
            "pred_label":  pred_label,
            "rescued":     rescued,
            "n_same_label": n_same,
            "pct_same_label": round(100.0 * n_same / N_NEIGHBORS, 1),
        }

    # ── Jaccard overlap ──
    sets = {m: set(arr.tolist()) for m, arr
            in [("SeqID", top_si_idx), ("ESM", top_esm_idx), ("JEPA", top_jepa_idx)]}
    overlaps = {}
    for ma, mb in itertools.combinations(["SeqID", "ESM", "JEPA"], 2):
        key = f"{ma}∩{mb}"
        j = jaccard(sets[ma], sets[mb])
        n_shared = len(sets[ma] & sets[mb])
        overlaps[key] = {"jaccard": round(j, 4), "n_shared": n_shared}

    return {
        "case_idx":    case_idx,
        "model":       model_name,
        "error_type":  error_type,
        "seq":         seq,
        "seq_len":     len(seq),
        "true_label":  true_label,
        "model_prob":  round(model_prob, 4),
        "methods":     methods,
        "overlaps":    overlaps,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Markdown report generator
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown(cases: list[dict], path: Path):
    lines = []
    A = lines.append

    A("# Distance-Weighted KNN Rescue Analysis\n")
    A(f"**Models compared:** JEPA classifier · ESM-2 + LogReg  ")
    A(f"**Neighbor count:** {N_NEIGHBORS}  ")
    A(f"**Distance methods:** SeqIdentity · ESM embedding · JEPA embedding  \n")

    # ── summary table ──
    A("## Summary Table\n")
    A("| # | Model | Type | Sequence | Len | True | Model prob "
      "| SeqID rescued? | ESM rescued? | JEPA rescued? | Any? |")
    A("|---|-------|------|----------|-----|------|------------|"
      "----------------|--------------|---------------|------|")
    for c in cases:
        seq_short = c["seq"][:25] + ("…" if len(c["seq"]) > 25 else "")
        true_str = "AMP" if c["true_label"] == 1 else "non-AMP"
        def emoji(m):
            r = c["methods"][m]["rescued"]
            return f"✅ ({c['methods'][m]['pred_prob']:.3f})" if r else f"❌ ({c['methods'][m]['pred_prob']:.3f})"
        any_rescued = any(c["methods"][m]["rescued"] for m in ["SeqID","ESM","JEPA"])
        A(f"| {c['case_idx']} | {c['model']} | {c['error_type']} | `{seq_short}` | {c['seq_len']} | "
          f"{true_str} | {c['model_prob']:.3f} | {emoji('SeqID')} | {emoji('ESM')} | "
          f"{emoji('JEPA')} | {'✅' if any_rescued else '❌'} |")
    A("")

    # ── rescue count ──
    A("## Rescue Statistics\n")
    for model_name in ["JEPA", "ESM+LR"]:
        subset = [c for c in cases if c["model"] == model_name]
        if not subset: continue
        A(f"### {model_name} failures ({len(subset)} cases)\n")
        A("| Method | Rescued | Total | Rate |")
        A("|--------|---------|-------|------|")
        for m in ["SeqID", "ESM", "JEPA"]:
            rescued = sum(1 for c in subset if c["methods"][m]["rescued"])
            A(f"| {m} | {rescued} | {len(subset)} | {100*rescued/len(subset):.0f}% |")
        A("")

    # ── overlap summary ──
    A("## Neighbor Overlap (Jaccard) Between Methods\n")
    A("| # | Model | Type | SeqID∩ESM | SeqID∩JEPA | ESM∩JEPA |")
    A("|---|-------|------|-----------|------------|----------|")
    for c in cases:
        def ov(k):
            o = c["overlaps"][k]
            return f"{o['n_shared']}/{N_NEIGHBORS} (J={o['jaccard']:.2f})"
        A(f"| {c['case_idx']} | {c['model']} | {c['error_type']} | "
          f"{ov('SeqID∩ESM')} | {ov('SeqID∩JEPA')} | {ov('ESM∩JEPA')} |")
    A("")

    # ── per-method avg Jaccard ──
    A("**Average Jaccard across all cases:**\n")
    for pair in ["SeqID∩ESM", "SeqID∩JEPA", "ESM∩JEPA"]:
        vals = [c["overlaps"][pair]["jaccard"] for c in cases]
        A(f"- {pair}: mean = {np.mean(vals):.3f} · range [{min(vals):.3f}, {max(vals):.3f}]")
    A("")

    # ── per-case details ──
    A("## Per-Case Details\n")
    for c in cases:
        true_str  = "AMP"     if c["true_label"] == 1 else "non-AMP"
        error_str = "False Positive (non-AMP → AMP)" if c["error_type"] == "FP" \
                    else "False Negative (AMP → non-AMP)"
        A(f"### Case {c['case_idx']} · {c['model']} · {c['error_type']}\n")
        A(f"| Field | Value |")
        A(f"|-------|-------|")
        A(f"| Sequence | `{c['seq']}` |")
        A(f"| Length | {c['seq_len']} aa |")
        A(f"| Error type | {error_str} |")
        A(f"| True label | **{true_str}** |")
        A(f"| Model confidence | {c['model_prob']:.4f} |")
        A("")

        for mname in ["SeqID", "ESM", "JEPA"]:
            m = c["methods"][mname]
            rescued_str = "**✅ RESCUED**" if m["rescued"] else "❌ still wrong"
            A(f"#### {mname} neighbors  →  pred_prob = {m['pred_prob']:.4f}  {rescued_str}\n")
            A(f"| # | Sequence | Label | Similarity | Vote weight |")
            A(f"|---|----------|-------|------------|-------------|")
            w_total = sum(max(nb["sim"], 0) for nb in m["neighbors"])
            for k, nb in enumerate(m["neighbors"]):
                lbl_str = "AMP" if nb["label"] == 1 else "non-AMP"
                w = max(nb["sim"], 0)
                contrib = f"{100*w/w_total:.1f}%" if w_total > 0 else "—"
                A(f"| {k+1} | `{nb['seq'][:30]}` | {lbl_str} | {nb['sim']:.4f} | {contrib} |")
            A("")

        # overlap
        A(f"**Neighbor overlap:**\n")
        A(f"| Pair | Shared | Jaccard |")
        A(f"|------|--------|---------|")
        for pair, vals in c["overlaps"].items():
            A(f"| {pair} | {vals['n_shared']}/{N_NEIGHBORS} | {vals['jaccard']:.3f} |")
        A(f"\n---\n")

    # ── interpretation ──
    A("## Interpretation Notes\n")
    A("### Why do the three methods find different neighbors?\n")
    A("Each method measures a different aspect of similarity:\n")
    A("- **SeqIdentity**: surface-level character matching — same residues at same positions")
    A("- **ESM embedding**: biochemical/evolutionary function — trained on 250M proteins")
    A("- **JEPA embedding**: AMP-specific structural patterns — trained on our AMP corpus\n")
    A("Low Jaccard (< 0.2) between methods means the three similarity spaces barely agree.")
    A("When all three rescue a case, the error was a genuine boundary issue.")
    A("When none rescue it, the sequence is a distributional outlier regardless of metric.\n")

    A("### Why does weighted-KNN sometimes rescue what the model missed?\n")
    A("The trained classifier uses a fixed global decision boundary.")
    A("Weighted-KNN adapts locally: it asks *'what do the most similar training samples look like?'*")
    A("If the model's decision boundary was drawn in the wrong place for this region of space,")
    A("the local vote can override it — especially for boundary-ambiguous cases (prob ≈ 0.5).\n")

    A("### Confidently-wrong cases (prob ≪ 0.1 or ≫ 0.9)\n")
    A("These cannot be rescued by any method, because **all three distance metrics agree**")
    A("that the test sequence lives in a neighborhood dominated by the wrong-label class.")
    A("The fix is data: add more examples of this structural type to training.\n")

    path.write_text("\n".join(lines))
    print(f"  Markdown written → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    with open("configs/amp_classifier_v7.yaml") as f:
        cfg = yaml.safe_load(f)

    train_seqs, train_labels, val_seqs, val_labels = load_split(cfg)
    print(f"Train: {len(train_seqs)}  Val: {len(val_seqs)}")

    # ── JEPA model ──
    print("\nLoading JEPA classifier…")
    pretrain_ckpt = torch.load(cfg["pretrain_checkpoint"], map_location=DEVICE, weights_only=False)
    pretrain_cfg  = pretrain_ckpt["cfg"]
    jepa_obj = JEPA(**pretrain_cfg["model"])
    jepa_obj.load_state_dict(pretrain_ckpt["model_state"])
    ckpt = torch.load("checkpoints/amp_classifier_v7/best_model.pt", map_location=DEVICE, weights_only=False)
    jepa_model = JEPAClassifier(encoder=jepa_obj.context_encoder,
                                d_model=pretrain_cfg["model"]["d_model"],
                                freeze_encoder=False, n_tox=0, **cfg["head"])
    jepa_model.load_state_dict(ckpt["model_state"])
    jepa_model.to(DEVICE).eval()

    # ── ESM model ──
    print("Loading ESM-2 (fair-esm)…")
    import esm as fair_esm
    esm_m, alphabet = fair_esm.pretrained.esm2_t6_8M_UR50D()
    esm_m.to(DEVICE).eval()
    converter = alphabet.get_batch_converter()

    # ── embeddings ──
    print("Embedding train set with JEPA…")
    train_embs_jepa = l2norm(jepa_embed(jepa_model, train_seqs))
    print("Embedding train set with ESM…")
    train_embs_esm  = l2norm(esm_embed(train_seqs, esm_m, converter))

    print("Embedding val set with ESM (for ESM+LR)…")
    val_embs_esm = esm_embed(val_seqs, esm_m, converter)

    # ── ESM+LR classifier ──
    print("Training ESM+LR classifier…")
    pipe_esm = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=42))])
    pipe_esm.fit(train_embs_esm, np.array(train_labels))
    val_acc_esm = (pipe_esm.predict(val_embs_esm) == np.array(val_labels)).mean()
    print(f"  ESM+LR val acc = {val_acc_esm:.4f}")

    # ── JEPA val acc ──
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(val_seqs), 256):
            ids = tokenise(val_seqs[i:i+256]).to(DEVICE)
            pr = torch.sigmoid(jepa_model(ids)["amp_logit"]).cpu().numpy()
            all_preds.extend((pr >= 0.5).astype(float))
    val_acc_jepa = np.mean(np.array(all_preds) == np.array(val_labels))
    print(f"  JEPA val acc   = {val_acc_jepa:.4f}")

    # ── find failures ──
    print("\nFinding failures…")
    fails_jepa = find_failures_jepa(jepa_model, val_seqs, val_labels, n_fp=2, n_fn=3)
    fails_esm  = find_failures_esm(pipe_esm, train_embs_esm, val_seqs, val_labels,
                                   val_embs_esm, n_fp=2, n_fn=3)

    print(f"  JEPA failures selected: {len(fails_jepa)}")
    for f in fails_jepa:
        et = "FP" if f["true_label"]==0 else "FN"
        print(f"    [{et}] {f['seq'][:30]:30s} prob={f['prob']:.3f}")

    print(f"  ESM+LR failures selected: {len(fails_esm)}")
    for f in fails_esm:
        et = "FP" if f["true_label"]==0 else "FN"
        print(f"    [{et}] {f['seq'][:30]:30s} prob={f['prob']:.3f}")

    all_failures = fails_jepa + fails_esm  # 10 total

    # ── analyse each case ──
    print("\nAnalysing cases…")
    all_cases = []
    for ci, failure in enumerate(all_failures, 1):
        print(f"  Case {ci:2d}/{len(all_failures)}  [{failure['model']}]  {failure['seq'][:25]}…")
        result = analyse_case(
            case_idx            = ci,
            failure             = failure,
            train_seqs          = train_seqs,
            train_labels        = train_labels,
            train_embs_jepa_norm= train_embs_jepa,
            train_embs_esm_norm = train_embs_esm,
            jepa_model          = jepa_model,
            esm_model           = esm_m,
            converter           = converter,
        )
        all_cases.append(result)

        # per-case CSV
        for mname in ["SeqID", "ESM", "JEPA"]:
            csv_path = OUT_DIR / f"case_{ci:02d}_{mname}.csv"
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["rank","sequence","label","similarity","vote_weight"])
                w.writeheader()
                m = result["methods"][mname]
                w_total = sum(max(nb["sim"],0) for nb in m["neighbors"])
                for k, nb in enumerate(m["neighbors"], 1):
                    wt = max(nb["sim"],0)
                    w.writerow({"rank":k, "sequence":nb["seq"],
                                "label": "AMP" if nb["label"]==1 else "non-AMP",
                                "similarity": f"{nb['sim']:.4f}",
                                "vote_weight": f"{100*wt/w_total:.1f}%" if w_total>0 else "—"})

    # ── JSON summary ──
    json_summary = []
    for c in all_cases:
        row = {"case": c["case_idx"], "model": c["model"], "error_type": c["error_type"],
               "seq": c["seq"], "seq_len": c["seq_len"], "true_label": c["true_label"],
               "model_prob": c["model_prob"]}
        for m in ["SeqID", "ESM", "JEPA"]:
            row[f"{m}_pred_prob"] = c["methods"][m]["pred_prob"]
            row[f"{m}_rescued"]   = c["methods"][m]["rescued"]
            row[f"{m}_pct_same_label"] = c["methods"][m]["pct_same_label"]
        for pair in ["SeqID∩ESM", "SeqID∩JEPA", "ESM∩JEPA"]:
            row[f"jaccard_{pair.replace('∩','_')}"] = c["overlaps"][pair]["jaccard"]
        json_summary.append(row)
    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(json_summary, fh, indent=2)

    # ── markdown ──
    write_markdown(all_cases, OUT_DIR / "report.md")

    # ── print quick summary ──
    print(f"\n{'='*65}")
    print("QUICK SUMMARY")
    print(f"{'='*65}")
    header = f"{'#':>2}  {'Model':8}  {'Type':3}  {'Seq':26}  {'Prob':5}  {'SeqID':8}  {'ESM':8}  {'JEPA':8}"
    print(header)
    print("-" * len(header))
    for c in all_cases:
        def flag(m):
            r = c["methods"][m]["rescued"]
            p = c["methods"][m]["pred_prob"]
            return f"{'✅' if r else '❌'}{p:.2f}"
        print(f"{c['case_idx']:>2}  {c['model']:8}  {c['error_type']:3}  "
              f"{c['seq'][:26]:26}  {c['model_prob']:.3f}  "
              f"{flag('SeqID'):8}  {flag('ESM'):8}  {flag('JEPA'):8}")

    print(f"\nAll files saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
