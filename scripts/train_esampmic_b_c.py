"""
Route B: 3-seed ensemble of JEPA SpecFiLM (unfreeze + FiLM-MLP)
Route C: per-species separate models (one model per species, no bacteria conditioning)
Route BC: ensemble of all B+C predictions

esAMPMIC target: E.coli 0.781 | S.aureus 0.756 | P.aeruginosa 0.802

Usage:
    uv run python scripts/train_esampmic_b_c.py --gpu 0
"""
from __future__ import annotations

import csv, io, json, random, sys, urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import encode, PAD_ID
from src.models.pretrain_utils import load_pretrained_encoder
from src.models.supervised_head import JEPAMICPredictor, MLPHead
from src.models.generator import Adapter

PRETRAIN_CKPT = PROJECT_ROOT / "checkpoints" / "jepa_pretrain_868k" / "last_jepa.pt"
OUT_DIR       = PROJECT_ROOT / "eval_results" / "esampmic_bc_ensemble"
CKPT_DIR      = PROJECT_ROOT / "checkpoints" / "esampmic_bc"
ESAMPMIC_BASE = "https://raw.githubusercontent.com/chungcr/esAMPMIC/main/data"
VALID_AA      = set("ACDEFGHIKLMNPQRSTVWY")
MAX_LEN       = 42
SPECIES       = [("EC","E. coli",0), ("SA","S. aureus",1), ("PA","P. aeruginosa",2)]
ESAMPMIC_PUB  = {"E. coli":0.781, "S. aureus":0.756, "P. aeruginosa":0.802}
SEEDS         = [42, 123, 7]

BATCH=256; EPOCHS=60; LR=3e-4; LR_ENC=5e-5; WD=0.1; PATIENCE=12; NOISE=0.3
D_MODEL=384; N_BACTERIA=3; BACT_DIM=64; HIDDEN=256; DROPOUT=0.4; ADAPT_BN=64


# ── data ─────────────────────────────────────────────────────────────────────

def download_csv(prefix, split):
    url = f"{ESAMPMIC_BASE}/{prefix}_X_{split}_40.csv"
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))

def parse(rows, bact_idx):
    out = []
    for r in rows:
        seq = r.get("SEQUENCE","").strip().upper()
        try: val = float(r["NEW-CONCENTRATION"])
        except: continue
        if not seq or len(seq) > MAX_LEN-2 or not all(c in VALID_AA for c in seq):
            continue
        out.append({"seq":seq, "log2_mic":val, "bact_idx":bact_idx})
    return out

class MICDataset(Dataset):
    def __init__(self, recs, noise=0.0):
        self.recs=recs; self.noise=noise
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r=self.recs[i]
        ids=torch.tensor(encode(r["seq"]),dtype=torch.long)
        val=torch.tensor(r["log2_mic"],dtype=torch.float32)
        if self.noise>0: val=val+torch.randn(())*self.noise
        return {"input_ids":ids,
                "bacteria_idx":torch.tensor(r["bact_idx"],dtype=torch.long),
                "log2_mic":val}

def collate(batch):
    ml=max(b["input_ids"].shape[0] for b in batch)
    ids=torch.full((len(batch),ml),PAD_ID,dtype=torch.long)
    for i,b in enumerate(batch): ids[i,:b["input_ids"].shape[0]]=b["input_ids"]
    return {"input_ids":ids,
            "bacteria_idx":torch.stack([b["bacteria_idx"] for b in batch]),
            "log2_mic":torch.stack([b["log2_mic"] for b in batch])}

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# ── training ──────────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, opt=None, fp16=False, mic_key="bacteria_idx"):
    train=opt is not None
    model.train() if train else model.eval()
    scaler=torch.cuda.amp.GradScaler(enabled=fp16)
    tot,n=0.0,0
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in loader:
            ids=b["input_ids"].to(device)
            y=b["log2_mic"].to(device)
            with torch.cuda.amp.autocast(enabled=fp16):
                if mic_key in b:
                    pred=model(ids, b[mic_key].to(device))
                else:
                    pred=model(ids, torch.zeros(len(ids),dtype=torch.long,device=device))
                loss=F.huber_loss(pred,y,delta=1.0)
            if train:
                opt.zero_grad(); scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            tot+=loss.item()*len(y); n+=len(y)
    return tot/max(n,1)

def fit(model, tr, va, device, fp16, unfreeze=True, tag=""):
    if unfreeze:
        enc_ids=set(id(p) for p in model.encoder.parameters())
        opt=torch.optim.AdamW([
            {"params":[p for p in model.parameters() if id(p) in enc_ids],"lr":LR_ENC},
            {"params":[p for p in model.parameters() if id(p) not in enc_ids],"lr":LR},
        ], weight_decay=WD)
    else:
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               lr=LR, weight_decay=WD)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    best,state,wait=float("inf"),None,0
    for ep in range(EPOCHS):
        run_epoch(model,tr,device,opt,fp16)
        vl=run_epoch(model,va,device,fp16=fp16)
        sched.step()
        if vl<best-1e-4: best,state,wait=vl,{k:v.clone() for k,v in model.state_dict().items()},0
        else:
            wait+=1
            if wait>=PATIENCE: print(f"    early stop ep={ep+1}"); break
        if (ep+1)%10==0: print(f"    ep={ep+1:3d}  va={vl:.4f}  best={best:.4f}")
    model.load_state_dict(state)
    return model

@torch.no_grad()
def predict(model, loader, device, fp16=False):
    model.eval()
    preds,trues=[],[]
    for b in loader:
        ids=b["input_ids"].to(device)
        bidx=b["bacteria_idx"].to(device)
        with torch.cuda.amp.autocast(enabled=fp16):
            p=model(ids,bidx).cpu().numpy()
        preds.extend(p.tolist()); trues.extend(b["log2_mic"].numpy().tolist())
    return np.array(preds), np.array(trues)

def mets(t,p):
    if len(t)<3: return {"pearson":float("nan"),"n":len(t)}
    r,_=pearsonr(t,p); rho,_=spearmanr(t,p)
    rmse=float(np.sqrt(np.mean((t-p)**2)))
    return {"pearson":round(float(r),4),"spearman":round(float(rho),4),"rmse":round(rmse,4),"n":int(len(t))}


# ── build helpers ─────────────────────────────────────────────────────────────

def build_specfilm(device, n_bact=N_BACTERIA):
    enc,cfg=load_pretrained_encoder(str(PRETRAIN_CKPT),device)
    enc=enc.to(device)
    m=JEPAMICPredictor(
        encoder=enc, d_model=D_MODEL, n_bacteria=n_bact,
        bacteria_dim=BACT_DIM, head_type="mlp",
        hidden=HIDDEN, dropout=DROPOUT, adapter_bottleneck=ADAPT_BN,
        freeze_encoder=False,
    ).to(device)
    return m

def build_plain(device):
    """Single-species model: frozen encoder + trainable adapter + MLPHead (no bacteria)."""
    enc,_=load_pretrained_encoder(str(PRETRAIN_CKPT),device)
    enc=enc.to(device)
    # unfreeze encoder
    for p in enc.parameters(): p.requires_grad_(True)
    adapter=Adapter(D_MODEL, bottleneck=ADAPT_BN).to(device)
    head=MLPHead(D_MODEL, HIDDEN, 1, dropout=DROPOUT).to(device)

    class PerSpeciesModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=enc; self.adapter=adapter; self.head=head
        def forward(self, ids, bidx=None):
            mask=(ids==PAD_ID)
            h=self.encoder(ids); h=self.adapter(h)
            return self.head(h, mask).squeeze(-1)

    return PerSpeciesModel().to(device)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--gpu",type=int,default=0)
    args=ap.parse_args()

    device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fp16=device.type=="cuda"
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    CKPT_DIR.mkdir(parents=True,exist_ok=True)
    print(f"Device: {device}  fp16={fp16}")

    # ── download data ──────────────────────────────────────────────────────
    print("\nDownloading esAMPMIC data...")
    tr_all,va_all,te_all=[],[],[]
    te_by_sp={}
    for prefix,sp,bidx in SPECIES:
        tr=parse(download_csv(prefix,"train"),bidx)
        va=parse(download_csv(prefix,"val"),  bidx)
        te=parse(download_csv(prefix,"test"), bidx)
        tr_all+=tr; va_all+=va; te_all+=te
        te_by_sp[sp]=te
        print(f"  {sp}: train={len(tr)} val={len(va)} test={len(te)}")

    te_loader=DataLoader(MICDataset(te_all),batch_size=BATCH,
                         shuffle=False,collate_fn=collate,num_workers=0)

    # storage for ensemble predictions
    b_preds=[]   # list of (preds_array, trues_array) for each seed
    c_preds={}   # sp_name -> list of preds arrays

    # ══════════════════════════════════════════════════════════════════════
    # ROUTE B: 3-seed SpecFiLM ensemble (shared head, bacteria embedding)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("ROUTE B: 3-seed SpecFiLM (unfreeze + FiLM-MLP)")
    print("="*60)

    tr_loader=DataLoader(MICDataset(tr_all,noise=NOISE),batch_size=BATCH,
                         shuffle=True,collate_fn=collate,num_workers=0)
    va_loader=DataLoader(MICDataset(va_all),batch_size=BATCH,
                         shuffle=False,collate_fn=collate,num_workers=0)

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        set_seed(seed)
        model=build_specfilm(device)
        model=fit(model,tr_loader,va_loader,device,fp16,unfreeze=True,tag=f"B_s{seed}")
        torch.save({"model_state":model.state_dict()},CKPT_DIR/f"B_seed{seed}.pt")
        preds,trues=predict(model,te_loader,device,fp16)
        b_preds.append(preds)
        m=mets(trues,preds)
        print(f"  Seed {seed} overall Pearson={m['pearson']:.4f}")
        del model; torch.cuda.empty_cache()

    # B ensemble
    b_ens=np.mean(b_preds,axis=0)
    m_b_ens=mets(trues,b_ens)
    print(f"\n  B ensemble overall: Pearson={m_b_ens['pearson']:.4f}")
    b_per_sp={}
    for _,sp,bidx in SPECIES:
        idx=[i for i,r in enumerate(te_all) if r["bact_idx"]==bidx]
        t=trues[idx]; p=b_ens[idx]
        m=mets(t,p); b_per_sp[sp]=m
        print(f"  B ens {sp:20s}  Pearson={m['pearson']:.4f}  vs {ESAMPMIC_PUB[sp]:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # ROUTE C: per-species separate models
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("ROUTE C: per-species separate models (unfreeze, no bacteria emb)")
    print("="*60)

    c_per_sp={}
    c_preds_list=[]   # parallel to te_all order
    c_trues_list=[]

    for prefix,sp,bidx in SPECIES:
        print(f"\n  --- {sp} ---")
        sp_tr=parse(download_csv(prefix,"train"),0)  # bact_idx=0 (ignored)
        sp_va=parse(download_csv(prefix,"val"),  0)
        sp_te=parse(download_csv(prefix,"test"), 0)

        sp_tr_l=DataLoader(MICDataset(sp_tr,noise=NOISE),batch_size=BATCH,
                           shuffle=True,collate_fn=collate,num_workers=0)
        sp_va_l=DataLoader(MICDataset(sp_va),batch_size=BATCH,
                           shuffle=False,collate_fn=collate,num_workers=0)
        sp_te_l=DataLoader(MICDataset(sp_te),batch_size=BATCH,
                           shuffle=False,collate_fn=collate,num_workers=0)

        set_seed(42)
        model=build_plain(device)
        model=fit(model,sp_tr_l,sp_va_l,device,fp16,unfreeze=True,tag=f"C_{prefix}")
        torch.save({"model_state":model.state_dict()},CKPT_DIR/f"C_{prefix}.pt")

        p,t=predict(model,sp_te_l,device,fp16)
        m=mets(t,p); c_per_sp[sp]=m
        c_preds.setdefault(sp,[]).append(p)
        c_preds_list.append(p); c_trues_list.append(t)
        pub=ESAMPMIC_PUB[sp]
        print(f"  C {sp:20s}  Pearson={m['pearson']:.4f}  vs {pub:.3f}  Δ={m['pearson']-pub:+.3f}")
        del model; torch.cuda.empty_cache()

    c_all_p=np.concatenate(c_preds_list)
    c_all_t=np.concatenate(c_trues_list)
    m_c_ens=mets(c_all_t,c_all_p)
    print(f"\n  C overall (concat): Pearson={m_c_ens['pearson']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # B+C ensemble: average B_ensemble + C predictions per species
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("B+C ENSEMBLE")
    print("="*60)

    bc_per_sp={}
    for _,sp,bidx in SPECIES:
        idx=[i for i,r in enumerate(te_all) if r["bact_idx"]==bidx]
        b_sp=b_ens[idx]
        c_sp=c_per_sp_preds=c_preds[sp][0]  # single C model
        bc_sp=(b_sp+c_sp)/2
        t_sp=trues[idx]
        m=mets(t_sp,bc_sp); bc_per_sp[sp]=m
        pub=ESAMPMIC_PUB[sp]
        print(f"  BC {sp:20s}  Pearson={m['pearson']:.4f}  vs {pub:.3f}  Δ={m['pearson']-pub:+.3f}")

    # ── save results ───────────────────────────────────────────────────────
    results={
        "B_ensemble":  {"overall":m_b_ens, "per_species":b_per_sp},
        "C_per_species":{"overall":m_c_ens, "per_species":c_per_sp},
        "BC_ensemble": {"per_species":bc_per_sp},
        "esampmic_published": ESAMPMIC_PUB,
        "baseline_frozen_transformer": {
            "E. coli":0.682,"S. aureus":0.673,"P. aeruginosa":0.643
        },
        "unfreeze_mlp_seed42": {
            "E. coli":0.754,"S. aureus":0.707,"P. aeruginosa":0.680
        },
    }
    (OUT_DIR/"metrics.json").write_text(json.dumps(results,indent=2))

    # ── summary table ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    print(f"{'Method':<35} {'E.coli':>8} {'S.aureus':>9} {'P.aerug':>8}")
    print("-"*62)
    print(f"{'esAMPMIC (target)':<35} {'0.781':>8} {'0.756':>9} {'0.802':>8}")
    print(f"{'JEPA frozen+Transformer (baseline)':<35} {'0.682':>8} {'0.673':>9} {'0.643':>8}")
    print(f"{'JEPA unfreeze+FiLM (seed42)':<35} {'0.754':>8} {'0.707':>9} {'0.680':>8}")

    b_ec=b_per_sp["E. coli"]["pearson"]; b_sa=b_per_sp["S. aureus"]["pearson"]
    b_pa=b_per_sp["P. aeruginosa"]["pearson"]
    c_ec=c_per_sp["E. coli"]["pearson"]; c_sa=c_per_sp["S. aureus"]["pearson"]
    c_pa=c_per_sp["P. aeruginosa"]["pearson"]
    bc_ec=bc_per_sp["E. coli"]["pearson"]; bc_sa=bc_per_sp["S. aureus"]["pearson"]
    bc_pa=bc_per_sp["P. aeruginosa"]["pearson"]

    print(f"{'JEPA B-ensemble (3 seeds)':<35} {b_ec:>8.3f} {b_sa:>9.3f} {b_pa:>8.3f}")
    print(f"{'JEPA C per-species':<35} {c_ec:>8.3f} {c_sa:>9.3f} {c_pa:>8.3f}")
    print(f"{'JEPA B+C ensemble':<35} {bc_ec:>8.3f} {bc_sa:>9.3f} {bc_pa:>8.3f}")

    summary_lines=[
        "# B+C Ensemble vs esAMPMIC\n",
        f"| Method | E.coli | S.aureus | P.aeruginosa |",
        f"|---|---:|---:|---:|",
        f"| esAMPMIC (published) | 0.781 | 0.756 | 0.802 |",
        f"| JEPA frozen+Transformer | 0.682 | 0.673 | 0.643 |",
        f"| JEPA unfreeze+FiLM (1 seed) | 0.754 | 0.707 | 0.680 |",
        f"| **JEPA B-ensemble (3 seeds)** | **{b_ec:.3f}** | **{b_sa:.3f}** | **{b_pa:.3f}** |",
        f"| JEPA C per-species | {c_ec:.3f} | {c_sa:.3f} | {c_pa:.3f} |",
        f"| **JEPA B+C ensemble** | **{bc_ec:.3f}** | **{bc_sa:.3f}** | **{bc_pa:.3f}** |",
    ]
    (OUT_DIR/"SUMMARY.md").write_text("\n".join(summary_lines)+"\n")
    print(f"\nSaved: {OUT_DIR}/metrics.json")
    print(f"Saved: {OUT_DIR}/SUMMARY.md")


if __name__=="__main__":
    main()
