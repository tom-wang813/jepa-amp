# JEPA-AMP 项目当前状态与待解决问题

## 2026-04-17 更新

- `[x]` Fine-tune 任务已从随机 target block 改为 `prefix -> suffix` seq2seq。
- `[x]` 新增独立表征评估入口 `src/eval/rep_eval.py`，并在 `run_eval.py` 中加入 JEPA vs random encoder probe。
- `[x]` 数据下载脚本已补齐 UniProt / APD3 / DBAASP mirror / DRAMP 路径。
- `[x]` 新增 `scripts/prepare_amp_dataset.py`，将 raw 多源数据过滤、去重、合并为 `data/processed/amp_corpus.fasta`，并输出来源统计。
- `[x]` 数据集 split 改为稳定去重排序后再 shuffle，避免不同进程的 train/val 不一致。
- `[x]` 当前公开多源 corpus 已扩展到 `28476` 条唯一序列，并补了更稳妥的小模型配置 `configs/jepa_pretrain_28k.yaml` / `configs/finetune_28k.yaml`。
- `[ ]` 现有 checkpoint 与 `eval_results/` 仍然是旧流程产物，需按新代码重新训练/评估后再更新结论。

## 整体架构

```
Pre-train (JEPA)                Fine-tune (条件生成)
─────────────────────           ──────────────────────────────────
序列 x → 切分 context/target     context → JEPA Encoder (冻住)
                                          ↓
f_theta(context) → h_c                 Adapter (小 MLP, 可训练)
f_xi(target) → h_t (stop-grad)         ↓
g_phi(h_c) → ĥ_t                Transformer Decoder (4层, 可训练)
Loss = ||ĥ_t - h_t||²           ↓  cross-attention ← encoder memory
                                生成 target tokens (自回归)
```

### 各模块参数量
| 模块 | 类型 | 参数量 | 训练状态 |
|---|---|---|---|
| JEPA Context Encoder | 6层 Transformer Encoder | ~5.4M | Pre-train时训练，finetune冻住 |
| JEPA Target Encoder | 同上（EMA copy） | ~5.4M | 始终 stop-grad |
| Predictor | 2层 Transformer Encoder | ~1M | 只在pre-train时训练 |
| Adapter | 2层 MLP (256→64→256) | ~33K | Finetune时训练 |
| AR Decoder | 4层 Transformer Decoder | ~3.5M | Finetune时训练 |

---

## 当前结果（初版）

### Pre-training (JEPA)
- 数据：1627条 UniProt AMP序列（reviewed + TrEMBL + DRAMP）
- 200 epochs, batch=128, lr=3e-4 (cosine warmup)
- val_loss: 1.24 → **0.013**（在表示空间的 MSE）

### Fine-tuning（条件生成）
- 100 epochs, batch=128, lr=1e-4
- val_loss（cross-entropy）: 62 → **2.34**（约 epoch 30 后收敛）
- random baseline: ln(25) ≈ 3.22，模型略好于随机

### Evaluation（500条生成序列）
| 指标 | 值 |
|---|---|
| Validity | 1.000 |
| Uniqueness | 1.000 |
| Novelty | 1.000 |
| Diversity | 0.850 |
| AMP score (mean) | 0.584 |
| P(AMP) > 0.5 的比例 | 62.6% |
| 平均长度 | 26.6（训练集 32.1）|

---

## 已知问题（需要修改）

### 问题 1：生成模型结构设计不合理 ⭐ 最重要

**现状：**
- Finetune 任务是"给定 context 片段，生成 target 片段（2个block ≈ 8个token）"
- Decoder 每次只学习预测 8 个 token，然后 EOS

**问题：**
- Target 来自序列中间的随机 block，不是完整序列，语义不完整
- EOS 只在每个短片段末尾出现一次，训练信号极少 → 模型不会停
- 生成时序列长度全部堆在 max_new_tokens（30）处，长度分布完全不对

**建议修法：**
```
方案 A（改 finetune 任务）：
  不再生成"片段"，改为给定前半段序列（prefix），
  让 decoder 生成后半段（suffix）直到 EOS。
  这样 EOS 出现更自然，decoder 也能学到完整 AMP 序列的统计规律。

方案 B（改 JEPA masking）：
  Pre-train 时把序列后半段全部 mask（而不是随机 block），
  Fine-tune 时 decoder 生成整个后半段。
  更接近 seq2seq 的思路。

推荐：方案 A，改动最小，只需修改 finetune.py 和 dataset.py。
```

---

### 问题 2：数据量不足

**现状：** 1627 条序列（reviewed 1143 + TrEMBL 734 + DRAMP 16）

**问题：** 对于训练一个有效的生成模型来说偏少，导致 finetune 过快收敛到 2.34

**建议：**
- 去掉 `reviewed:true` 限制，从 UniProt 拉全量 AMP（keyword KW-0929，length≤50）
- 加入 DRAMP 完整数据集（需要手动注册下载）
- 预计可以扩充到 5000-10000 条

---

### 问题 3：AMP 分类器是弱基线

**现状：** 用理化性质特征 + LogisticRegression，负样本是真实非 AMP UniProt 序列

**问题：**
- 分类器精度有限，AMP score 不够精准
- AMP score 分布双峰（一部分低分，一部分高分），说明生成质量不均

**建议：**
- 用预训练好的 JEPA encoder 提取表示，训练一个更强的分类头（而不是手工特征）
- 或者直接调用外部工具（AMPscanner, iAMPpred）打分，更有说服力

---

### 问题 4：JEPA pre-train 和下游任务脱节

**现状：**
- JEPA pre-train：在表示空间预测被 mask 的 block
- Finetune：decoder cross-attend encoder 输出，生成 token

**问题：**
- Encoder 被冻住，Adapter 只有 33K 参数，表达能力有限
- JEPA 学到的表示质量好不好，没有独立评估

**建议：**
- 增加下游分类任务评估（用 encoder 输出做 AMP 活性分类），
  验证 JEPA 表示质量比随机初始化好
- 或者放开 encoder 部分层做 partial fine-tuning

---

## 待完成事项（优先级排序）

1. **[ ] 修改 finetune 任务（方案 A）**
   - `src/data/dataset.py`：新增 `AMPSeq2SeqDataset`，prefix/suffix 切分
   - `src/train/finetune.py`：修改 `build_generator_batch`，直接用 suffix 做 target
   - 预期效果：EOS 学习正常，长度分布合理

2. **[ ] 扩充数据**
   - `scripts/download_data.py`：去掉 `reviewed:true` 过滤
   - 目标：5000+ 条

3. **[ ] 加 JEPA 表示质量评估**
   - `src/eval/rep_eval.py`：用 encoder 输出做 AMP 分类，对比随机初始化 baseline
   - 这是 Proposal 里 "JEPA vs MLM" 对比的关键实验

4. **[ ] 增强 AMP 分类器**
   - 用 JEPA encoder 特征替换手工理化特征

---

## 文件结构
```
jepa-test/
├── src/
│   ├── data/
│   │   ├── tokenizer.py      词表（25 tokens: 20AA + PAD/UNK/BOS/EOS/MASK）
│   │   └── dataset.py        FASTA加载 + JEPA block masking
│   ├── models/
│   │   ├── encoder.py        Transformer Encoder（6层, d=256）
│   │   ├── jepa.py           JEPA预训练模型（context encoder + EMA target + predictor）
│   │   └── generator.py      ConditionalGenerator（Adapter + AR Decoder）
│   ├── train/
│   │   ├── pretrain.py       JEPA预训练主循环
│   │   └── finetune.py       生成模型微调
│   └── eval/
│       ├── metrics.py        Validity/Uniqueness/Novelty/Diversity/理化性质
│       ├── amp_classifier.py LogisticRegression AMP分类器
│       └── run_eval.py       完整评估流程（生成500条→打分→出图）
├── configs/
│   ├── jepa_pretrain.yaml    预训练超参
│   └── finetune.yaml         微调超参
├── checkpoints/
│   ├── jepa_pretrain/best_jepa.pt    epoch 196, val_loss=0.013
│   └── generator/best_generator.pt  epoch 72, val_loss=2.34
├── data/raw/
│   ├── uniprot_amps.fasta     1143条 reviewed AMP
│   ├── uniprot_amps_trembl.fasta  734条 TrEMBL AMP
│   └── dramp_amps.fasta       16条 DRAMP
└── eval_results/
    ├── eval_report.json
    ├── aa_freq_comparison.png
    ├── amp_score_dist.png
    └── length_dist.png
```

## 运行命令
```bash
# 下载数据
uv run python scripts/download_data.py

# 预训练
uv run python -m src.train.pretrain --config configs/jepa_pretrain.yaml

# 微调
uv run python -m src.train.finetune --config configs/finetune.yaml

# 评估
uv run python -m src.eval.run_eval
```
