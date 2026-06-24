# JEPA-AMP Paper Status
_Last updated: 2026-06-06_

---

## 一句话定位

**JEPA-AMP** 是第一个将 Joint-Embedding Predictive Architecture（JEPA）用于抗菌肽（AMP）
表示学习的工作，提供了统一backbone下的分类、MIC回归、和定量条件生成三个任务的系统评估，
并在同源感知MIC基准（QMAP）上达到SOTA水平，同时提供三种生成范式（AR、NAR、Masked
Diffusion）的对比分析。

---

## 我们做了什么（按时间顺序）

### Phase 1：基础预训练（JEPA Encoder）
- 在 **868,724** 条AMP序列上预训练JEPA
- 数据来源：UniProt reviewed (1,035) + TrEMBL (597) + APD3 (2,872) + AMPSphere (~860k)
- 长度 5–50 AA，均值 35.4 AA
- 架构：8层 Transformer Encoder，d=384，nhead=8，ff=1536，~14.2M参数
- 训练目标：预测被 mask 的 block 在**嵌入空间**中的表示（而非序列空间）
- 最终 val_loss = 0.2442（嵌入空间 MSE）

### Phase 2：下游任务评估

#### 2a. AMP分类（AMPlify benchmark）
- 用JEPA冻结embedding + 分类头，与AMPlify相同训练数据
- AUROC 0.958, MCC 0.802, F1 0.899
- 与ESM2相当（AUROC 0.963），低于AMPlify ensemble（0.984）
- APD3 独立测试集迁移：AUROC 0.944, MCC 0.758

#### 2b. MIC回归（GRAMPA）
- 868k 伪标签预训练 + GRAMPA 真实MIC微调
- JEPA Transformer head: Pearson 0.640, RMSE 0.627
- JEPA FiLM-MLP head: Pearson 0.622, RMSE 0.619
- ESM-2 (35M) baseline: Pearson 0.554, RMSE 0.635
- **JEPA 比 ESM2 提升 Pearson +0.086**

#### 2c. QMAP同源感知基准（最强证据）
- 用官方 qmap-benchmark==0.1.1，5个预定义同源感知split
- Full E.coli: 0.512 ± 0.009（3 seeds）—— 与Cai et al. 2025 SOTA（0.52）持平
- High-eff E.coli: **0.388 ± 0.013**（SOTA 0.29，**提升 34%**）
- HC50: 0.327 ± 0.004（需要task-specific head）

### Phase 3：条件生成器（AR，Dual-Pathway v4）

架构（ConditionalGeneratorV4）：
```
JEPA Encoder (frozen, 14.2M)
  → Adapter (MLP 384→128→384, 100K)
  → Condition Encoder ([len,charge,GRAVY] → 128d, 17K)
  → Dual-pathway AdaLN AR Decoder (4层, 10.7M可训)
     - 每层 AdaLN：用 cond_emb 调制 scale/shift
     - Cross-attn memory：在encoder输出前prepend一个 condition token
     - Context dropout 0.15：强迫模型依赖condition
     - CFG dropout 0.30：支持 Classifier-Free Guidance
总参数：25M（10.8M可训练）
```

**物理化学条件控制（formal evaluation，10个target，n=200/target）：**

| Variant | Charge R² | GRAVY R² | Length R² |
|---|---|---|---|
| 弱条件基线 (v2) | -11.83 | -1.02 | -1.91 |
| AdaLN-only (v3) | -17.90 | -0.83 | -9.54 |
| **Dual-pathway (v4，我们)** | **0.866** | 0.020 | -1.71 |

- 电荷控制有效（R²=0.866），两个baseline完全失败
- GRAVY 和 Length 控制失败 → 作为 limitation 明确报告

**MIC条件生成（oracle-mediated）：**

| 场景 | JEPA Δ(E.coli) | ESM2 Δ(E.coli) |
|---|---|---|
| 广谱强效 | -0.25 log₂ | -0.29 log₂ |
| 全菌失活 | **+0.52 log₂** | +0.44 log₂ |
| 物种选择性 | ≈0（失败） | ≈0（失败） |

两个scorer方向一致，说明不是oracle circularity。物种选择性failure作为negative result报告。

### Phase 4：消融实验（进行中）

- v4_no_aux（去掉辅助physicochemical loss）— 🔄 训练中
- v4_no_dropout（去掉CFG dropout）— ⏳ 排队
- 目的：证明dual-pathway哪个组件是必要的

### Phase 5：新增生成范式（进行中）

基于同一JEPA encoder，对比三种生成范式：

| 范式 | 架构 | 状态 | 特点 |
|---|---|---|---|
| AR Decoder（v4）| AdaLN AR Transformer | ✅ 已训练 | 逐token，现有方法 |
| NAR Decoder | 双向Transformer+Length Head | ⏳ 排队 | 并行解码，推理快10-50x |
| Masked Diffusion | MDLM风格+Timestep Embedding | ⏳ 排队 | 迭代去噪，不同�ductive bias |

### Phase 6：表示质量分析（已完成）

**JEPA vs ESM2 冻结嵌入对比：**

| 指标 | JEPA | ESM2 |
|---|---|---|
| k-NN MIC Pearson (k=5, frozen) | 0.598 | 0.618 |
| Linear Probe AUROC (frozen) | 0.888 | 0.935 |
| Silhouette (AMP/non-AMP) | 0.043 | 0.063 |
| **Fine-tuned MIC Pearson** | **0.640** | **0.554** |

**关键insight**：冻结ESM2略好于冻结JEPA，但fine-tune后JEPA反超ESM2。
说明JEPA表示更"可塑"——预训练目标（嵌入空间预测）创造了更适合下游适配的表示，
而非最终优化的判别表示。

### Phase 7：其他分析（已完成/进行中）

- UMAP可视化（5张图，JEPA vs ESM2，AMP分类+MIC+generated overlay）✅
- MC-Dropout uncertainty：**formal checkpoint上无改善**（Δ RMSE = +0.0028），
  旧"5.4%提升"claim来自不同checkpoint，已从paper中移除 ✅
- Charge interpolation（连续charge sweep）— ⏳ 等GPU空闲

---

## 整体Model结构（一句话版）

JEPA-AMP是一个**统一的AMP表示backbone**：在868k序列上用JEPA自监督预训练，
然后通过不同的head（分类头、回归头、生成decoder）对接三个任务，
共享同一个冻结的8层Transformer encoder。

---

## 当前锁定的主要结果

| Claim | 数字 | 状态 |
|---|---|---|
| AMP分类 AUROC | 0.958 | ✅ LOCKED |
| GRAMPA MIC Pearson | 0.640 | ✅ LOCKED |
| QMAP High-eff E.coli | 0.388 | ✅ LOCKED |
| 电荷条件控制 R² | 0.866 | ✅ LOCKED |
| GRAVY/Length控制失败 | R²≈0/negative | ✅ LOCKED（negative result） |
| 物种选择性失败 | delta≈0 | ✅ LOCKED（negative result） |
| MC-Dropout无改善 | Δ=+0.0028 | ✅ LOCKED（旧claim移除） |

---

## 可以发conference吗？

### 答案：**可以，但要选对venue和timing**

**计算工作本身目前够发：**
- JEPA用于肽是**第一次**，novelty明确
- QMAP high-efficiency结果（0.388 vs SOTA 0.29）是真正有竞争力的数字
- 三种生成范式对比（等NAR+Diffusion训完）是方法贡献
- Honest evaluation（明确报告failure cases）是对审稿人友好的写法

**但有两个坑需要注意：**

1. **LLAMP（Briefings 2025）已经做了MIC条件生成**——我们需要在Related Work里
   清楚说明差异：LLAMP是ESM-2微调，没有独立生成架构，没有QMAP评估，没有三范式对比。
   我们的novelty在于JEPA backbone + 三范式统一框架 + 同源感知评估。

2. **如果加入wet lab结果，定位可以升一级。** 20条肽的MIC数据哪怕是初步的，
   都能把paper从"纯计算方法"升级到"计算+实验pipeline"。

### 推荐Venue

| Venue | IF/级别 | 适合场景 | 截止时间 |
|---|---|---|---|
| **Bioinformatics (OUP)** | IF ~6 | 干lab paper完整即可投 | 滚动 |
| **PLOS Comp Bio** | IF ~4 | 方法+评估，开放获取 | 滚动 |
| **ISMB 2027** | 顶会 | 等三范式训完+消融，无wet lab也可 | 2027年初 |
| **NeurIPS 2026** | 顶会ML | 如果强调representation learning | ~5月截止 |
| **Cell Systems** | IF ~15 | 需要wet lab数据 | 滚动 |
| **Nature Comp Sci** | IF ~15 | 需要wet lab+更强故事 | 滚动 |

**当前最快路径**：
- 等ablation + NAR + Diffusion训完（~10小时）
- 补charge interpolation图
- 写完Methods和Results更新
- → 投 **Bioinformatics** 或 **PLOS Comp Bio** 作为保底
- → 同时等wet lab结果，若有数据再改投 **Cell Systems**

---

## 下一步清单

- [ ] Ablation training 完成后跑 generation_control_ablation evaluation
- [ ] NAR训练完成后跑generation质量评估
- [ ] Diffusion训练完成后跑generation质量评估  
- [ ] 跑charge interpolation（等GPU空闲）
- [ ] 更新paper methods section（加NAR/Diffusion架构描述）
- [ ] 更新paper results section（加embedding quality表，更新ablation表）
- [ ] 更新Related Work（明确与LLAMP/HydrAMP的差异）
- [ ] UMAP图放入paper（选最清晰的2张）
- [ ] 压缩到目标页数（ISMB 8页 / Bioinformatics 无硬限制）
