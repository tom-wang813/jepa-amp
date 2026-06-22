# TODO — JEPA-AMP
_Updated: 2026-06-06_

---

## 🔄 正在进行（等GPU通知）

### Job b71d3k5g2：NAR → Diffusion → v6 串行训练
```bash
# 检查进度
tail -5 logs/finetune_nar.log
tail -5 logs/finetune_diffusion.log
tail -5 logs/finetune_v6.log
```
- [ ] NAR 训练完成（epoch ~17/80，约剩1.5小时）
- [ ] Diffusion 训练完成（接续，约3小时）
- [ ] v6 (7维条件) 训练完成（接续，约2小时）

---

## ⏳ 训练完成后立刻做（按顺序）

### Step 1: 评估三种生成范式的 charge control

需要先创建评估config，再运行：

```bash
# NAR evaluation
uv run python scripts/evaluate_generation_control.py \
    --config configs/generation_control_nar.yaml

# Diffusion evaluation
uv run python scripts/evaluate_generation_control.py \
    --config configs/generation_control_diffusion.yaml
```

- [ ] 创建 `configs/generation_control_nar.yaml`（参考 generation_control_ablation.yaml，只加 NAR variant）
- [ ] 创建 `configs/generation_control_diffusion.yaml`
- [ ] 跑 NAR evaluation → `eval_results/generation_control_nar/SUMMARY.md`
- [ ] 跑 Diffusion evaluation → `eval_results/generation_control_diffusion/SUMMARY.md`

### Step 2: 评估 v6 (7维) 的多属性控制

v6 多了4个条件维度，需要专门的 evaluation 脚本：

```bash
uv run python scripts/evaluate_generation_control_v6.py \
    --config configs/generation_control_v6.yaml
```

- [ ] 创建 `configs/generation_control_v6.yaml`（10个target，指定7维条件）
- [ ] 创建/修改 evaluation 脚本支持 v6 的7维条件向量
- [ ] 跑 v6 evaluation：记录 charge/helix/pI/HM/AMP_score 各自的 R²

### Step 3: Charge interpolation（被kill，需重跑）

```bash
uv run python scripts/charge_interpolation.py 2>&1 | tee logs/charge_interpolation.log
```

- [ ] 重跑 charge sweep（-9 到 +13，23个点，50个context序列）
- [ ] 产出 `eval_results/charge_interpolation/charge_sweep.png` 和 `metrics.json`

### Step 4: 填写 Paper 占位符

Paper 中所有 `\todo{---}` 需要替换成真实数字：

**`paper/sections/results.tex` Table 4（Generation Ablation）：**
- [ ] `v4-no-aux` 的 Charge R², MAE, GRAVY R², Length R², Unique → 已知，填入
  - Charge R² = **-16.763**, MAE = 5.571, Unique = 0.975
- [ ] `v4-no-dropout` 的各指标 → 已知，填入
  - Charge R² = **0.805**, MAE = 1.629, Unique = 0.836

**`paper/sections/results.tex` Table 5（Generation Paradigm Comparison）：**
- [ ] NAR: Charge R², Charge MAE, Novelty, Diversity → 等评估完成
- [ ] Diffusion: 同上 → 等评估完成
- [ ] v6 (AR, 7-dim): 同上 + helix R², pI R², AMP score R² → 等评估完成

---

## 📝 写作 TODO（可以现在做，不需要等GPU）

- [ ] `paper/sections/related_work.tex` — 明确与 LLAMP、HydrAMP 的差异
  - LLAMP：ESM2微调，无独立生成架构，无QMAP评估，无三范式对比
  - HydrAMP：binary活性条件，VAE架构，无MIC定量条件
- [ ] Figure 描述：在 methods/results 里引用的图要确认存在
  - `fig:arch` — 需要画架构图（JEPA encoder + 三种decoder）
  - `fig:umap` — UMAP图已存在 (`eval_results/umap/`)
  - `fig:charge_sweep` — 等charge_interpolation跑完
- [ ] Abstract 最后一句 `[repo]` 改成真实GitHub链接（如果有的话）
- [ ] 检查并统一 QMAP 表格：abstract 用 multi-seed mean，Table 3 也要统一

---

## 🎯 收到结果后预期能说什么

### NAR（预期）
- Charge R² 预期接近 v4（0.866），因为共享同样的条件设计
- 生成多样性（Diversity）预期高于 AR（双向注意力允许更全局的序列选择）
- 长度控制预期**好于 AR**（NAR有专用 length prediction head）
- 推理速度约 30× 快于 AR（不需要顺序生成 token）

### Masked Diffusion（预期）
- Charge R² 可能略低于 AR（迭代去噪 vs 直接条件注入）
- GRAVY/Length 控制可能**优于 AR**（全局去噪使全局属性更易约束）
- 多样性最高（随机扩散过程引入结构噪声）

### v6 7维条件（预期）
- Charge R² 维持在 0.85 左右（机制不变）
- Helix R² 期望 0.3-0.6（Chou-Fasman分数是纯计算，应该可学）
- pI R² 期望 0.4-0.7（pI与charge强相关，应该可学）
- Hydrophobic moment R² 期望 0.2-0.5（两亲性部分独立于charge）
- AMP score R² **最不确定**——可能 0.1-0.5，取决于模型能否把分类器信号转化为生成控制

---

## 🔬 可选后续实验（不急）

- [ ] Wet lab 结果整合（当20条候选肽有MIC数据时）
- [ ] v6 + MIC 条件结合（把7维physchem + 43维MIC合并成50维条件）
- [ ] HydrAMP 直接对比（同样训练数据跑HydrAMP的charge control）
- [ ] 生成序列多样性详细分析（sequence logo + 系统发育树）
- [ ] 投稿压缩：把 main.tex 压到 ISMB 8页或 Bioinformatics 限制

---

## 📌 不需要做的事

- ❌ 不要重跑 QMAP（3-seed pack已锁定）
- ❌ 不要重新生成数据split（已锁定 seed=42）
- ❌ 不要在paper里claim MC-Dropout有提升（negative result，已删除）
- ❌ 不要同时跑多个GPU训练任务（单卡3090）
