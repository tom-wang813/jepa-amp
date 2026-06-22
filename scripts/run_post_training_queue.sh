#!/usr/bin/env bash
# Post-training analysis queue.
# Run this AFTER the NAR→Diffusion→v6 training queue finishes.
# Usage: bash scripts/run_post_training_queue.sh 2>&1 | tee logs/post_training_queue.log

set -e
GPU=0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "============================================"
echo "Post-training queue started: $(date)"
echo "============================================"

# ── Step 1: Evaluate NAR generation control ───────────────────────────────────
echo ""
echo "[1/6] NAR generation control evaluation"
uv run python scripts/evaluate_generation_control.py \
    --config configs/generation_control_nar.yaml \
    2>&1 | tee logs/eval_gen_control_nar.log

# ── Step 2: Evaluate Diffusion generation control ────────────────────────────
echo ""
echo "[2/6] Diffusion generation control evaluation"
uv run python scripts/evaluate_generation_control.py \
    --config configs/generation_control_diffusion.yaml \
    2>&1 | tee logs/eval_gen_control_diffusion.log

# ── Step 3: Charge interpolation (was killed earlier) ────────────────────────
echo ""
echo "[3/6] Charge interpolation"
uv run python scripts/charge_interpolation.py \
    2>&1 | tee logs/charge_interpolation.log

# ── Step 4: Learning curve (JEPA + ESM2, 6 fractions × 3 seeds) ──────────────
echo ""
echo "[4/6] Data efficiency learning curve"
uv run python scripts/run_learning_curve.py \
    --gpu $GPU --model both \
    2>&1 | tee logs/learning_curve.log

# ── Step 5: Cross-species zero-shot transfer ─────────────────────────────────
echo ""
echo "[5/6] Cross-species MIC transfer"
uv run python scripts/cross_species_transfer.py \
    --gpu $GPU \
    2>&1 | tee logs/cross_species_transfer.log

# ── Step 6: Attention saliency visualization ─────────────────────────────────
echo ""
echo "[6/6] Attention saliency"
uv run python scripts/attention_saliency.py \
    2>&1 | tee logs/attention_saliency.log || echo "[WARN] attention_saliency.py not yet written, skipping"

echo ""
echo "============================================"
echo "All done: $(date)"
echo "============================================"
