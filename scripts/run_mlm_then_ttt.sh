#!/bin/bash
# Full pipeline: wait for MLM pretraining (epoch 100) → finetune MIC head → all TTT benchmarks
# Usage: nohup bash scripts/run_mlm_then_ttt.sh [gpu_id] > logs/pipeline.log 2>&1 &

GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Wait for MLM pretraining to FULLY finish (epoch_100.pt = definitive done) ─
log "Waiting for MLM pretraining epoch 100..."
until [ -f checkpoints/mlm_pretrain_868k/epoch_100.pt ]; do
    LATEST=$(ls checkpoints/mlm_pretrain_868k/epoch_*.pt 2>/dev/null | tail -1 | xargs -I{} basename {} 2>/dev/null || echo "none")
    log "  still training... latest=$LATEST"
    sleep 120
done
log "MLM pretraining done: epoch_100.pt found"
sleep 30   # let disk flush

# ── 2. Finetune MIC head on MLM encoder ───────────────────────────────────────
log "Starting MLM MIC finetuning..."
uv run python -m src.train.finetune_supervised \
    --config configs/mic_mlm_868k_transformer.yaml \
    --gpu "$GPU" \
    > logs/mlm_finetune_mic.log 2>&1

if [ $? -ne 0 ]; then
    log "ERROR: MLM MIC finetuning failed. Check logs/mlm_finetune_mic.log"
    tail -20 logs/mlm_finetune_mic.log
    exit 1
fi
log "MLM MIC finetuning done."

# ── 3. Run comprehensive TTT benchmark suite (B1–B4) ──────────────────────────
# B1: cross-species MIC transfer (OOD species)
# B2: AMP classification OOD (AMPlify train → APD3 test)
# B3: MIC low-data (5% labels)
# B4: head-free perplexity scoring (zero-shot AMP vs non-AMP)
log "Starting TTT benchmark suite..."
uv run python scripts/eval_ttt_benchmarks.py \
    --gpu "$GPU" \
    --ttt_steps 10 \
    --ttt_lr 1e-3 \
    --n_masks 3 \
    --benchmarks B1 B2 B3 B4 \
    > logs/ttt_benchmarks.log 2>&1

if [ $? -ne 0 ]; then
    log "ERROR: TTT benchmarks failed. Check logs/ttt_benchmarks.log"
    tail -20 logs/ttt_benchmarks.log
    exit 1
fi
log "All benchmarks done. Results in eval_results/ttt_benchmarks/"

# ── 4. Print summary ──────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
cat eval_results/ttt_benchmarks/SUMMARY.md
