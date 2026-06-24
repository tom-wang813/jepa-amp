#!/bin/bash
# Sequential training queue: NAR → Diffusion → v6 (7-dim conditions)
# NAR is already running; this script handles Diffusion + v6.
set -e
cd /home/ziwei/jepa-amp

echo "=== [$(date)] Starting Diffusion training ==="
uv run python -m src.train.finetune_nar_diffusion \
    --config configs/finetune_868k_diffusion.yaml --gpu 0 \
    2>&1 | tee logs/finetune_diffusion.log

echo "=== [$(date)] Diffusion done. Starting v6 (7-dim conditions) training ==="
uv run python -m src.train.finetune \
    --config configs/finetune_868k_v6.yaml --gpu 0 \
    2>&1 | tee logs/finetune_v6.log

echo "=== [$(date)] All generation models trained ==="
