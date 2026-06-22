#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m src.train.finetune_supervised \
  --config configs/mic_868k_transformer_formal.yaml --gpu 0 \
  > logs/formal_mic_868k_transformer.log 2>&1 &

transformer_pid=$!

(
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 \
    .venv/bin/python -m src.train.finetune_supervised \
    --config configs/mic_868k_mlp_formal.yaml --gpu 0 \
    > logs/formal_mic_868k_mlp.log 2>&1

  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 \
    .venv/bin/python -m src.train.train_esm_supervised \
    --config configs/esm2_mic_formal.yaml --gpu 0 \
    > logs/formal_esm2_mic.log 2>&1
) &

baseline_lane_pid=$!

echo "formal_mic_transformer_pid=${transformer_pid}"
echo "formal_mlp_then_esm_pid=${baseline_lane_pid}"

wait "${transformer_pid}"
transformer_status=$?
wait "${baseline_lane_pid}"
baseline_lane_status=$?

echo "formal_mic_transformer_exit=${transformer_status}"
echo "formal_mlp_then_esm_exit=${baseline_lane_status}"

exit $(( transformer_status != 0 || baseline_lane_status != 0 ))
