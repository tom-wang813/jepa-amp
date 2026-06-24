#!/bin/bash
# Serial queue: wait for current warmstart to finish, then run v2 for all models.
# Usage: nohup bash run_v2_queue.sh > logs/v2_queue.log 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

echo "[$(date)] Waiting for warmstart run to finish..."
while pgrep -f "fewshot_cross_species.py.*warmstart" > /dev/null; do
    sleep 60
done
echo "[$(date)] Warmstart done."

for MODEL in jepa esm2 mlm esm2_650m; do
    echo ""
    echo "[$(date)] === Starting v2: $MODEL ==="
    uv run python scripts/fewshot_cross_species_v2.py --gpu 0 --model "$MODEL" \
        >> "logs/v2_${MODEL}.log" 2>&1
    echo "[$(date)] === Done: $MODEL ==="
done

echo ""
echo "[$(date)] All models done. Generating scatter plots..."
uv run python scripts/plot_scatter_pred_actual.py --model jepa esm2 mlm esm2_650m \
    >> logs/v2_plots.log 2>&1

echo "[$(date)] Queue complete."
