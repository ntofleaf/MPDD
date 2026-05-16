#!/usr/bin/env bash
# Waits for the grid orchestrator to finish, then runs pack_submissions.py.
# Designed to be launched in the background alongside the grid.
set -u
cd "$(dirname "$0")"

GRID_LOG="parallel_logs/grid/run.log"
PACK_LOG="parallel_logs/grid/pack.log"
EXPECTED_DONE_MARK="done grid Track2/ternary"
REF_FOR_FILTER="experiments/grid_baseline"

echo "[$(date +%H:%M:%S)] auto-pack waiting for '$EXPECTED_DONE_MARK' in $GRID_LOG" | tee "$PACK_LOG"

# Wait until orchestrator writes the final marker AND no train.py workers remain.
while true; do
  if grep -q "$EXPECTED_DONE_MARK" "$GRID_LOG" 2>/dev/null; then
    if ! ps -eo cmd | grep -E "train\.py --track" | grep -qv grep; then
      break
    fi
  fi
  sleep 15
done

echo "[$(date +%H:%M:%S)] grid finished -- launching pack_submissions.py" | tee -a "$PACK_LOG"

# Pin pack inference to a single idle GPU so it does not fight grid workers.
CUDA_VISIBLE_DEVICES=6 python pack_submissions.py \
  --out_dir submissions \
  --newer_than "$REF_FOR_FILTER" 2>&1 | tee -a "$PACK_LOG"

EC=${PIPESTATUS[0]}
echo "[$(date +%H:%M:%S)] pack_submissions.py exit=$EC" | tee -a "$PACK_LOG"
echo "[$(date +%H:%M:%S)] zips written under: $(pwd)/submissions" | tee -a "$PACK_LOG"
ls -la submissions/*.zip 2>/dev/null | tee -a "$PACK_LOG"