#!/usr/bin/env bash
# Wait for the binary+ternary grid to finish, then pack 6 submission zips.
# phq9_pred is sourced from the highest val-CCC ckpt among binary+ternary runs
# for each (track, subtrack); no separate regression task is trained.
set -u
cd "$(dirname "$0")"

GRID_LOG="parallel_logs/grid/run.log"
PACK_LOG="parallel_logs/grid/pack.log"
TARGET_OUT="experiments/grid_baseline"

echo "[$(date +%H:%M:%S)] wait_and_pack waiting for 'done grid Track2/ternary' + no train.py workers" | tee -a "$PACK_LOG"
while true; do
  if grep -q "done grid Track2/ternary" "$GRID_LOG" 2>/dev/null; then
    if ! ps -eo cmd | grep -E "train\.py --track" | grep -qv grep; then
      break
    fi
  fi
  sleep 20
done

echo "[$(date +%H:%M:%S)] grid finished -- running pack_submissions.py" | tee -a "$PACK_LOG"
CUDA_VISIBLE_DEVICES=6 python pack_submissions.py \
  --out_dir sub/submissions \
  --newer_than "$TARGET_OUT" 2>&1 | tee -a "$PACK_LOG"

EC=${PIPESTATUS[0]}
echo "[$(date +%H:%M:%S)] pack_submissions.py exit=$EC" | tee -a "$PACK_LOG"
echo "[$(date +%H:%M:%S)] zips:" | tee -a "$PACK_LOG"
ls -la sub/submissions/*.zip 2>/dev/null | tee -a "$PACK_LOG"