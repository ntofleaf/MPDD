#!/usr/bin/env bash
# Wait for the in-flight binary+ternary grid to finish, then run the regression
# grid for Track1 + Track2, then pack 6 submission zips.
#
# Why a separate stage for regression:
#   The competition wants three predictions per test sample (binary, ternary,
#   phq9). The classification training already produces a PHQ head as a side
#   effect, but its CCC tends to be near zero. By training a dedicated
#   regression model (task=regression, selected by val CCC) and using *its*
#   phq9 prediction for both binary.csv and ternary.csv, we get a much better
#   phq9_pred column at the cost of one extra training stage.
set -u
cd "$(dirname "$0")"

GRID_LOG="parallel_logs/grid/run.log"
PACK_LOG="parallel_logs/grid/pack.log"
TARGET_OUT="experiments/grid_baseline"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
GPUS="1,1,1,4,4,4,5,5,5,7,7,7"   # 4 GPUs × 3 workers = 12 (matches the binary/ternary stage)

echo "[$(date +%H:%M:%S)] regression_then_pack waiting for binary+ternary grid to finish" | tee -a "$PACK_LOG"
# Wait until BOTH markers are present (Track1/ternary + Track2/ternary done)
# AND no train.py workers are alive.
while true; do
  if grep -q "done grid Track2/ternary" "$GRID_LOG" 2>/dev/null; then
    if ! ps -eo cmd | grep -E "train\.py --track" | grep -qv grep; then
      break
    fi
  fi
  sleep 20
done

echo "[$(date +%H:%M:%S)] binary+ternary grid finished -- starting regression grid" | tee -a "$PACK_LOG"

for TRACK in Track1 Track2; do
  LOG="parallel_logs/grid/${TRACK}_regression.log"
  echo "===== $(date '+%H:%M:%S') starting grid $TRACK/regression =====" | tee -a "$GRID_LOG"
  python run_pipeline.py \
    --track "$TRACK" --task regression --mode grid \
    --grid_subtracks "G+P,A-V+P,A-V-G+P" \
    --grid_audios "mfcc,opensmile,wav2vec" \
    --grid_videos "resnet,densenet,openface" \
    --seeds 42,3407,2026 \
    --eval frozen_val \
    --gpus "$GPUS" \
    --output_dir "$TARGET_OUT" \
    --quiet > "$LOG" 2>&1
  echo "===== $(date '+%H:%M:%S') done grid $TRACK/regression exit=$? =====" | tee -a "$GRID_LOG"
done

echo "[$(date +%H:%M:%S)] regression grid finished -- running pack_submissions.py" | tee -a "$PACK_LOG"

# Use GPU 6 for inference so it does not fight any remaining workers.
CUDA_VISIBLE_DEVICES=6 python pack_submissions.py \
  --out_dir submissions \
  --newer_than "$TARGET_OUT" 2>&1 | tee -a "$PACK_LOG"

EC=${PIPESTATUS[0]}
echo "[$(date +%H:%M:%S)] pack_submissions.py exit=$EC" | tee -a "$PACK_LOG"
echo "[$(date +%H:%M:%S)] zips:" | tee -a "$PACK_LOG"
ls -la submissions/*.zip 2>/dev/null | tee -a "$PACK_LOG"