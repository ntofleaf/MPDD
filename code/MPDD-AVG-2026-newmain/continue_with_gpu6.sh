#!/usr/bin/env bash
# Wait for the in-flight Track2/binary run_pipeline.py to exit, then run the
# remaining ternary stages on 5 GPUs × 3 workers = 15 workers (adds GPU 6).
# Writes the final 'done grid Track2/ternary' marker that auto_pack waits for.
set -u
cd "$(dirname "$0")"

PIPELINE_PID="${1:-1524480}"   # Track2/binary pipeline PID
GRID_LOG="parallel_logs/grid/run.log"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
GPUS_EXTENDED="1,1,1,4,4,4,5,5,5,6,6,6,7,7,7"   # 5 GPUs × 3 workers = 15 parallel

echo "[$(date +%H:%M:%S)] continue_with_gpu6 waiting on PID=$PIPELINE_PID (Track2/binary)" | tee -a "$GRID_LOG"
while kill -0 "$PIPELINE_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] Track2/binary pipeline exited -- continuing ternary stages on GPUs $GPUS_EXTENDED" | tee -a "$GRID_LOG"

for TRACK in Track1 Track2; do
  LOG="parallel_logs/grid/${TRACK}_ternary.log"
  echo "===== $(date '+%H:%M:%S') starting grid $TRACK/ternary (gpu6 included) =====" | tee -a "$GRID_LOG"
  python run_pipeline.py \
    --track "$TRACK" --task ternary --mode grid \
    --grid_subtracks "G+P,A-V+P,A-V-G+P" \
    --grid_audios "mfcc,opensmile,wav2vec" \
    --grid_videos "resnet,densenet,openface" \
    --seeds 42,3407,2026 \
    --eval frozen_val \
    --gpus "$GPUS_EXTENDED" \
    --output_dir "experiments/grid_baseline" \
    --quiet > "$LOG" 2>&1
  echo "===== $(date '+%H:%M:%S') done grid $TRACK/ternary exit=$? =====" | tee -a "$GRID_LOG"
done
echo "[$(date +%H:%M:%S)] all stages complete" | tee -a "$GRID_LOG"