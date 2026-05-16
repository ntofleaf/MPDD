#!/usr/bin/env bash
# Process top combos × 5 folds with 4 parallel run_pipeline.py instances,
# each instance owning its own 3-slot GPU subset to avoid CUDA contention.
set -u
cd "$(dirname "$0")"

SEL_JSON="${SEL_JSON:-experiments/grid_baseline/top_combos_for_kfold.json}"
SEED="${SEED:-42}"
N_SPLITS="${N_SPLITS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/grid_baseline}"
RUN_LOG="parallel_logs/grid/kfold_validation.log"
: > "$RUN_LOG"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# 4 parallel workers; each owns 3 GPU slots (3 folds in flight per combo).
GPU_GROUPS=("1,1,1" "4,4,4" "5,5,5" "7,7,7")

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN_LOG"; }

log "kfold validation starting -- selections from $SEL_JSON"
COMBOS=$(python3 -c "
import json
sel = json.load(open('$SEL_JSON'))
for track, by_task in sel.items():
    for task, by_sub in by_task.items():
        for sub, combos in by_sub.items():
            for c in combos:
                print(f\"{track}|{task}|{sub}|{c['audio']}|{c['video']}|{c['encoder']}\")
")

TOTAL=$(echo "$COMBOS" | wc -l)
log "running $TOTAL combos × $N_SPLITS folds = $((TOTAL*N_SPLITS)) train.py runs across 4 parallel groups"

# Round-robin combos onto the 4 GPU groups via batched parallel execution.
# Each batch: launch up to 4 combos simultaneously and wait for all to finish.
IDX=0
BATCH=0
TMP_BATCH=$(mktemp)
echo "$COMBOS" > "$TMP_BATCH"

run_one_combo() {
  local TRACK="$1" TASK="$2" SUB="$3" AUDIO="$4" VIDEO="$5" ENC="$6" GPUS_LOCAL="$7" IDX_LOCAL="$8"
  local TAG="${TRACK}/${TASK}/${SUB}/${AUDIO}+${VIDEO}"
  local STAGE_LOG="parallel_logs/grid/kfold_${IDX_LOCAL}_${TRACK}_${TASK}_${SUB//+/p}_${AUDIO}_${VIDEO}_${ENC}.log"
  log "[$IDX_LOCAL/$TOTAL] gpus=$GPUS_LOCAL  $TAG"
  python run_pipeline.py \
    --track "$TRACK" --task "$TASK" --mode kfold \
    --subtrack "$SUB" \
    --audio_feature "$AUDIO" \
    --video_feature "$VIDEO" \
    --encoder "$ENC" \
    --seed "$SEED" \
    --n_splits "$N_SPLITS" \
    --gpus "$GPUS_LOCAL" \
    --output_dir "$OUTPUT_DIR" \
    --quiet > "$STAGE_LOG" 2>&1
  log "[$IDX_LOCAL/$TOTAL] done   $TAG  exit=$?"
}

declare -a LINES
mapfile -t LINES < "$TMP_BATCH"
rm -f "$TMP_BATCH"

while [ ${#LINES[@]} -gt 0 ]; do
  BATCH=$((BATCH+1))
  log "--- batch $BATCH (remaining=${#LINES[@]}) ---"
  PIDS=()
  for slot in 0 1 2 3; do
    [ ${#LINES[@]} -eq 0 ] && break
    LINE="${LINES[0]}"
    LINES=("${LINES[@]:1}")
    IDX=$((IDX+1))
    IFS='|' read -r TRACK TASK SUB AUDIO VIDEO ENC <<< "$LINE"
    run_one_combo "$TRACK" "$TASK" "$SUB" "$AUDIO" "$VIDEO" "$ENC" "${GPU_GROUPS[$slot]}" "$IDX" &
    PIDS+=($!)
  done
  # Wait for batch
  for pid in "${PIDS[@]}"; do wait "$pid"; done
done

log "all kfold validations complete"