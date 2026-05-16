#!/usr/bin/env bash
# End-to-end MPDD-AVG pipeline. From scratch to 6 submission zips, one command.
#
#   1. (optional) Regenerate frozen single + 5-fold stratified splits.
#   2. Run grid over feature combinations for binary, ternary, regression.
#   3. Auto-pack 6 submission zips:
#         binary_pred  from best val Macro-F1 binary checkpoint
#         ternary_pred from best val Macro-F1 ternary checkpoint
#         phq9_pred    from best val CCC      regression checkpoint
#
# Usage:
#   bash run_full_pipeline.sh                # standard run
#   REGEN_SPLITS=1 bash run_full_pipeline.sh # also regenerate frozen splits
#   GPUS="1,1,4,4,5,5,6,6,7,7" bash run_full_pipeline.sh
#   SEEDS="42,3407,2026,7,123"  bash run_full_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"

# ----- Knobs ---------------------------------------------------------------
GPUS="${GPUS:-1,1,1,4,4,4,5,5,5,7,7,7}"   # repeats = workers per GPU
SEEDS="${SEEDS:-42,3407,2026}"
SUBTRACKS="${SUBTRACKS:-G+P,A-V+P,A-V-G+P}"
AUDIO_FEATURES="${AUDIO_FEATURES:-mfcc,opensmile,wav2vec}"
VIDEO_FEATURES="${VIDEO_FEATURES:-resnet,densenet,openface}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/grid_baseline}"
SUBMISSIONS_DIR="${SUBMISSIONS_DIR:-sub/submissions}"
REGEN_SPLITS="${REGEN_SPLITS:-0}"

# Limit each train.py to a small thread count so the box doesn't choke on 128
# core oversubscription (load > #cores -> data loading becomes the bottleneck).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

mkdir -p parallel_logs/grid "$OUTPUT_DIR"
RUN_LOG="parallel_logs/grid/run.log"
: > "$RUN_LOG"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN_LOG"; }

# ----- Step 1: optional split regeneration ---------------------------------
if [[ "$REGEN_SPLITS" == "1" ]]; then
  log "regenerating stratified single-split CSVs (Track1, Track2)"
  python generate_stratified_splits.py --track Track1 --val_ratio 0.1 --n_candidates 400 | tee -a "$RUN_LOG"
  python generate_stratified_splits.py --track Track2 --val_ratio 0.1 --n_candidates 400 | tee -a "$RUN_LOG"
  log "regenerating stratified 5-fold CSVs (Track1, Track2)"
  python generate_kfold_splits.py --track Track1 --n_splits 5 --n_candidates 400 | tee -a "$RUN_LOG"
  python generate_kfold_splits.py --track Track2 --n_splits 5 --n_candidates 400 | tee -a "$RUN_LOG"
fi

# ----- Step 2: grid over feature combos for each (track, task) -------------
# Only binary + ternary. The regression head trained alongside each
# classification model is sufficient to source phq9_pred (pack_submissions.py
# picks the highest val-CCC ckpt across both tasks).
for TASK in binary ternary; do
  for TRACK in Track1 Track2; do
    STAGE_LOG="parallel_logs/grid/${TRACK}_${TASK}.log"
    log "starting grid $TRACK/$TASK -> $STAGE_LOG"
    python run_pipeline.py \
      --track "$TRACK" --task "$TASK" --mode grid \
      --grid_subtracks "$SUBTRACKS" \
      --grid_audios "$AUDIO_FEATURES" \
      --grid_videos "$VIDEO_FEATURES" \
      --seeds "$SEEDS" \
      --eval frozen_val \
      --gpus "$GPUS" \
      --output_dir "$OUTPUT_DIR" \
      --quiet > "$STAGE_LOG" 2>&1
    log "done grid $TRACK/$TASK exit=$?"
  done
done

# ----- Step 3: pack 6 submission zips --------------------------------------
log "all training stages complete -- packing submissions"

# Pin packing inference to a single GPU.
PACK_GPU="${PACK_GPU:-6}"
CUDA_VISIBLE_DEVICES="$PACK_GPU" python pack_submissions.py \
  --out_dir "$SUBMISSIONS_DIR" \
  --newer_than "$OUTPUT_DIR" 2>&1 | tee -a "$RUN_LOG"

log "submission zips:"
ls -la "$SUBMISSIONS_DIR"/*.zip 2>&1 | tee -a "$RUN_LOG"
log "summary JSON: $SUBMISSIONS_DIR/submissions_summary.json"