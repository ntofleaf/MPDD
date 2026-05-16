#!/usr/bin/env bash
# Elder A-V-G+P reproducible training pipeline.
#
# Phases:
#   1. Sanity training (single seed, single task) — quickly verify code works.
#   2. Full 5-seed × 2-task training in parallel across GPUs.
#   3. Per-ckpt sanity gates (drop collapsed/overfit ckpts).
#   4. Build ensemble ckpt lists + run pack_ensemble.py + zip.
#
# Usage:
#   bash avgp/run_avgp_pipeline.sh sanity        # just phase 1
#   bash avgp/run_avgp_pipeline.sh full          # phases 2-4
#   bash avgp/run_avgp_pipeline.sh all           # phases 1-4
#
# Env overrides:
#   SEEDS="42,3407,2026,7,123"       — seed pool for ensemble
#   GPUS="1,4,5,7"                    — GPUs available for parallel training
#   OUT_TAG="v1"                      — tag for output dirs
#   SKIP_TRAIN=1                      — skip retraining, just re-pack ensemble
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------- config ----------------
PHASE="${1:-all}"
SEEDS="${SEEDS:-42,3407,2026,7,123}"
GPUS="${GPUS:-1,4,5,7}"
OUT_TAG="${OUT_TAG:-v1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

BIN_RECIPE="avgp/recipe_binary.json"
TER_RECIPE="avgp/recipe_ternary.json"
EXP_PREFIX_BIN="avgp_repro_${OUT_TAG}_binary"
EXP_PREFIX_TER="avgp_repro_${OUT_TAG}_ternary"
LOG_DIR="avgp/logs_${OUT_TAG}"
mkdir -p "$LOG_DIR"

# Limit threads so parallel workers don't oversubscribe.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

# Parse comma-lists into bash arrays.
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra GPU_ARR  <<< "$GPUS"
N_SEEDS=${#SEED_ARR[@]}
N_GPUS=${#GPU_ARR[@]}

# ---------------- helpers ----------------
log()  { echo "[$(date +%H:%M:%S)] $*"; }

# Build CLI args from a recipe JSON.
# Reads numeric/string keys and emits `--key value` (skips keys starting with `_`).
build_args() {
  local recipe="$1"
  python3 -c "
import json, sys
d = json.load(open('$recipe'))
for k, v in d.items():
    if k.startswith('_'): continue
    if isinstance(v, bool):
        if v: print(f'--{k}')
    else:
        print(f'--{k}')
        print(str(v))
"
}

# Train one (task, seed) on a chosen GPU. Records ckpt dir.
train_one() {
  local task="$1" seed="$2" gpu="$3" recipe="$4" exp_prefix="$5"
  local exp_name="${exp_prefix}_seed${seed}"
  local logf="$LOG_DIR/${task}_seed${seed}_gpu${gpu}.log"
  log "training $task seed=$seed on GPU $gpu  -> $logf"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    $(build_args "$recipe") \
    --seed "$seed" \
    --experiment_name "$exp_name" \
    > "$logf" 2>&1
}

# ---------------- phase 1: sanity ----------------
phase_sanity() {
  log "=== Phase 1: sanity training (binary, seed=42, 30 epoch) ==="
  local logf="$LOG_DIR/sanity_binary.log"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python train.py \
    $(build_args "$BIN_RECIPE") \
    --seed 42 \
    --epochs 30 \
    --experiment_name "avgp_sanity_binary" \
    > "$logf" 2>&1
  log "sanity training done. Tail of log:"
  tail -20 "$logf"

  # Find the ckpt
  local ckpt
  ckpt=$(find checkpoints/Track1/A-V-G+P/binary/avgp_sanity_binary -name "best_model_*.pth" | head -1)
  if [[ -z "$ckpt" ]]; then
    log "ERROR: no sanity ckpt produced"
    return 1
  fi
  log "running sanity gates on $ckpt"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python avgp/sanity_check.py --ckpt "$ckpt"
}

# ---------------- phase 2: full 5-seed training ----------------
phase_train() {
  log "=== Phase 2: full training, ${N_SEEDS} seeds × 2 tasks = $((N_SEEDS*2)) runs on $N_GPUS GPUs ==="
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    log "SKIP_TRAIN=1 — skipping"
    return
  fi
  # Queue up all (task, seed) jobs, assign GPUs round-robin, run in parallel
  # waves of N_GPUS.
  local jobs=()
  for s in "${SEED_ARR[@]}"; do
    jobs+=("binary $s $BIN_RECIPE $EXP_PREFIX_BIN")
    jobs+=("ternary $s $TER_RECIPE $EXP_PREFIX_TER")
  done
  local i=0
  while (( i < ${#jobs[@]} )); do
    # launch N_GPUS jobs in parallel
    local pids=()
    for ((g=0; g<N_GPUS && i<${#jobs[@]}; g++,i++)); do
      read -r task seed recipe prefix <<< "${jobs[i]}"
      gpu="${GPU_ARR[g]}"
      train_one "$task" "$seed" "$gpu" "$recipe" "$prefix" &
      pids+=($!)
    done
    log "wave launched: ${#pids[@]} jobs (PIDs ${pids[*]}); waiting..."
    local fail=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then fail=$((fail+1)); fi
    done
    log "wave done. failures: $fail"
  done
  log "all training jobs finished"
}

# ---------------- phase 3: sanity check all ckpts ----------------
phase_sanity_all() {
  log "=== Phase 3: sanity gating ==="
  local bin_list="$LOG_DIR/ckpts_bin_passing.txt"
  local ter_list="$LOG_DIR/ckpts_ter_passing.txt"
  : > "$bin_list"; : > "$ter_list"
  local n_bin_pass=0 n_ter_pass=0 n_bin_total=0 n_ter_total=0

  for s in "${SEED_ARR[@]}"; do
    for task in binary ternary; do
      local prefix
      if [[ "$task" == "binary" ]]; then prefix="$EXP_PREFIX_BIN"; else prefix="$EXP_PREFIX_TER"; fi
      local exp_dir="checkpoints/Track1/A-V-G+P/$task/${prefix}_seed${s}"
      local ckpt
      ckpt=$(ls "$exp_dir"/best_model_*.pth 2>/dev/null | head -1)
      if [[ -z "$ckpt" ]]; then
        log "  skip $task seed=$s: no ckpt"
        continue
      fi
      if [[ "$task" == "binary" ]]; then n_bin_total=$((n_bin_total+1)); else n_ter_total=$((n_ter_total+1)); fi
      if CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python avgp/sanity_check.py --ckpt "$ckpt" \
            > "$LOG_DIR/sanity_${task}_seed${s}.log" 2>&1; then
        log "  ✓ $task seed=$s passed"
        if [[ "$task" == "binary" ]]; then
          echo "$ckpt" >> "$bin_list"; n_bin_pass=$((n_bin_pass+1))
        else
          echo "$ckpt" >> "$ter_list"; n_ter_pass=$((n_ter_pass+1))
        fi
      else
        log "  ✗ $task seed=$s FAILED sanity"
        tail -5 "$LOG_DIR/sanity_${task}_seed${s}.log" | sed 's/^/      /'
      fi
    done
  done
  log "sanity: binary $n_bin_pass/$n_bin_total, ternary $n_ter_pass/$n_ter_total"
  if (( n_bin_pass == 0 || n_ter_pass == 0 )); then
    log "ERROR: zero ckpts passed sanity for at least one task. Aborting ensemble."
    return 1
  fi
}

# ---------------- phase 4: ensemble + pack ----------------
phase_pack() {
  log "=== Phase 4: ensemble + pack zip ==="
  local bin_list="$LOG_DIR/ckpts_bin_passing.txt"
  local ter_list="$LOG_DIR/ckpts_ter_passing.txt"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python avgp/pack_ensemble.py \
    --binary_ckpts "$bin_list" \
    --ternary_ckpts "$ter_list" \
    --out_subdir "avgp_repro_${OUT_TAG}"
  log "submission zip:"
  ls -la "sub/avgp_repro_${OUT_TAG}/"*.zip
  log "summary:"
  cat "sub/avgp_repro_${OUT_TAG}/summary.json" | python3 -m json.tool | head -40
}

# ---------------- dispatch ----------------
case "$PHASE" in
  sanity) phase_sanity ;;
  train)  phase_train ;;
  check)  phase_sanity_all ;;
  pack)   phase_pack ;;
  full)   phase_train && phase_sanity_all && phase_pack ;;
  all)    phase_sanity && phase_train && phase_sanity_all && phase_pack ;;
  *) echo "Usage: $0 {sanity|train|check|pack|full|all}"; exit 2 ;;
esac

log "DONE."
