#!/usr/bin/env bash
# Pure baseline pipeline — no modifications, no CCC loss, no determinism,
# no fixed split_seed, no top-K. Just like the original baseline winner.
#
# Trains 5 seeds × 2 tasks, ensembles softmax probs across all surviving
# (non-collapsed) ckpts, and packs a submission zip.
#
# Phases:
#   train  — 5 seeds × 2 tasks on parallel GPUs (no special flags)
#   check  — drop ckpts with class-collapsed predictions on the local val
#   pack   — softmax ensemble + zip
#   all    — full pipeline
#
# Usage:
#   bash avgp/run_pure_baseline.sh all
#
# Env:
#   SEEDS="42,3407,2026,7,123"     — seed pool
#   GPUS="1,4,5,7"                  — parallel GPUs
#   OUT_TAG="pure_v1"               — output dir tag
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:-all}"
SEEDS="${SEEDS:-42,3407,2026,7,123}"
GPUS="${GPUS:-1,4,5,7}"
OUT_TAG="${OUT_TAG:-pure_v1}"

BIN_RECIPE="avgp/recipe_baseline_binary.json"
TER_RECIPE="avgp/recipe_baseline_ternary.json"
EXP_PREFIX_BIN="avgp_pure_${OUT_TAG}_binary"
EXP_PREFIX_TER="avgp_pure_${OUT_TAG}_ternary"
LOG_DIR="avgp/logs_${OUT_TAG}"
mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra GPU_ARR  <<< "$GPUS"
N_SEEDS=${#SEED_ARR[@]}
N_GPUS=${#GPU_ARR[@]}

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Convert recipe JSON to CLI args. Skip "_*" keys.
build_args() {
  python3 -c "
import json
d = json.load(open('$1'))
for k, v in d.items():
    if k.startswith('_'): continue
    if isinstance(v, bool):
        if v: print(f'--{k}')
    else:
        print(f'--{k}'); print(str(v))
"
}

train_one() {
  local task="$1" seed="$2" gpu="$3" recipe="$4" exp_prefix="$5"
  local exp_name="${exp_prefix}_seed${seed}"
  local logf="$LOG_DIR/${task}_seed${seed}_gpu${gpu}.log"
  log "training $task seed=$seed on GPU $gpu  -> $logf"
  # NOTE: pass --seed only (no --split_seed → random val each run, like baseline winner)
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    $(build_args "$recipe") \
    --seed "$seed" \
    --experiment_name "$exp_name" \
    > "$logf" 2>&1
}

phase_train() {
  log "=== Pure baseline training: ${N_SEEDS} seeds × 2 tasks on $N_GPUS GPUs ==="
  local jobs=()
  for s in "${SEED_ARR[@]}"; do
    jobs+=("binary $s $BIN_RECIPE $EXP_PREFIX_BIN")
    jobs+=("ternary $s $TER_RECIPE $EXP_PREFIX_TER")
  done
  local i=0
  while (( i < ${#jobs[@]} )); do
    local pids=()
    for ((g=0; g<N_GPUS && i<${#jobs[@]}; g++,i++)); do
      read -r task seed recipe prefix <<< "${jobs[i]}"
      gpu="${GPU_ARR[g]}"
      train_one "$task" "$seed" "$gpu" "$recipe" "$prefix" &
      pids+=($!)
    done
    log "wave: ${#pids[@]} jobs (PIDs ${pids[*]}); waiting..."
    local fail=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then fail=$((fail+1)); fi
    done
    log "wave done. failures: $fail"
  done
  log "all training jobs finished"
}

# Light check: keep ckpts whose val predictions span >= 2 classes
# (the only non-negotiable failure mode — predicting 1 class is unfixable).
phase_check() {
  log "=== Light sanity check: filter only fully-collapsed ckpts ==="
  local bin_list="$LOG_DIR/ckpts_bin_passing.txt"
  local ter_list="$LOG_DIR/ckpts_ter_passing.txt"
  : > "$bin_list"; : > "$ter_list"
  local n_bin_pass=0 n_ter_pass=0

  for s in "${SEED_ARR[@]}"; do
    for task in binary ternary; do
      local prefix
      if [[ "$task" == "binary" ]]; then prefix="$EXP_PREFIX_BIN"; else prefix="$EXP_PREFIX_TER"; fi
      local exp_dir="checkpoints/Track1/A-V-G+P/$task/${prefix}_seed${s}"
      local ckpt
      ckpt=$(ls "$exp_dir"/best_model_*.pth 2>/dev/null | head -1)
      if [[ -z "$ckpt" ]]; then
        log "  $task seed=$s: no ckpt, skip"
        continue
      fi
      # Use light collapse-only check: just verify pred_dist has >= 2 classes
      local pred_classes
      pred_classes=$(CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -c "
import sys
sys.path.insert(0, '.')
from avgp.sanity_check import evaluate_on, check_ckpt
from pathlib import Path
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
v = evaluate_on(Path('$ckpt'), 'val', device)
n_pred = len([c for c, n in v['pred_dist'].items() if n > 0])
print(n_pred, v.get('f1', 0), v.get('kappa', 0), v.get('ccc', 0), v.get('phq_pred_std', 0))
" 2>/dev/null)
      local n_pred f1 kappa ccc phq_std
      read -r n_pred f1 kappa ccc phq_std <<< "$pred_classes"
      local n_pred_int=${n_pred%.*}
      local expected_classes=2
      if [[ "$task" == "ternary" ]]; then expected_classes=2; fi  # accept 2/3, reject 1
      if [[ -z "$n_pred_int" ]] || (( n_pred_int < expected_classes )); then
        log "  ✗ $task seed=$s COLLAPSED (predicted $n_pred_int classes; f1=$f1)"
        continue
      fi
      log "  ✓ $task seed=$s  f1=$f1 kappa=$kappa ccc=$ccc phq_std=$phq_std"
      if [[ "$task" == "binary" ]]; then
        echo "$ckpt" >> "$bin_list"; n_bin_pass=$((n_bin_pass+1))
      else
        echo "$ckpt" >> "$ter_list"; n_ter_pass=$((n_ter_pass+1))
      fi
    done
  done
  log "non-collapsed: binary $n_bin_pass/$N_SEEDS, ternary $n_ter_pass/$N_SEEDS"
  if (( n_bin_pass == 0 || n_ter_pass == 0 )); then
    log "ERROR: no surviving ckpts for at least one task. Cannot ensemble."
    return 1
  fi
}

phase_pack() {
  log "=== Ensemble + pack ==="
  local bin_list="$LOG_DIR/ckpts_bin_passing.txt"
  local ter_list="$LOG_DIR/ckpts_ter_passing.txt"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python avgp/pack_ensemble.py \
    --binary_ckpts "$bin_list" \
    --ternary_ckpts "$ter_list" \
    --out_subdir "avgp_pure_${OUT_TAG}"
  log "zip:"
  ls -la "sub/avgp_pure_${OUT_TAG}/"*.zip
}

case "$PHASE" in
  train) phase_train ;;
  check) phase_check ;;
  pack)  phase_pack ;;
  all)   phase_train && phase_check && phase_pack ;;
  *) echo "Usage: $0 {train|check|pack|all}"; exit 2 ;;
esac

log "DONE."
