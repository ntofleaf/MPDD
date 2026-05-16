#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$PROJECT_ROOT/test_scripts/_common.sh"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/Track2/G-P/binary}"
DATA_ROOT="${DATA_ROOT:-$(dataset_root_for_split "Young")}"
SPLIT_CSV="${SPLIT_CSV:-$(split_csv_for_split "Young")}"
PERSONALITY_NPY="${PERSONALITY_NPY:-$(resolve_personality_npy "Young")}"
LOGS_DIR="${LOGS_DIR:-logs/test_scripts}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

mapfile -t CHECKPOINTS < <(find "$CHECKPOINT_DIR" -type f -name "best_model_*.pth" | sort)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No checkpoint found in: $CHECKPOINT_DIR" >&2
  exit 1
fi

if [[ ! -f "$PERSONALITY_NPY" ]]; then
  echo "Personality embeddings not found. Set PERSONALITY_NPY explicitly." >&2
  exit 1
fi

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
  echo "[Track2][G+P][binary] $CHECKPOINT"
  "$PYTHON_BIN" test.py \
    --checkpoint "$CHECKPOINT" \
    --data_root "$DATA_ROOT" \
    --split_csv "$SPLIT_CSV" \
    --personality_npy "$PERSONALITY_NPY" \
    --device "$DEVICE" \
    --logs_dir "$LOGS_DIR" \
    "$@"
done
