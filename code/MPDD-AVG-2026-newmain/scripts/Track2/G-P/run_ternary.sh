#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.5}"
PATIENCE="${PATIENCE:-20}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
TARGET_T="${TARGET_T:-128}"

"$PYTHON_BIN" "$PROJECT_ROOT/train.py" \
  --track Track2 \
  --task ternary \
  --subtrack G+P \
  --encoder_type bilstm_mean \
  --data_root MPDD-AVG2026/MPDD-AVG2026-trainval/Young \
  --split_csv MPDD-AVG2026/MPDD-AVG2026-trainval/Young/split_labels_train.csv \
  --personality_npy MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --hidden_dim "$HIDDEN_DIM" \
  --dropout "$DROPOUT" \
  --patience "$PATIENCE" \
  --min_delta "$MIN_DELTA" \
  --target_t "$TARGET_T" \
  --device "$DEVICE" \
  "$@"
