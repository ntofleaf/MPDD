#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
LOGS_DIR="${LOGS_DIR:-logs}"
DATA_ROOT="${DATA_ROOT:-MPDD-AVG2026/MPDD-AVG2026-trainval/Elder}"
SPLIT_CSV="${SPLIT_CSV:-MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/split_labels_train.csv}"
PERSONALITY_NPY="${PERSONALITY_NPY:-MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/descriptions_embeddings_with_ids.npy}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.1}"
EPOCHS="${EPOCHS:-140}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
HIDDEN_DIM="${HIDDEN_DIM:-160}"
DROPOUT="${DROPOUT:-0.5}"
PATIENCE="${PATIENCE:-30}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
TARGET_T="${TARGET_T:-128}"
AUDIO_FEATURE="${AUDIO_FEATURE:-mfcc}"
VIDEO_FEATURE="${VIDEO_FEATURE:-resnet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-track1_elder_final6_avp_ternary_mfcc_resnet_h160}"

echo "[Track1][A-V+P][ternary] audio=${AUDIO_FEATURE} video=${VIDEO_FEATURE} seed=${SEED} target_t=${TARGET_T}"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$PROJECT_ROOT/train.py" \
  --track Track1 \
  --task ternary \
  --subtrack A-V+P \
  --encoder_type bilstm_mean \
  --audio_feature "$AUDIO_FEATURE" \
  --video_feature "$VIDEO_FEATURE" \
  --experiment_name "$EXPERIMENT_NAME" \
  --data_root "$DATA_ROOT" \
  --split_csv "$SPLIT_CSV" \
  --personality_npy "$PERSONALITY_NPY" \
  --seed "$SEED" \
  --val_ratio "$VAL_RATIO" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --hidden_dim "$HIDDEN_DIM" \
  --dropout "$DROPOUT" \
  --patience "$PATIENCE" \
  --min_delta "$MIN_DELTA" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --logs_dir "$LOGS_DIR" \
  --target_t "$TARGET_T" \
  --device "$DEVICE" \
  "$@"
