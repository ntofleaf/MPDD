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
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.4}"
PATIENCE="${PATIENCE:-20}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
TARGET_T="${TARGET_T:-128}"
SELECTION_CCC_FLOOR="${SELECTION_CCC_FLOOR:-0.1}"
AUDIO_FEATURE="${AUDIO_FEATURE:-opensmile}"
VIDEO_FEATURE="${VIDEO_FEATURE:-resnet}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/Track1/A-V-P/binary/best_model_2026-04-30-08.18.33.pth}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-track1_elder_continue_avp_binary_20260430_ccc01_v8}"

echo "[Track1][A-V+P][binary] audio=${AUDIO_FEATURE} video=${VIDEO_FEATURE} seed=${SEED} target_t=${TARGET_T}"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$PROJECT_ROOT/train.py" \
  --track Track1 \
  --task binary \
  --subtrack A-V+P \
  --encoder_type bilstm_mean \
  --audio_feature "$AUDIO_FEATURE" \
  --video_feature "$VIDEO_FEATURE" \
  --experiment_name "$EXPERIMENT_NAME" \
  --init_checkpoint "$INIT_CHECKPOINT" \
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
  --selection_metric acc_with_ccc_floor \
  --selection_ccc_floor "$SELECTION_CCC_FLOOR" \
  --cls_loss_weight 1.0 \
  --reg_loss_weight 1.4 \
  --label_smoothing 0.02 \
  "$@"
