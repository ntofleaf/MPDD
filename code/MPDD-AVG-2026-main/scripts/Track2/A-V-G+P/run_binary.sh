#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.5}"
PATIENCE="${PATIENCE:-20}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
TARGET_T="${TARGET_T:-128}"
AUDIO_FEATURES=(mfcc opensmile wav2vec)
VIDEO_FEATURES=(densenet resnet openface)

for AUDIO_FEATURE in "${AUDIO_FEATURES[@]}"; do
  for VIDEO_FEATURE in "${VIDEO_FEATURES[@]}"; do
    echo "[Track2][A-V-G+P][binary] audio=${AUDIO_FEATURE} video=${VIDEO_FEATURE}"
    "$PYTHON_BIN" "$PROJECT_ROOT/train.py" \
      --track Track2 \
      --task binary \
      --subtrack A-V-G+P \
      --encoder_type bilstm_mean \
      --audio_feature "$AUDIO_FEATURE" \
      --video_feature "$VIDEO_FEATURE" \
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
  done
done
