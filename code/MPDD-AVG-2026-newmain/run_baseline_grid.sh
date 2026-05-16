#!/usr/bin/env bash
# Run a representative baseline grid:
#   2 tracks (Track1 Elder / Track2 Young)
#   2 tasks  (binary / ternary)
#   3 subtracks (G+P / A-V+P / A-V-G+P)
# Canonical features for audio/video subtracks: wav2vec + resnet, bilstm_mean encoder.
# Uses the hyperparameters from scripts/Track*/A-V-G+P/run_*.sh (without the
# unsupported --selection_metric / --weighted_sampler / --label_smoothing flags).
set -uo pipefail
cd "$(dirname "$0")"

TRACK=$1        # Track1 | Track2
TASK=$2         # binary | ternary
SUBTRACK=$3     # G+P | A-V+P | A-V-G+P
GPU=$4

if [[ "$TRACK" == "Track1" ]]; then
  DATA_ROOT="MPDD-AVG2026/MPDD-AVG2026-trainval/Elder"
else
  DATA_ROOT="MPDD-AVG2026/MPDD-AVG2026-trainval/Young"
fi
SPLIT_CSV="$DATA_ROOT/split_labels_train.csv"
PERS_NPY="$DATA_ROOT/descriptions_embeddings_with_ids.npy"

EXP="baseline_${TRACK}_${TASK}_${SUBTRACK//+/p}_wav2vec_resnet"

CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --track $TRACK \
  --task $TASK \
  --subtrack "$SUBTRACK" \
  --encoder_type bilstm_mean \
  --audio_feature wav2vec \
  --video_feature resnet \
  --experiment_name "$EXP" \
  --data_root "$DATA_ROOT" \
  --split_csv "$SPLIT_CSV" \
  --personality_npy "$PERS_NPY" \
  --seed 42 \
  --val_ratio 0.1 \
  --epochs 140 \
  --batch_size 4 \
  --lr 8e-5 \
  --weight_decay 1e-5 \
  --hidden_dim 128 \
  --dropout 0.45 \
  --patience 35 \
  --min_delta 1e-4 \
  --target_t 128 \
  --device cuda
