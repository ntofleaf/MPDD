#!/usr/bin/env bash
# =============================================================================
#  Route 1 — ordinal_v1 (clean baseline)
#  GPU 4 | thresholds=[5,10,15,20] | pos_weight_clamp=5 | no mask | no empirical
#
#  用法：
#    bash run_route1_ordinal_v1.sh Track2          # 全量 5 折 + 300 epoch
#    bash run_route1_ordinal_v1.sh Track2 --smoke  # 仅 fold_0 + 30 epoch (smoke test)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")" || exit 1

# ── 命令行 ──────────────────────────────────────────────────────────────────
TRACK="${1:-Track2}"
MODE="${2:-full}"   # full | --smoke

ROUTE_TAG="route1_ordinal_v1"
GPU_IDS="${GPU_IDS:-4}"

# ── 路线 1 专属超参 ─────────────────────────────────────────────────────────
ORDINAL_THRESHOLDS="5,10,15,20"
ORDINAL_POS_WEIGHT_CLAMP=5.0
ORDINAL_POS_WEIGHT_MIN=0.5
ORDINAL_MASK_MIN_POS=0           # R1 不 mask（干净对照）
USE_EMPIRICAL=""                 # R1 用几何中点
MULTITASK_MODE="off"

# ── 共享：track 相关数据路径 / subtrack ─────────────────────────────────────
case "${TRACK}" in
    Track1)
        SUBTRACK="A-V-G+P"
        TRAIN_DATA_ROOT="/home/niutao/data/datasets_MPDD/Train-MPDD-Elder"
        TEST_DATA_ROOT="/home/niutao/data/datasets_MPDD/Test-MPDD-Elder/Elder"
        KFOLD_TRACK_TAG="Track1_Elder"
        ;;
    Track2)
        SUBTRACK="A-V+P"
        TRAIN_DATA_ROOT="/home/niutao/data/datasets_MPDD/Train-MPDD-Young"
        TEST_DATA_ROOT="/home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young"
        KFOLD_TRACK_TAG="Track2_Young"
        ;;
    *) echo "ERROR: TRACK must be Track1 or Track2, got '${TRACK}'"; exit 1 ;;
esac

PERSONALITY_NPY="${TRAIN_DATA_ROOT}/descriptions_embeddings_with_ids.npy"
SPLIT_CSV="${TRAIN_DATA_ROOT}/split_labels_train.csv"

# ── smoke vs full ───────────────────────────────────────────────────────────
if [ "${MODE}" = "--smoke" ]; then
    EPOCHS=30
    FOLD_LIST="0"
    EXP_TAG="${ROUTE_TAG}_smoke"
else
    EPOCHS=300
    FOLD_LIST="0 1 2 3 4"
    EXP_TAG="${ROUTE_TAG}"
fi

# ── 共享训练超参（与 run_kfold_ensemble_pipeline.sh 保持一致） ──────────────
ENCODER_TYPE="bilstm_mean"
HIDDEN_DIM=64
DROPOUT=0.5
AUDIO_FEATURE="mfcc"
VIDEO_FEATURE="densenet"
BATCH_SIZE=2
LR=5e-5
PATIENCE=40
WEIGHT_DECAY=1e-3
SEED=42
N_FOLDS=5

KFOLD_BASE_DIR="kfold_splits"
BINARY_KFOLD_DIR="${KFOLD_BASE_DIR}/${KFOLD_TRACK_TAG}_binary"
TERNARY_KFOLD_DIR="${KFOLD_BASE_DIR}/${KFOLD_TRACK_TAG}_ternary"
REG_KFOLD_DIR="${KFOLD_BASE_DIR}/${KFOLD_TRACK_TAG}_regression"

mkdir -p parallel_logs
TIMESTAMP=$(date +"%Y-%m-%d_%H.%M.%S")
LOG_FILE="parallel_logs/${EXP_TAG}_${TRACK}_${TIMESTAMP}.log"

case "${SUBTRACK}" in
    "A-V+P")   SUBTRACK_DIR="A-V-P"   ;;
    "A-V-G+P") SUBTRACK_DIR="A-V-G+P" ;;
    "G+P")     SUBTRACK_DIR="G-P"     ;;
    *)         SUBTRACK_DIR="${SUBTRACK}" ;;
esac

CKPT_ROOT="checkpoints"

COMMON_ARGS=(
    --track         "${TRACK}"
    --subtrack      "${SUBTRACK}"
    --encoder_type  "${ENCODER_TYPE}"
    --audio_feature "${AUDIO_FEATURE}"
    --video_feature "${VIDEO_FEATURE}"
    --data_root     "${TRAIN_DATA_ROOT}"
    --split_csv     "${SPLIT_CSV}"
    --personality_npy "${PERSONALITY_NPY}"
    --epochs        "${EPOCHS}"
    --batch_size    "${BATCH_SIZE}"
    --lr            "${LR}"
    --hidden_dim    "${HIDDEN_DIM}"
    --dropout       "${DROPOUT}"
    --patience      "${PATIENCE}"
    --weight_decay  "${WEIGHT_DECAY}"
    --seed          "${SEED}"
    --device        cuda
)

REG_ROUTE_ARGS=(
    --use_ordinal
    --ordinal_thresholds        "${ORDINAL_THRESHOLDS}"
    --ordinal_pos_weight_clamp  "${ORDINAL_POS_WEIGHT_CLAMP}"
    --ordinal_pos_weight_min    "${ORDINAL_POS_WEIGHT_MIN}"
    --ordinal_mask_min_pos      "${ORDINAL_MASK_MIN_POS}"
    --multitask_mode            "${MULTITASK_MODE}"
)
if [ -n "${USE_EMPIRICAL}" ]; then
    REG_ROUTE_ARGS+=(--ordinal_use_empirical_midpoints)
fi

# ── Banner ──────────────────────────────────────────────────────────────────
{
echo "================================================================"
echo "  Route 1 (ordinal_v1 clean baseline)  GPU=${GPU_IDS}  TRACK=${TRACK}"
echo "  thresholds=${ORDINAL_THRESHOLDS}  pos_weight_clamp=${ORDINAL_POS_WEIGHT_CLAMP}"
echo "  mask_min_pos=${ORDINAL_MASK_MIN_POS}  empirical_midpoints=${USE_EMPIRICAL:-off}"
echo "  EPOCHS=${EPOCHS}  FOLDS='${FOLD_LIST}'  LOG=${LOG_FILE}"
echo "================================================================"
} | tee -a "${LOG_FILE}"

# ── Step 0: 生成 fold CSV (幂等) ────────────────────────────────────────────
generate_fold_csvs_if_needed() {
    local TASK="$1"; local OUT_DIR="$2"
    local LAST_FOLD="${OUT_DIR}/fold_$(( N_FOLDS - 1 )).csv"
    if [ -f "${LAST_FOLD}" ]; then
        echo "  [OK] fold CSV 已存在: ${OUT_DIR}"; return 0
    fi
    python generate_kfold_splits.py \
        --split_csv "${SPLIT_CSV}" --task "${TASK}" \
        --out_dir "${OUT_DIR}" --n_folds "${N_FOLDS}" --seed "${SEED}"
}

generate_fold_csvs_if_needed "binary"     "${BINARY_KFOLD_DIR}"  | tee -a "${LOG_FILE}"
generate_fold_csvs_if_needed "ternary"    "${TERNARY_KFOLD_DIR}" | tee -a "${LOG_FILE}"
generate_fold_csvs_if_needed "regression" "${REG_KFOLD_DIR}"     | tee -a "${LOG_FILE}"

# ── Step 1: 5 折训练 (binary + ternary + regression) ────────────────────────
BINARY_CKPTS=(); TERNARY_CKPTS=(); REG_CKPTS=()

for FOLD_IDX in ${FOLD_LIST}; do
    {
    echo ""
    echo "################################################################"
    echo "##  ${ROUTE_TAG} | ${TRACK} | Fold ${FOLD_IDX}"
    echo "################################################################"
    } | tee -a "${LOG_FILE}"

    BINARY_FOLD_CSV="${BINARY_KFOLD_DIR}/fold_${FOLD_IDX}.csv"
    TERNARY_FOLD_CSV="${TERNARY_KFOLD_DIR}/fold_${FOLD_IDX}.csv"
    REG_FOLD_CSV="${REG_KFOLD_DIR}/fold_${FOLD_IDX}.csv"

    BINARY_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/binary/fold_${FOLD_IDX}_${EXP_TAG}"
    TERNARY_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/ternary/fold_${FOLD_IDX}_${EXP_TAG}"
    REG_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/regression/fold_${FOLD_IDX}_${EXP_TAG}"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py \
        --task binary  --fold_csv "${BINARY_FOLD_CSV}" \
        --experiment_name "fold_${FOLD_IDX}_${EXP_TAG}" \
        "${COMMON_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    BINARY_CKPT=$(ls -t "${BINARY_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
    [ -z "${BINARY_CKPT}" ] && { echo "[ERROR] no binary ckpt"; exit 1; }
    BINARY_CKPTS+=("${BINARY_CKPT}")

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py \
        --task ternary --fold_csv "${TERNARY_FOLD_CSV}" \
        --experiment_name "fold_${FOLD_IDX}_${EXP_TAG}" \
        "${COMMON_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    TERNARY_CKPT=$(ls -t "${TERNARY_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
    [ -z "${TERNARY_CKPT}" ] && { echo "[ERROR] no ternary ckpt"; exit 1; }
    TERNARY_CKPTS+=("${TERNARY_CKPT}")

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py \
        --task regression --regression_label label2 \
        --fold_csv "${REG_FOLD_CSV}" \
        --experiment_name "fold_${FOLD_IDX}_${EXP_TAG}" \
        "${REG_ROUTE_ARGS[@]}" "${COMMON_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    REG_CKPT=$(ls -t "${REG_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
    [ -z "${REG_CKPT}" ] && { echo "[ERROR] no regression ckpt"; exit 1; }
    REG_CKPTS+=("${REG_CKPT}")
done

# ── Step 2: ensemble 推理 (smoke 模式跳过) ──────────────────────────────────
if [ "${MODE}" != "--smoke" ]; then
    OUTPUT_DIR="make_submission_forcodabench/my_submissions/${EXP_TAG}_${TRACK}_${TIMESTAMP}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python make_submission_forcodabench/ensemble_infer_and_pack.py \
        --binary_ckpts   "${BINARY_CKPTS[@]}" \
        --ternary_ckpts  "${TERNARY_CKPTS[@]}" \
        --reg_ckpts      "${REG_CKPTS[@]}" \
        --test_root      "${TEST_DATA_ROOT}" \
        --personality    "${PERSONALITY_NPY}" \
        --output_dir     "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    echo "✅ Submission: ${OUTPUT_DIR}/submission.zip" | tee -a "${LOG_FILE}"
else
    echo "[smoke] 跳过 ensemble，仅完成 fold_0 训练验证" | tee -a "${LOG_FILE}"
fi
