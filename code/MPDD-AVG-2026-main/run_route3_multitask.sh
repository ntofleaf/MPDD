#!/usr/bin/env bash
# =============================================================================
#  Route 3 — ordinal_multitask_consistency (Champion-style)
#  GPU 6 | thresholds 按 track 切换 | pos_weight=[0.5,8] | mask_min_pos=5
#         empirical | multitask=ord_bin_ter
#         loss_weights=ord:1.0 bin:0.15 ter:0.30 phq_l1:0.05 cons:0.10
#
#  注意：R3 只训 regression task。一个 ckpt 同时含 ord/bin/ter 三头，
#        ensemble 时三个任务的 ckpt 列表都指向同一组 multitask ckpt。
#        binary_ckpts/ternary_ckpts 通过 target_task 路由到对应 head。
#
#  用法：
#    bash run_route3_multitask.sh Track2          # 全量 5 折 + 300 epoch
#    bash run_route3_multitask.sh Track2 --smoke  # 仅 fold_0 + 30 epoch
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")" || exit 1

TRACK="${1:-Track2}"
MODE="${2:-full}"

ROUTE_TAG="route3_multitask"
GPU_IDS="${GPU_IDS:-6}"

# ── 路线 3 专属超参 ─────────────────────────────────────────────────────────
ORDINAL_POS_WEIGHT_CLAMP=8.0
ORDINAL_POS_WEIGHT_MIN=0.5
ORDINAL_MASK_MIN_POS=5
USE_EMPIRICAL="yes"
MULTITASK_MODE="ord_bin_ter"
LOSS_WEIGHTS="1.0,0.15,0.30,0.05,0.10"   # ord, bin, ter, phq_l1, consistency
MULTITASK_PHQ_WARMUP=30
BIN_BOUNDARY=5
TER_BOUNDARIES="5,10"

case "${TRACK}" in
    Track1)
        SUBTRACK="A-V-G+P"
        TRAIN_DATA_ROOT="/home/niutao/data/datasets_MPDD/Train-MPDD-Elder"
        TEST_DATA_ROOT="/home/niutao/data/datasets_MPDD/Test-MPDD-Elder/Elder"
        KFOLD_TRACK_TAG="Track1_Elder"
        ORDINAL_THRESHOLDS="3,5,10,15"
        ;;
    Track2)
        SUBTRACK="A-V+P"
        TRAIN_DATA_ROOT="/home/niutao/data/datasets_MPDD/Train-MPDD-Young"
        TEST_DATA_ROOT="/home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young"
        KFOLD_TRACK_TAG="Track2_Young"
        ORDINAL_THRESHOLDS="3,5,8,10"
        ;;
    *) echo "ERROR: TRACK must be Track1 or Track2"; exit 1 ;;
esac

PERSONALITY_NPY="${TRAIN_DATA_ROOT}/descriptions_embeddings_with_ids.npy"
SPLIT_CSV="${TRAIN_DATA_ROOT}/split_labels_train.csv"

if [ "${MODE}" = "--smoke" ]; then
    EPOCHS=30; FOLD_LIST="0"; EXP_TAG="${ROUTE_TAG}_smoke"
else
    EPOCHS=300; FOLD_LIST="0 1 2 3 4"; EXP_TAG="${ROUTE_TAG}"
fi

ENCODER_TYPE="bilstm_mean"; HIDDEN_DIM=64; DROPOUT=0.5
AUDIO_FEATURE="mfcc"; VIDEO_FEATURE="densenet"
BATCH_SIZE=2; LR=5e-5; PATIENCE=40; WEIGHT_DECAY=1e-3; SEED=42; N_FOLDS=5

KFOLD_BASE_DIR="kfold_splits"
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
    --track "${TRACK}" --subtrack "${SUBTRACK}" --encoder_type "${ENCODER_TYPE}"
    --audio_feature "${AUDIO_FEATURE}" --video_feature "${VIDEO_FEATURE}"
    --data_root "${TRAIN_DATA_ROOT}" --split_csv "${SPLIT_CSV}"
    --personality_npy "${PERSONALITY_NPY}"
    --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --lr "${LR}"
    --hidden_dim "${HIDDEN_DIM}" --dropout "${DROPOUT}"
    --patience "${PATIENCE}" --weight_decay "${WEIGHT_DECAY}"
    --seed "${SEED}" --device cuda
)

REG_ROUTE_ARGS=(
    --use_ordinal
    --ordinal_thresholds        "${ORDINAL_THRESHOLDS}"
    --ordinal_pos_weight_clamp  "${ORDINAL_POS_WEIGHT_CLAMP}"
    --ordinal_pos_weight_min    "${ORDINAL_POS_WEIGHT_MIN}"
    --ordinal_mask_min_pos      "${ORDINAL_MASK_MIN_POS}"
    --multitask_mode            "${MULTITASK_MODE}"
    --loss_weights              "${LOSS_WEIGHTS}"
    --multitask_phq_warmup      "${MULTITASK_PHQ_WARMUP}"
    --bin_boundary              "${BIN_BOUNDARY}"
    --ter_boundaries            "${TER_BOUNDARIES}"
    --ordinal_use_empirical_midpoints
)

{
echo "================================================================"
echo "  Route 3 (multitask)  GPU=${GPU_IDS}  TRACK=${TRACK}"
echo "  thresholds=${ORDINAL_THRESHOLDS}  pos_weight=[${ORDINAL_POS_WEIGHT_MIN},${ORDINAL_POS_WEIGHT_CLAMP}]"
echo "  mask_min_pos=${ORDINAL_MASK_MIN_POS}  multitask_mode=${MULTITASK_MODE}"
echo "  loss_weights=${LOSS_WEIGHTS}  phq_warmup=${MULTITASK_PHQ_WARMUP}"
echo "  bin_boundary=${BIN_BOUNDARY}  ter_boundaries=${TER_BOUNDARIES}"
echo "  EPOCHS=${EPOCHS}  FOLDS='${FOLD_LIST}'  LOG=${LOG_FILE}"
echo "================================================================"
} | tee -a "${LOG_FILE}"

# R3 只训 regression（multitask ckpt 自带 bin/ter 头）
generate_fold_csvs_if_needed() {
    local TASK="$1"; local OUT_DIR="$2"
    local LAST_FOLD="${OUT_DIR}/fold_$(( N_FOLDS - 1 )).csv"
    [ -f "${LAST_FOLD}" ] && { echo "  [OK] fold CSV: ${OUT_DIR}"; return 0; }
    python generate_kfold_splits.py --split_csv "${SPLIT_CSV}" --task "${TASK}" \
        --out_dir "${OUT_DIR}" --n_folds "${N_FOLDS}" --seed "${SEED}"
}
generate_fold_csvs_if_needed "regression" "${REG_KFOLD_DIR}" | tee -a "${LOG_FILE}"

REG_CKPTS=()
for FOLD_IDX in ${FOLD_LIST}; do
    {
    echo ""
    echo "## ${ROUTE_TAG} | ${TRACK} | Fold ${FOLD_IDX}"
    } | tee -a "${LOG_FILE}"

    REG_FOLD_CSV="${REG_KFOLD_DIR}/fold_${FOLD_IDX}.csv"
    REG_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/regression/fold_${FOLD_IDX}_${EXP_TAG}"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py --task regression \
        --regression_label label2 \
        --fold_csv "${REG_FOLD_CSV}" \
        --experiment_name "fold_${FOLD_IDX}_${EXP_TAG}" \
        "${REG_ROUTE_ARGS[@]}" "${COMMON_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    REG_CKPT=$(ls -t "${REG_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
    [ -z "${REG_CKPT}" ] && { echo "[ERROR] no regression ckpt"; exit 1; }
    REG_CKPTS+=("${REG_CKPT}")
done

# multitask ckpt 同时给 binary/ternary/regression 三个 ensemble 用
if [ "${MODE}" != "--smoke" ]; then
    OUTPUT_DIR="make_submission_forcodabench/my_submissions/${EXP_TAG}_${TRACK}_${TIMESTAMP}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python make_submission_forcodabench/ensemble_infer_and_pack.py \
        --binary_ckpts   "${REG_CKPTS[@]}" \
        --ternary_ckpts  "${REG_CKPTS[@]}" \
        --reg_ckpts      "${REG_CKPTS[@]}" \
        --test_root      "${TEST_DATA_ROOT}" \
        --personality    "${PERSONALITY_NPY}" \
        --output_dir     "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    echo "✅ Submission: ${OUTPUT_DIR}/submission.zip" | tee -a "${LOG_FILE}"
else
    echo "[smoke] 跳过 ensemble，仅完成 fold_0 训练验证" | tee -a "${LOG_FILE}"
fi
