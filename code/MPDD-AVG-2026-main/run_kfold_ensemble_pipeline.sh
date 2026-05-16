#!/usr/bin/env bash
# =============================================================================
#  run_kfold_ensemble_pipeline.sh
#  MPDD-AVG-2026 全流程脚本（五折交叉验证 + 5 折集成推理）
#
#  与 run_best_single_pipeline.sh 的区别
#  ─────────────────────────────────────────────────────────────────────────────
#  run_best_single_pipeline.sh：训练 5 折，推理时从中挑 val 最优的 1 个 checkpoint
#  本脚本（run_kfold_ensemble_pipeline.sh）：
#      训练 5 折，推理时把 5 个 checkpoint **全部传入**集成脚本做预测平均
#        binary   : 5 个 fold 各出 1 个 checkpoint → 5 路概率平均 → argmax
#        ternary  : 同上
#        regression: 5 个 fold 各出 1 个 checkpoint → 5 路 PHQ-9 预测值平均
#
#  流程
#  ─────────────────────────────────────────────────────────────────────────────
#  Step 0: 幂等生成 fold CSV（若已存在则跳过，只需生成一次）
#  Step 1: 五折训练（N_FOLDS × 3 任务）
#  Step 2: 5 折集成推理 + 打包提交（全部 checkpoint 参与平均）
#
#  用法：
#    bash run_kfold_ensemble_pipeline.sh
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1. 显卡设置
# ─────────────────────────────────────────────────────────────────────────────
GPU_IDS="1"
NCCL_P2P_DISABLE=1

# ─────────────────────────────────────────────────────────────────────────────
# 2. 任务与数据
# ─────────────────────────────────────────────────────────────────────────────
TRACK="Track2"
SUBTRACK="A-V+P"

# ─────────────────────────────────────────────────────────────────────────────
# 3. 模型结构
# ─────────────────────────────────────────────────────────────────────────────
ENCODER_TYPE="bilstm_mean"
HIDDEN_DIM=64
DROPOUT=0.5

# ─────────────────────────────────────────────────────────────────────────────
# 4. 特征
# ─────────────────────────────────────────────────────────────────────────────
AUDIO_FEATURE="mfcc"
VIDEO_FEATURE="densenet"

# ─────────────────────────────────────────────────────────────────────────────
# 5. 数据路径
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_DATA_ROOT="/home/niutao/data/datasets_MPDD/Train-MPDD-Young"
TEST_DATA_ROOT="/home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young"
PERSONALITY_NPY="${TRAIN_DATA_ROOT}/descriptions_embeddings_with_ids.npy"
SPLIT_CSV="${TRAIN_DATA_ROOT}/split_labels_train.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 6. 训练超参数
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS=300
BATCH_SIZE=2
LR=5e-5
PATIENCE=40
WEIGHT_DECAY=1e-3
# 注：五折模式下不使用 VAL_RATIO，由 fold CSV 固定分割

# ─────────────────────────────────────────────────────────────────────────────
# 7. 五折交叉验证设置
#
#    与 run_best_single_pipeline.sh 共用同一套 fold CSV（KFOLD_BASE_DIR 相同）
#    fold CSV 只需生成一次，两个脚本可互相复用
# ─────────────────────────────────────────────────────────────────────────────
N_FOLDS=5
SEED=42
KFOLD_BASE_DIR="kfold_splits"
BINARY_KFOLD_DIR="${KFOLD_BASE_DIR}/Track2_Young_binary"
TERNARY_KFOLD_DIR="${KFOLD_BASE_DIR}/Track2_Young_ternary"
REG_KFOLD_DIR="${KFOLD_BASE_DIR}/Track2_Young_regression"

# ─────────────────────────────────────────────────────────────────────────────
# 8. 回归任务配置
#    方案 A v1 (2026-05-15): 启用 ordinal BCE 替代 CCCLoss
#      USE_ORDINAL=true → 模型输出 4 个 logits（PHQ>=5/10/15/20），BCE+pos_weight 训练
#      USE_ORDINAL=false → 原 CCCLoss + MSE warmup 路径
#    ordinal 模式下 mse_warmup 自动失效，无需关心
# ─────────────────────────────────────────────────────────────────────────────
REGRESSION_LABEL="label2"
MSE_WARMUP_EPOCHS=60
USE_ORDINAL=true

# ─────────────────────────────────────────────────────────────────────────────
# 9. 多卡 batch 自动缩放
# ─────────────────────────────────────────────────────────────────────────────
AUTO_SCALE_BATCH=false

# =============================================================================
#  以下内容一般不需要修改
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")" || exit 1

TIMESTAMP=$(date +"%Y-%m-%d_%H.%M.%S")
OUTPUT_DIR="make_submission_forcodabench/my_submissions/kfold_ensemble_${TIMESTAMP}"

NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | grep -c '[0-9]')
ACTUAL_BATCH=${BATCH_SIZE}
ACTUAL_LR=${LR}
if [ "${AUTO_SCALE_BATCH}" = "true" ] && [ "${NUM_GPUS}" -gt 1 ]; then
    ACTUAL_BATCH=$(( BATCH_SIZE * NUM_GPUS ))
    ACTUAL_LR=$(awk "BEGIN {printf \"%.2e\", ${LR} * ${NUM_GPUS}}")
fi

case "${SUBTRACK}" in
    "A-V+P")   SUBTRACK_DIR="A-V-P"   ;;
    "A-V-G+P") SUBTRACK_DIR="A-V-G+P" ;;
    "G+P")     SUBTRACK_DIR="G-P"     ;;
    *)         SUBTRACK_DIR="${SUBTRACK}" ;;
esac

CKPT_ROOT="checkpoints"

# COMMON_ARGS：五折模式下不含 --val_ratio
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
    --batch_size    "${ACTUAL_BATCH}"
    --lr            "${ACTUAL_LR}"
    --hidden_dim    "${HIDDEN_DIM}"
    --dropout       "${DROPOUT}"
    --patience      "${PATIENCE}"
    --weight_decay  "${WEIGHT_DECAY}"
    --seed          "${SEED}"
    --device        cuda
)

echo ""
echo "================================================================"
echo "  [五折集成模式] 5 折全部参与推理平均"
echo "  GPU         : ${GPU_IDS}  (${NUM_GPUS} card(s))"
echo "  Track       : ${TRACK}  Subtrack: ${SUBTRACK}"
echo "  Encoder     : ${ENCODER_TYPE}  hidden_dim=${HIDDEN_DIM}  dropout=${DROPOUT}"
echo "  Feature     : audio=${AUDIO_FEATURE}  video=${VIDEO_FEATURE}"
echo "  LR=${ACTUAL_LR}  batch=${ACTUAL_BATCH}  epochs=${EPOCHS}  patience=${PATIENCE}"
echo "  seed=${SEED}  N_FOLDS=${N_FOLDS}  (共 ${N_FOLDS} × 3 任务 = $(( N_FOLDS * 3 )) 次训练)"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 0: 幂等生成 fold CSV（若已存在则跳过）
# ─────────────────────────────────────────────────────────────────────────────
generate_fold_csvs_if_needed() {
    local TASK="$1"
    local OUT_DIR="$2"
    local LAST_FOLD="${OUT_DIR}/fold_$(( N_FOLDS - 1 )).csv"

    if [ -f "${LAST_FOLD}" ]; then
        echo "  [OK] fold CSV 已存在，跳过生成: ${OUT_DIR}"
        return 0
    fi

    echo "  [生成] task=${TASK}  out_dir=${OUT_DIR}"
    python generate_kfold_splits.py \
        --split_csv "${SPLIT_CSV}"  \
        --task      "${TASK}"       \
        --out_dir   "${OUT_DIR}"    \
        --n_folds   "${N_FOLDS}"    \
        --seed      42
    echo "  [完成] ${TASK} fold CSV 已写入 ${OUT_DIR}"
}

echo ""
echo "################################################################"
echo "##  Step 0: 检查 / 生成 fold CSV（每任务独立分层）"
echo "################################################################"
generate_fold_csvs_if_needed "binary"     "${BINARY_KFOLD_DIR}"
generate_fold_csvs_if_needed "ternary"    "${TERNARY_KFOLD_DIR}"
generate_fold_csvs_if_needed "regression" "${REG_KFOLD_DIR}"
echo ""
echo "  ✅ fold CSV 就绪，开始五折训练..."

# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: 五折训练（每折各训练 binary / ternary / regression）
# ─────────────────────────────────────────────────────────────────────────────
BINARY_CKPTS=()
TERNARY_CKPTS=()
REG_CKPTS=()

for FOLD_IDX in $(seq 0 $(( N_FOLDS - 1 ))); do
    echo ""
    echo "################################################################"
    echo "##  Fold ${FOLD_IDX} / $(( N_FOLDS - 1 ))"
    echo "################################################################"

    BINARY_FOLD_CSV="${BINARY_KFOLD_DIR}/fold_${FOLD_IDX}.csv"
    TERNARY_FOLD_CSV="${TERNARY_KFOLD_DIR}/fold_${FOLD_IDX}.csv"
    REG_FOLD_CSV="${REG_KFOLD_DIR}/fold_${FOLD_IDX}.csv"

    BINARY_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/binary/fold_${FOLD_IDX}"
    TERNARY_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/ternary/fold_${FOLD_IDX}"
    REG_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/regression/fold_${FOLD_IDX}"

    # ── Fold N.A: Binary ────────────────────────────────────────────────────
    echo ""
    echo "  >> [Fold ${FOLD_IDX}.A] 训练 Binary（二分类）"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
        python train.py \
            --task            binary                  \
            --fold_csv        "${BINARY_FOLD_CSV}"    \
            --experiment_name "fold_${FOLD_IDX}"      \
            "${COMMON_ARGS[@]}"

    BINARY_CKPT=$(ls -t "${BINARY_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
    if [ -z "${BINARY_CKPT}" ]; then
        echo "  [ERROR] 未找到 binary checkpoint！查找路径: ${BINARY_CKPT_DIR}"
        exit 1
    fi
    BINARY_CKPTS+=("${BINARY_CKPT}")
    echo "  ✅ Binary  checkpoint [fold ${FOLD_IDX}]: ${BINARY_CKPT}"

    # ── Fold N.B: Ternary ───────────────────────────────────────────────────
    echo ""
    echo "  >> [Fold ${FOLD_IDX}.B] 训练 Ternary（三分类）"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
        python train.py \
            --task            ternary                 \
            --fold_csv        "${TERNARY_FOLD_CSV}"   \
            --experiment_name "fold_${FOLD_IDX}"      \
            "${COMMON_ARGS[@]}"

    TERNARY_CKPT=$(ls -t "${TERNARY_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
    if [ -z "${TERNARY_CKPT}" ]; then
        echo "  [ERROR] 未找到 ternary checkpoint！查找路径: ${TERNARY_CKPT_DIR}"
        exit 1
    fi
    TERNARY_CKPTS+=("${TERNARY_CKPT}")
    echo "  ✅ Ternary checkpoint [fold ${FOLD_IDX}]: ${TERNARY_CKPT}"

    # ── Fold N.C: Regression ────────────────────────────────────────────────
    echo ""
    echo "  >> [Fold ${FOLD_IDX}.C] 训练 Regression（PHQ-9 回归）"
    REG_EXTRA_ARGS=()
    if [ "${USE_ORDINAL}" = "true" ]; then
        REG_EXTRA_ARGS+=(--use_ordinal)
        echo "     [Ordinal Mode] 启用 ordinal BCE，忽略 mse_warmup"
    fi
    CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
        python train.py \
            --task              regression             \
            --regression_label  "${REGRESSION_LABEL}" \
            --fold_csv          "${REG_FOLD_CSV}"      \
            --experiment_name   "fold_${FOLD_IDX}"     \
            --mse_warmup_epochs "${MSE_WARMUP_EPOCHS}" \
            "${REG_EXTRA_ARGS[@]}"                     \
            "${COMMON_ARGS[@]}"

    REG_CKPT=$(ls -t "${REG_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
    if [ -z "${REG_CKPT}" ]; then
        echo "  [ERROR] 未找到 regression checkpoint！查找路径: ${REG_CKPT_DIR}"
        exit 1
    fi
    REG_CKPTS+=("${REG_CKPT}")
    echo "  ✅ Regression checkpoint [fold ${FOLD_IDX}]: ${REG_CKPT}"

    echo ""
    echo "  Fold ${FOLD_IDX} 全部任务训练完成 ✓"
done

# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: 5 折集成推理 + 打包
#
#  每个任务把全部 N_FOLDS 个 checkpoint 同时传入 ensemble_infer_and_pack.py：
#    binary   : 5 个 fold 各推一次 → softmax 概率平均 → argmax 取最终类别
#    ternary  : 同上
#    regression: 5 个 fold 各推一次 → PHQ-9 预测值平均
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "################################################################"
echo "##  Step 2: 5 折集成推理 + 提交打包"
echo "################################################################"
echo ""
echo "  Binary    checkpoints (${#BINARY_CKPTS[@]} 个):"
for c in "${BINARY_CKPTS[@]}";  do echo "    ${c}"; done
echo ""
echo "  Ternary   checkpoints (${#TERNARY_CKPTS[@]} 个):"
for c in "${TERNARY_CKPTS[@]}"; do echo "    ${c}"; done
echo ""
echo "  Regression checkpoints (${#REG_CKPTS[@]} 个):"
for c in "${REG_CKPTS[@]}";    do echo "    ${c}"; done
echo ""

CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
    python make_submission_forcodabench/ensemble_infer_and_pack.py \
        --binary_ckpts   "${BINARY_CKPTS[@]}"   \
        --ternary_ckpts  "${TERNARY_CKPTS[@]}"  \
        --reg_ckpts      "${REG_CKPTS[@]}"      \
        --test_root      "${TEST_DATA_ROOT}"     \
        --personality    "${PERSONALITY_NPY}"    \
        --output_dir     "${OUTPUT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: 汇总
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "################################################################"
echo "##  全流程完成！"
echo "################################################################"
echo ""
echo "  集成方式：${N_FOLDS} 折各出 1 个最优 checkpoint，共 3×${N_FOLDS} 个权重参与平均"
echo ""
echo "  fold CSV 目录（固定，下次运行自动复用）："
echo "    binary    : ${BINARY_KFOLD_DIR}"
echo "    ternary   : ${TERNARY_KFOLD_DIR}"
echo "    regression: ${REG_KFOLD_DIR}"
echo ""
echo "  输出目录：${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "  上传此文件到 CodaBench："
echo "  → ${OUTPUT_DIR}/submission.zip"
echo ""
