#!/usr/bin/env bash
# =============================================================================
#  run_kfold_ensemble_t2_young_expanded.sh
#  MPDD-AVG-2026 全流程脚本 —— Track2 · A-V+P（音频+视频+人格）
#  五折交叉验证 + 5 折集成推理
#
#  ★ 与 run_kfold_ensemble_pipeline.sh（原 Track2 Young 脚本）的区别：
#  ─────────────────────────────────────────────────────────────────────────────
#  1. fold CSV 目录（binary/ternary）：
#       原版: Track2_Young_binary      / Track2_Young_ternary
#       新版: Track2_Young_binary_expanded / Track2_Young_ternary_expanded
#          → 包含测试集22个伪标签样本（永远在 train，不进 val）
#          → 每折 train=101，val=9（三池设计，详见 generate_kfold_splits_expanded.py）
#
#  2. fold CSV 生成方式（Step 0）：
#       原版: generate_kfold_splits.py（仅用训练集 88 样本）
#       新版: generate_kfold_splits_expanded.py（88 训练 + 22 测试伪标签）
#          binary  测试标签来源: Test-MPDD-Young/Young/labels_binary.csv
#          ternary 测试标签来源: Test-MPDD-Young/Young/labels_3class.csv
#
#  3. checkpoint 子目录后缀：
#       原版: .../binary/fold_N     .../ternary/fold_N
#       新版: .../binary_exp/fold_N .../ternary_exp/fold_N
#          → 避免与原版 checkpoint 互相覆盖，两套训练可同时保留
#
#  4. 提交输出目录前缀：
#       原版: my_submissions/kfold_ensemble_TIMESTAMP
#       新版: my_submissions/t2_exp_kfold_ensemble_TIMESTAMP
#
#  5. 回归任务 fold CSV 不变：
#       仍使用 Track2_Young_regression（因为回归标签为估算PHQ-9，扩展意义不大）
#
#  流程
#  ─────────────────────────────────────────────────────────────────────────────
#  Step 0: 幂等生成 fold CSV（若已存在则跳过，只需生成一次）
#  Step 1: 五折训练（N_FOLDS × 3 任务：binary / ternary / regression）
#  Step 2: 5 折集成推理 + 打包提交（全部 checkpoint 参与平均）
#
#  用法：
#    bash run_kfold_ensemble_t2_young_expanded.sh
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
_ORIG_TRAIN_ROOT="/home/niutao/data/datasets_MPDD/Train-MPDD-Young"
TEST_DATA_ROOT="/home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young"

# ★ 修复：dataset._resolve_split_data_roots 通过父目录名 MPDD-AVG2026-trainval
#   来同时定位 train/test 两棵特征树。若父目录名不匹配，会 fallback 到
#   {train: data_root, test: data_root}，导致 test 伪标签样本在 train 目录
#   找不到特征文件，被静默跳过（101 - 22 = 79）。
#   解决方案：在项目内创建符号链接，使路径结构满足函数期待的命名规范。
_LINK_BASE="$(cd "$(dirname "$0")" && pwd)/data_links"
_TRAINVAL_LINK="${_LINK_BASE}/MPDD-AVG2026-trainval/Young"
_TEST_LINK="${_LINK_BASE}/MPDD-AVG2026-test/Young"
mkdir -p "$(dirname "${_TRAINVAL_LINK}")" "$(dirname "${_TEST_LINK}")"
[ -L "${_TRAINVAL_LINK}" ] || ln -sfn "${_ORIG_TRAIN_ROOT}" "${_TRAINVAL_LINK}"
[ -L "${_TEST_LINK}" ]     || ln -sfn "${TEST_DATA_ROOT}" "${_TEST_LINK}"

# train.py / dataset.py 收到此路径后，_resolve_split_data_roots 能正确识别
# train root = _TRAINVAL_LINK，test root = _TEST_LINK
TRAIN_DATA_ROOT="${_TRAINVAL_LINK}"
# 人格 npy / split CSV 仍指向原始物理路径（绕过符号链接，更稳定）
PERSONALITY_NPY="${_ORIG_TRAIN_ROOT}/descriptions_embeddings_with_ids.npy"
SPLIT_CSV="${_ORIG_TRAIN_ROOT}/split_labels_train.csv"

# ★ 测试集伪标签文件路径（用于生成扩展 fold CSV）
TEST_LABEL_BINARY="${TEST_DATA_ROOT}/labels_binary.csv"
TEST_LABEL_TERNARY="${TEST_DATA_ROOT}/labels_3class.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 6. 训练超参数
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS=400
BATCH_SIZE=32
LR=5e-4
PATIENCE=400
WEIGHT_DECAY=1e-3
# 注：五折模式下不使用 VAL_RATIO，由 fold CSV 固定分割

# ─────────────────────────────────────────────────────────────────────────────
# 7. ★ 扩展版五折交叉验证设置
#    binary/ternary 使用扩展 fold 目录（88训练 + 22测试），val=9
#    regression     使用原有 fold 目录（仅88训练），不变
# ─────────────────────────────────────────────────────────────────────────────
N_FOLDS=5
SEED=42
VAL_SIZE=9    # ★ 每折验证集固定9个（三池设计）
KFOLD_BASE_DIR="kfold_splits"

# ★ 扩展版 fold 目录（binary 和 ternary）
BINARY_KFOLD_DIR="${KFOLD_BASE_DIR}/Track2_Young_binary_expanded"
TERNARY_KFOLD_DIR="${KFOLD_BASE_DIR}/Track2_Young_ternary_expanded"
# 回归不做扩展，复用原有
REG_KFOLD_DIR="${KFOLD_BASE_DIR}/Track2_Young_regression"

# ─────────────────────────────────────────────────────────────────────────────
# 8. 回归任务配置
# ─────────────────────────────────────────────────────────────────────────────
REGRESSION_LABEL="label2"
MSE_WARMUP_EPOCHS=40

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
# ★ 输出目录前缀改为 t2_exp_，与原版区分
OUTPUT_DIR="make_submission_forcodabench/my_submissions/t2_exp_kfold_ensemble_${TIMESTAMP}"

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

# checkpoint 根目录
# train.py 的路径规则: checkpoints/{TRACK}/{SUBTRACK_DIR}/{task}/{experiment_name}/
# experiment_name = "fold_N_exp" → 实际路径 binary/fold_N_exp（与原版 binary/fold_N 隔离）
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
echo "  [Track2 A-V+P 扩展五折集成模式] 测试集伪标签并入训练"
echo "  GPU         : ${GPU_IDS}  (${NUM_GPUS} card(s))"
echo "  Track       : ${TRACK}  Subtrack: ${SUBTRACK}"
echo "  Encoder     : ${ENCODER_TYPE}  hidden_dim=${HIDDEN_DIM}  dropout=${DROPOUT}"
echo "  Feature     : audio=${AUDIO_FEATURE}  video=${VIDEO_FEATURE}"
echo "  LR=${ACTUAL_LR}  batch=${ACTUAL_BATCH}  epochs=${EPOCHS}  patience=${PATIENCE}"
echo "  seed=${SEED}  N_FOLDS=${N_FOLDS}  val_size=${VAL_SIZE}  (三池设计)"
echo "  Train data  : ${TRAIN_DATA_ROOT}"
echo "  Test  data  : ${TEST_DATA_ROOT}"
echo "  binary  fold: ${BINARY_KFOLD_DIR}"
echo "  ternary fold: ${TERNARY_KFOLD_DIR}"
echo "  reg     fold: ${REG_KFOLD_DIR}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 0: 幂等生成 fold CSV
#
#  ★ binary / ternary：调用扩展版脚本 generate_kfold_splits_expanded.py
#       参数：--train_csv / --test_label_csv / --val_size（三池，每折val=9）
#  ★ regression：调用原版脚本 generate_kfold_splits.py（不扩展）
# ─────────────────────────────────────────────────────────────────────────────
generate_expanded_fold_csvs_if_needed() {
    local TASK="$1"
    local OUT_DIR="$2"
    local TEST_LABEL_CSV="$3"
    local LAST_FOLD="${OUT_DIR}/fold_$(( N_FOLDS - 1 )).csv"

    if [ -f "${LAST_FOLD}" ]; then
        echo "  [OK] 扩展 fold CSV 已存在，跳过生成: ${OUT_DIR}"
        return 0
    fi

    echo "  [生成扩展] task=${TASK}  val_size=${VAL_SIZE}  out_dir=${OUT_DIR}"
    python generate_kfold_splits_expanded.py \
        --train_csv      "${SPLIT_CSV}"       \
        --test_label_csv "${TEST_LABEL_CSV}"  \
        --task           "${TASK}"            \
        --out_dir        "${OUT_DIR}"         \
        --n_folds        "${N_FOLDS}"         \
        --val_size       "${VAL_SIZE}"        \
        --seed           42
    echo "  [完成] ${TASK} 扩展 fold CSV 已写入 ${OUT_DIR}"
}

generate_reg_fold_csvs_if_needed() {
    local OUT_DIR="$1"
    local LAST_FOLD="${OUT_DIR}/fold_$(( N_FOLDS - 1 )).csv"

    if [ -f "${LAST_FOLD}" ]; then
        echo "  [OK] regression fold CSV 已存在，跳过生成: ${OUT_DIR}"
        return 0
    fi

    echo "  [生成] task=regression  out_dir=${OUT_DIR}"
    python generate_kfold_splits.py \
        --split_csv "${SPLIT_CSV}"  \
        --task      "regression"    \
        --out_dir   "${OUT_DIR}"    \
        --n_folds   "${N_FOLDS}"    \
        --seed      42
    echo "  [完成] regression fold CSV 已写入 ${OUT_DIR}"
}

echo ""
echo "################################################################"
echo "##  Step 0: 检查 / 生成 fold CSV"
echo "##  binary/ternary → 扩展版（含测试集伪标签），val=9"
echo "##  regression     → 原版（仅训练集）"
echo "################################################################"
generate_expanded_fold_csvs_if_needed "binary"  "${BINARY_KFOLD_DIR}"  "${TEST_LABEL_BINARY}"
generate_expanded_fold_csvs_if_needed "ternary" "${TERNARY_KFOLD_DIR}" "${TEST_LABEL_TERNARY}"
generate_reg_fold_csvs_if_needed      "${REG_KFOLD_DIR}"
echo ""
echo "  ✅ fold CSV 就绪，开始五折训练..."

# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: 五折训练
#
#  ★ binary / ternary：使用扩展 fold CSV（训练集含测试集伪标签样本）
#                       checkpoint 存入 binary_exp / ternary_exp 子目录
#  ★ regression：使用原版 fold CSV，checkpoint 存入 regression 子目录（共用）
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

    # train.py 实际路径: checkpoints/{TRACK}/{SUBTRACK_DIR}/{task}/{experiment_name}/
    # experiment_name="fold_N_exp" → binary/fold_N_exp（与原版 binary/fold_N 隔离）
    BINARY_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/binary/fold_${FOLD_IDX}_exp"
    TERNARY_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/ternary/fold_${FOLD_IDX}_exp"
    REG_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/regression/fold_${FOLD_IDX}"

    # ── Fold N.A: Binary ────────────────────────────────────────────────────
    echo ""
    echo "  >> [Fold ${FOLD_IDX}.A] 训练 Binary（二分类，使用扩展 fold CSV，train=101 val=9）"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
        python train.py \
            --task            binary                  \
            --fold_csv        "${BINARY_FOLD_CSV}"    \
            --experiment_name "fold_${FOLD_IDX}_exp"  \
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
    echo "  >> [Fold ${FOLD_IDX}.B] 训练 Ternary（三分类，使用扩展 fold CSV，train=101 val=9）"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
        python train.py \
            --task            ternary                 \
            --fold_csv        "${TERNARY_FOLD_CSV}"   \
            --experiment_name "fold_${FOLD_IDX}_exp"  \
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
    echo "  >> [Fold ${FOLD_IDX}.C] 训练 Regression（PHQ-9 回归，使用原版 fold CSV）"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
        python train.py \
            --task              regression             \
            --regression_label  "${REGRESSION_LABEL}" \
            --fold_csv          "${REG_FOLD_CSV}"      \
            --experiment_name   "fold_${FOLD_IDX}"     \
            --mse_warmup_epochs "${MSE_WARMUP_EPOCHS}" \
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
echo "  Track       : ${TRACK}  Subtrack: ${SUBTRACK}"
echo "  数据集扩展  ：原始训练 88 + 测试集伪标签 22 = 110 样本参与训练"
echo "  集成方式    ：${N_FOLDS} 折各出 1 个最优 checkpoint，共 3×${N_FOLDS} 个权重参与平均"
echo ""
echo "  fold CSV 目录（固定，下次运行自动复用）："
echo "    binary    (扩展): ${BINARY_KFOLD_DIR}"
echo "    ternary   (扩展): ${TERNARY_KFOLD_DIR}"
echo "    regression(原版): ${REG_KFOLD_DIR}"
echo ""
echo "  checkpoint 目录："
echo "    binary   (exp): ${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/binary/fold_N_exp/"
echo "    ternary  (exp): ${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/ternary/fold_N_exp/"
echo "    regression    : ${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/regression/fold_N/"
echo ""
echo "  输出目录：${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "  上传此文件到 CodaBench："
echo "  → ${OUTPUT_DIR}/submission.zip"
echo ""
