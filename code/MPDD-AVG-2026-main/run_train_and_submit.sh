#!/usr/bin/env bash
# =============================================================================
#  MPDD-AVG-2026 训练 + 测试集推理 + 打包 一条龙脚本
#
#  流程：
#    1. 用相同超参数分别训练 binary 和 ternary 模型
#    2. 训练结束后自动找到各自的最优 checkpoint（train.py 保存的 best_model_*.pth）
#    3. 调用 infer_and_pack.py 对测试集做推理
#    4. 将 binary.csv + ternary.csv 打包成 submission.zip（CodaBench 可直接上传）
#
#  用法:
#    bash run_train_and_submit.sh
#
#  输出:
#    make_submission_forcodabench/my_submissions/<timestamp>/
#        binary.csv
#        ternary.csv
#        submission.zip   ← 上传此文件到 CodaBench
# =============================================================================

# ---------------------------------------------------------------------------
# 1. 显卡设置
# ---------------------------------------------------------------------------
GPU_IDS="0,1,2,3,4,5,6,7"

# ---------------------------------------------------------------------------
# 2. 任务定义（track 和 subtrack，task 不用填——脚本内部自动运行 binary + ternary）
#    track   : Track1（老年） | Track2（青年）
#    subtrack: A-V+P | A-V-G+P | G+P
# ---------------------------------------------------------------------------
TRACK="Track2"
SUBTRACK="A-V+P"

# ---------------------------------------------------------------------------
# 3. 模型结构
# ---------------------------------------------------------------------------
ENCODER_TYPE="bilstm_mean"
HIDDEN_DIM=64
DROPOUT=0.6

# ---------------------------------------------------------------------------
# 4. 特征选择
# ---------------------------------------------------------------------------
AUDIO_FEATURE="mfcc"
VIDEO_FEATURE="densenet"

# ---------------------------------------------------------------------------
# 5. 数据路径
#    TRAIN_DATA_ROOT : 训练集根目录（含 split_labels_train.csv）
#    TEST_DATA_ROOT  : 测试集根目录（含 split_labels_test.csv）
#    PERSONALITY_NPY : 人格嵌入文件
# ---------------------------------------------------------------------------
TRAIN_DATA_ROOT="/home/niutao/data/datasets_MPDD/Train-MPDD-Young"
TEST_DATA_ROOT="/home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young"
PERSONALITY_NPY="${TRAIN_DATA_ROOT}/descriptions_embeddings_with_ids.npy"
SPLIT_CSV="${TRAIN_DATA_ROOT}/split_labels_train.csv"

# ---------------------------------------------------------------------------
# 6. 训练超参数
# ---------------------------------------------------------------------------
SEED=42
EPOCHS=400
BATCH_SIZE=32
LR=5e-4
PATIENCE=400
WEIGHT_DECAY=1e-3
VAL_RATIO=0.2

# ---------------------------------------------------------------------------
# 7. 多卡自动扩 batch（同 run_train.sh）
# ---------------------------------------------------------------------------
AUTO_SCALE_BATCH=false

# =============================================================================
#  以下内容一般不需要修改
# =============================================================================

set -euo pipefail          # 任何命令失败立刻退出
cd "$(dirname "$0")" || exit 1    # 切换到项目根目录

# ── 时间戳（用于输出目录命名）────────────────────────────────────────────
TIMESTAMP=$(date +"%Y-%m-%d_%H.%M.%S")
OUTPUT_DIR="make_submission_forcodabench/my_submissions/${TIMESTAMP}"

# ── 多卡扩 batch ─────────────────────────────────────────────────────────
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | grep -c '[0-9]')
ACTUAL_BATCH=${BATCH_SIZE}
ACTUAL_LR=${LR}
if [ "${AUTO_SCALE_BATCH}" = "true" ] && [ "${NUM_GPUS}" -gt 1 ]; then
    ACTUAL_BATCH=$(( BATCH_SIZE * NUM_GPUS ))
    ACTUAL_LR=$(awk "BEGIN {printf \"%.2e\", ${LR} * ${NUM_GPUS}}")
fi

# ── subtrack 目录名映射（与 train.py 的 SUBTRACK_LOG_DIRS 保持一致）───
case "${SUBTRACK}" in
    "A-V+P")   SUBTRACK_DIR="A-V-P"   ;;
    "A-V-G+P") SUBTRACK_DIR="A-V-G+P" ;;
    "G+P")     SUBTRACK_DIR="G-P"     ;;
    *)         SUBTRACK_DIR="${SUBTRACK}" ;;
esac

# ── 特征标签（与 build_experiment_name 逻辑一致）─────────────────────
TRACK_LOWER="${TRACK,,}"
FEATURE_TAG="${AUDIO_FEATURE}__${VIDEO_FEATURE}"
BINARY_EXP="${TRACK_LOWER}_binary_${SUBTRACK}_${ENCODER_TYPE}_${FEATURE_TAG}"
TERNARY_EXP="${TRACK_LOWER}_ternary_${SUBTRACK}_${ENCODER_TYPE}_${FEATURE_TAG}"

CKPT_ROOT="checkpoints"
BINARY_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/binary/${BINARY_EXP}"
TERNARY_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/ternary/${TERNARY_EXP}"

# ── 公共 python 参数（不含 --task）──────────────────────────────────────
COMMON_ARGS=(
    --track         "${TRACK}"
    --subtrack      "${SUBTRACK}"
    --encoder_type  "${ENCODER_TYPE}"
    --audio_feature "${AUDIO_FEATURE}"
    --video_feature "${VIDEO_FEATURE}"
    --data_root     "${TRAIN_DATA_ROOT}"
    --split_csv     "${SPLIT_CSV}"
    --personality_npy "${PERSONALITY_NPY}"
    --seed          "${SEED}"
    --epochs        "${EPOCHS}"
    --batch_size    "${ACTUAL_BATCH}"
    --lr            "${ACTUAL_LR}"
    --hidden_dim    "${HIDDEN_DIM}"
    --dropout       "${DROPOUT}"
    --patience      "${PATIENCE}"
    --weight_decay  "${WEIGHT_DECAY}"
    --val_ratio     "${VAL_RATIO}"
    --device        cuda
)

# ── 打印配置概览 ─────────────────────────────────────────────────────────
echo "=================================================="
echo "  GPU         : ${GPU_IDS}  (${NUM_GPUS} card(s))"
echo "  Track       : ${TRACK}  Subtrack: ${SUBTRACK}"
echo "  Encoder     : ${ENCODER_TYPE}  hidden_dim=${HIDDEN_DIM}  dropout=${DROPOUT}"
echo "  Feature     : audio=${AUDIO_FEATURE}  video=${VIDEO_FEATURE}"
echo "  LR=${ACTUAL_LR}  batch=${ACTUAL_BATCH}  epochs=${EPOCHS}  patience=${PATIENCE}"
echo "  seed=${SEED}  weight_decay=${WEIGHT_DECAY}  val_ratio=${VAL_RATIO}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "=================================================="

# ============================================================================
#  Step 1: 训练 binary
# ============================================================================
echo ""
echo "##################################################"
echo "##  [Step 1/4] 训练 Binary（二分类）           ##"
echo "##################################################"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py \
    --task binary \
    "${COMMON_ARGS[@]}"

# 找最新的 binary best checkpoint
BINARY_CKPT=$(ls -t "${BINARY_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
if [ -z "${BINARY_CKPT}" ]; then
    echo "[ERROR] 未找到 binary checkpoint，训练可能失败！"
    echo "  查找路径: ${BINARY_CKPT_DIR}"
    exit 1
fi
echo ""
echo "✅ Binary 最优 checkpoint: ${BINARY_CKPT}"

# ============================================================================
#  Step 2: 训练 ternary
# ============================================================================
echo ""
echo "##################################################"
echo "##  [Step 2/4] 训练 Ternary（三分类）          ##"
echo "##################################################"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py \
    --task ternary \
    "${COMMON_ARGS[@]}"

# 找最新的 ternary best checkpoint
TERNARY_CKPT=$(ls -t "${TERNARY_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
if [ -z "${TERNARY_CKPT}" ]; then
    echo "[ERROR] 未找到 ternary checkpoint，训练可能失败！"
    echo "  查找路径: ${TERNARY_CKPT_DIR}"
    exit 1
fi
echo ""
echo "✅ Ternary 最优 checkpoint: ${TERNARY_CKPT}"

# ============================================================================
#  Step 3: 测试集推理
# ============================================================================
echo ""
echo "##################################################"
echo "##  [Step 3/4] 测试集推理（binary + ternary）  ##"
echo "##################################################"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python make_submission_forcodabench/infer_and_pack.py \
    --binary_ckpt  "${BINARY_CKPT}"    \
    --ternary_ckpt "${TERNARY_CKPT}"   \
    --test_root    "${TEST_DATA_ROOT}" \
    --personality  "${PERSONALITY_NPY}"\
    --output_dir   "${OUTPUT_DIR}"

# ============================================================================
#  Step 4: 汇报
# ============================================================================
echo ""
echo "##################################################"
echo "##  [Step 4/4] 完成                            ##"
echo "##################################################"
echo ""
echo "  Binary  checkpoint : ${BINARY_CKPT}"
echo "  Ternary checkpoint : ${TERNARY_CKPT}"
echo ""
echo "  提交文件目录 : ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "  上传到 CodaBench："
echo "  → ${OUTPUT_DIR}/submission.zip"
echo ""
