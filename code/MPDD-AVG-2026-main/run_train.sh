#!/usr/bin/env bash
# =============================================================================
#  MPDD-AVG-2026 训练启动脚本
#  用法: bash run_train.sh
# =============================================================================

# ---------------------------------------------------------------------------
# 1. 显卡设置
#    多张卡时填卡号, 例如: "0"  "1"  "2"  "0,1"（多卡DataParallel）
# ---------------------------------------------------------------------------
GPU_IDS="0,1,2,3,4,5,6,7"

# ---------------------------------------------------------------------------
# 2. 任务定义
#    track       : Track1（老年）| Track2（青年）
#    task        : binary（二分类）| ternary（三分类）| regression（回归）
#    subtrack    : A-V+P | A-V-G+P | G+P
# ---------------------------------------------------------------------------
TRACK="Track2"
TASK="binary"
SUBTRACK="A-V+P"

# ---------------------------------------------------------------------------
# 3. 模型结构
#    encoder_type: bilstm_mean | hybrid_attn
#    hidden_dim  : BiLSTM 隐藏层维度，越大表达能力越强
#    dropout     : Dropout 概率，0~1 之间，越大正则越强
# ---------------------------------------------------------------------------
ENCODER_TYPE="bilstm_mean"
HIDDEN_DIM=64
DROPOUT=0.6           # 从 0.5 提高到 0.6，增强正则

# ---------------------------------------------------------------------------
# 4. 特征选择
#    audio_feature: wav2vec | ...（与数据目录下子文件夹名一致）
#    video_feature: densenet | ...（与数据目录下子文件夹名一致）
# ---------------------------------------------------------------------------
AUDIO_FEATURE="mfcc"
VIDEO_FEATURE="densenet"

# ---------------------------------------------------------------------------
# 5. 数据路径
# ---------------------------------------------------------------------------
DATA_ROOT="/home/niutao/data/datasets_MPDD/Train-MPDD-Young"
SPLIT_CSV="${DATA_ROOT}/split_labels_train.csv"
PERSONALITY_NPY="${DATA_ROOT}/descriptions_embeddings_with_ids.npy"

# ---------------------------------------------------------------------------
# 6. 训练超参数
#    seed      : 随机种子，固定可复现
#    epochs    : 最大训练轮数
#    batch_size: 每批样本数
#    lr        : 初始学习率（AdamW + CosineAnnealingLR）
#    patience  : Early Stopping 耐心轮数（验证集无提升时触发）
# ---------------------------------------------------------------------------
SEED=42
EPOCHS=400
BATCH_SIZE=32         # 从 16 提高到 32，稳定梯度
LR=5e-4               # 从 1e-3 降低，避免过快记忆训练集
PATIENCE=400           # 从 100 降到 20，开启有效 Early Stopping
WEIGHT_DECAY=1e-3     # 从 1e-4 提高到 1e-3，增强 L2 正则
VAL_RATIO=0.2         # 验证集比例：0.1→9条, 0.2→16条, 0.3→23条

# =============================================================================
#  以下内容一般不需要修改
# =============================================================================

# 切换到脚本所在目录（即项目根目录），保证相对路径正确
cd "$(dirname "$0")" || exit 1

# ---------------------------------------------------------------------------
# 多卡自动推导：统计 GPU_IDS 中的卡数，按 Linear Scaling Rule 缩放参数
#
#   AUTO_SCALE_BATCH=true  → 多卡时 batch_size 自动乘以卡数，LR 同步缩放
#   AUTO_SCALE_BATCH=false → batch_size / LR 保持原值不变（DataParallel 自动拆分）
#
# 建议：单机多卡且显存充足时开启，可以更充分利用多卡吞吐量
# ---------------------------------------------------------------------------
AUTO_SCALE_BATCH=false

# 统计逗号分隔的 GPU_IDS 中有几张卡
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | grep -c '[0-9]')

ACTUAL_BATCH=${BATCH_SIZE}
ACTUAL_LR=${LR}

if [ "${AUTO_SCALE_BATCH}" = "true" ] && [ "${NUM_GPUS}" -gt 1 ]; then
    ACTUAL_BATCH=$(( BATCH_SIZE * NUM_GPUS ))
    # 用 awk 做浮点乘法：LR × NUM_GPUS
    ACTUAL_LR=$(awk "BEGIN {printf \"%.2e\", ${LR} * ${NUM_GPUS}}")
fi

echo "========================================"
echo "  GPU        : ${GPU_IDS}  (${NUM_GPUS} card(s))"
echo "  Track      : ${TRACK}  Task: ${TASK}  Subtrack: ${SUBTRACK}"
echo "  Encoder    : ${ENCODER_TYPE}  hidden_dim=${HIDDEN_DIM}  dropout=${DROPOUT}"
echo "  Audio feat : ${AUDIO_FEATURE}  Video feat: ${VIDEO_FEATURE}"
if [ "${NUM_GPUS}" -gt 1 ] && [ "${AUTO_SCALE_BATCH}" = "true" ]; then
    echo "  LR         : ${LR} → ${ACTUAL_LR}  (×${NUM_GPUS} auto-scaled)"
    echo "  batch_size : ${BATCH_SIZE} → ${ACTUAL_BATCH}  (×${NUM_GPUS} auto-scaled)"
else
    echo "  LR=${ACTUAL_LR}  batch=${ACTUAL_BATCH}"
fi
echo "  epochs=${EPOCHS}  patience=${PATIENCE}  seed=${SEED}  weight_decay=${WEIGHT_DECAY}"
echo "========================================"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py \
    --track          "${TRACK}"            \
    --task           "${TASK}"             \
    --subtrack       "${SUBTRACK}"         \
    --encoder_type   "${ENCODER_TYPE}"     \
    --audio_feature  "${AUDIO_FEATURE}"    \
    --video_feature  "${VIDEO_FEATURE}"    \
    --data_root      "${DATA_ROOT}"        \
    --split_csv      "${SPLIT_CSV}"        \
    --personality_npy "${PERSONALITY_NPY}" \
    --seed           "${SEED}"             \
    --epochs         "${EPOCHS}"           \
    --batch_size     "${ACTUAL_BATCH}"     \
    --lr             "${ACTUAL_LR}"        \
    --hidden_dim     "${HIDDEN_DIM}"       \
    --dropout        "${DROPOUT}"          \
    --patience        "${PATIENCE}"         \
    --weight_decay   "${WEIGHT_DECAY}"     \
    --val_ratio      "${VAL_RATIO}"        \
    --device         cuda
