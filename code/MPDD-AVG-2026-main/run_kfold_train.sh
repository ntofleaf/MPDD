#!/usr/bin/env bash
# =============================================================================
#  MPDD-AVG-2026 五折交叉验证训练脚本
#
#  用法（两步走）：
#  ─────────────────────────────────────────────────────────────────────────────
#  Step 0（只做一次，生成固定 fold CSV）：
#    python generate_kfold_splits.py \
#        --split_csv <SPLIT_CSV> \
#        --task      <TASK> \
#        --out_dir   <KFOLD_DIR>
#
#  Step 1（跑 5 折训练）：
#    bash run_kfold_train.sh
# =============================================================================

# ---------------------------------------------------------------------------
# 1. 显卡设置
# ---------------------------------------------------------------------------
GPU_IDS="0,1,2,3,4,5,6,7"

# ---------------------------------------------------------------------------
# 2. 任务定义
# ---------------------------------------------------------------------------
TRACK="Track2"
TASK="binary"
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
# ---------------------------------------------------------------------------
DATA_ROOT="/home/niutao/data/datasets_MPDD/Train-MPDD-Young"
SPLIT_CSV="${DATA_ROOT}/split_labels_train.csv"
PERSONALITY_NPY="${DATA_ROOT}/descriptions_embeddings_with_ids.npy"

# ---------------------------------------------------------------------------
# 6. 训练超参数
# ---------------------------------------------------------------------------
SEED=42          # 只影响模型初始化 & DataLoader shuffle，不影响 fold 分割
EPOCHS=400
BATCH_SIZE=32
LR=5e-4
PATIENCE=400
WEIGHT_DECAY=1e-3

# ---------------------------------------------------------------------------
# 7. 五折设置
#    KFOLD_DIR  : generate_kfold_splits.py 的 --out_dir，需与生成时一致
#    N_FOLDS    : fold 数，需与生成时一致
# ---------------------------------------------------------------------------
N_FOLDS=5
KFOLD_DIR="kfold_splits"

# =============================================================================
#  以下内容一般不需要修改
# =============================================================================

cd "$(dirname "$0")" || exit 1

# ── 统计卡数 ─────────────────────────────────────────────────────────────────
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | grep -c '[0-9]')

echo "========================================"
echo "  五折交叉验证训练"
echo "  GPU        : ${GPU_IDS}  (${NUM_GPUS} 张)"
echo "  Track      : ${TRACK}  Task: ${TASK}  Subtrack: ${SUBTRACK}"
echo "  Encoder    : ${ENCODER_TYPE}  hidden_dim=${HIDDEN_DIM}  dropout=${DROPOUT}"
echo "  Audio feat : ${AUDIO_FEATURE}  Video feat: ${VIDEO_FEATURE}"
echo "  LR=${LR}  batch=${BATCH_SIZE}  epochs=${EPOCHS}  patience=${PATIENCE}"
echo "  seed=${SEED}  weight_decay=${WEIGHT_DECAY}"
echo "  N_FOLDS=${N_FOLDS}  KFOLD_DIR=${KFOLD_DIR}"
echo "========================================"

# ── 检查 fold CSV 是否已生成 ─────────────────────────────────────────────────
for FOLD_IDX in $(seq 0 $((N_FOLDS - 1))); do
    FOLD_CSV="${KFOLD_DIR}/fold_${FOLD_IDX}.csv"
    if [ ! -f "${FOLD_CSV}" ]; then
        echo ""
        echo "[ERROR] fold CSV 不存在: ${FOLD_CSV}"
        echo "请先运行（只需一次）："
        echo ""
        echo "  python generate_kfold_splits.py \\"
        echo "      --split_csv ${SPLIT_CSV} \\"
        echo "      --task      ${TASK} \\"
        echo "      --out_dir   ${KFOLD_DIR}"
        echo ""
        exit 1
    fi
done

echo ""
echo "fold CSV 检查通过，开始循环训练..."
echo ""

# ── 逐 fold 训练 ──────────────────────────────────────────────────────────────
for FOLD_IDX in $(seq 0 $((N_FOLDS - 1))); do
    FOLD_CSV="${KFOLD_DIR}/fold_${FOLD_IDX}.csv"

    echo "──────────────────────────────────────────"
    echo "  Fold ${FOLD_IDX} / $((N_FOLDS - 1))  →  ${FOLD_CSV}"
    echo "──────────────────────────────────────────"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py \
        --track           "${TRACK}"            \
        --task            "${TASK}"             \
        --subtrack        "${SUBTRACK}"         \
        --encoder_type    "${ENCODER_TYPE}"     \
        --audio_feature   "${AUDIO_FEATURE}"    \
        --video_feature   "${VIDEO_FEATURE}"    \
        --data_root       "${DATA_ROOT}"        \
        --split_csv       "${SPLIT_CSV}"        \
        --fold_csv        "${FOLD_CSV}"         \
        --personality_npy "${PERSONALITY_NPY}"  \
        --seed            "${SEED}"             \
        --epochs          "${EPOCHS}"           \
        --batch_size      "${BATCH_SIZE}"       \
        --lr              "${LR}"               \
        --hidden_dim      "${HIDDEN_DIM}"       \
        --dropout         "${DROPOUT}"          \
        --patience        "${PATIENCE}"         \
        --weight_decay    "${WEIGHT_DECAY}"     \
        --experiment_name "fold_${FOLD_IDX}"   \
        --device          cuda

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo ""
        echo "[ERROR] Fold ${FOLD_IDX} 训练失败（exit code=${EXIT_CODE}），中止。"
        exit ${EXIT_CODE}
    fi

    echo "  Fold ${FOLD_IDX} 训练完成 ✓"
    echo ""
done

echo "========================================"
echo "  全部 ${N_FOLDS} 折训练完成！"
echo ""
echo "  checkpoint 位置（按 experiment_name 区分）："
echo "  checkpoints/${TRACK}/*/  中寻找 fold_0 ~ fold_$((N_FOLDS-1)) 子目录"
echo ""
echo "  下一步：将 5 个 best checkpoint 路径传入集成推理脚本："
echo "  python make_submission_forcodabench/ensemble_infer_and_pack.py \\"
echo "      --binary_ckpts  <fold_0_ckpt> <fold_1_ckpt> ... \\"
echo "      --ternary_ckpts <...> \\"
echo "      --reg_ckpts     <...> \\"
echo "      --test_root     <TEST_ROOT> \\"
echo "      --personality   ${PERSONALITY_NPY} \\"
echo "      --output_dir    make_submission_forcodabench/my_submissions/kfold_ensemble"
echo "========================================"
