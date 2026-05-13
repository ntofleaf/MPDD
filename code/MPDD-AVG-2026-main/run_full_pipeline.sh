#!/usr/bin/env bash
# =============================================================================
#  run_full_pipeline.sh
#  MPDD-AVG-2026 全流程脚本：训练 → 集成推理 → 提交打包
#
#  流程概述
#  ────────
#  对 binary、ternary、regression 三个任务，分别以 NUM_SEEDS 个不同的随机种子
#  各训练一次，共产生 NUM_SEEDS × 3 个 best-checkpoint。
#
#  推理阶段：
#    ① binary_pred  = 对 NUM_SEEDS 个 binary  checkpoint 做 softmax 概率平均 → argmax
#    ② ternary_pred = 对 NUM_SEEDS 个 ternary checkpoint 做 softmax 概率平均 → argmax
#    ③ phq9_pred    = 对 NUM_SEEDS 个 regression checkpoint 做 PHQ-9 预测值平均
#
#  输出：
#    make_submission_forcodabench/my_submissions/<时间戳>/
#        binary.csv       ← binary_pred (来自①) + phq9_pred (来自③)
#        ternary.csv      ← ternary_pred(来自②) + phq9_pred (来自③)
#        submission.zip   ← 以上两个 CSV 打包，可直接上传 CodaBench
#
#  用法：
#    bash run_full_pipeline.sh
#
#  注意事项：
#    - 分类任务（binary/ternary）不带回归头（纯分类，避免梯度竞争）
#    - 回归任务（regression）带回归头，checkpoint 按 CCC 选最优
#    - patience 默认设为 epochs（不早停），确保每次都跑满 EPOCHS 轮
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1. 显卡设置
# ─────────────────────────────────────────────────────────────────────────────
GPU_IDS="0"

# NCCL 多卡通信模式控制：
#   NCCL_P2P_DISABLE=1  → 禁用 GPU 间点对点直接通信（P2P），改走 CPU 中转
#                         适用于 PCIe 拓扑不理想、NCCL 报 "unhandled cuda error" 的情况
#   如果训练正常（不报 NCCL 错误），可以改为 0 以获得更高的多卡通信速度
NCCL_P2P_DISABLE=1

# ─────────────────────────────────────────────────────────────────────────────
# 2. 任务与数据定义
#    TRACK   : Track1（老年）| Track2（青年）
#    SUBTRACK: A-V+P | A-V-G+P | G+P
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
EPOCHS=400
BATCH_SIZE=32          # 2张卡时每卡处理 16 个样本，8张卡时每卡 4 个；
                       # 与 8 卡比总 batch 不变，LR 无需调整（保持 5e-4）
LR=5e-4
PATIENCE=400       # 等于 EPOCHS 表示不早停（跑满 EPOCHS 轮）
WEIGHT_DECAY=1e-3
VAL_RATIO=0.2

# ─────────────────────────────────────────────────────────────────────────────
# 7. 集成种子列表
#    训练 3 个种子 → 得到 3 个 checkpoint → 集成时取平均
#    增加/减少种子数量只需修改此列表
# ─────────────────────────────────────────────────────────────────────────────
SEEDS=(3407 42 2024)

# ─────────────────────────────────────────────────────────────────────────────
# 8. 回归任务配置
#    regression_label: 辅助分类头使用哪种标签（binary 还是 ternary 粒度）
#    选 label2（二分类粒度）即可，PHQ-9 预测质量不依赖于辅助分类类型
# ─────────────────────────────────────────────────────────────────────────────
REGRESSION_LABEL="label2"

# [Fix #1] MSE 预热 epoch 数
# 前 MSE_WARMUP_EPOCHS 轮用 MSELoss 让模型先学到大致方向，
# 之后切换到 CCCLoss+方差惩罚，防止小样本下直接用 CCC 导致的均值坍缩。
# 推荐值：30~50（数据集越小越需要更多预热）
MSE_WARMUP_EPOCHS=40

# ─────────────────────────────────────────────────────────────────────────────
# 9. 多卡 batch 自动缩放（同 run_train.sh）
# ─────────────────────────────────────────────────────────────────────────────
AUTO_SCALE_BATCH=false

# =============================================================================
#  以下内容一般不需要修改
# =============================================================================

# 任何命令失败立刻退出，防止在错误状态下继续运行
set -euo pipefail

# 切换到脚本所在的目录（即项目根目录），确保相对路径正确
cd "$(dirname "$0")" || exit 1

# ── 时间戳用于输出目录命名，确保每次运行不覆盖前一次的结果 ────────────────
TIMESTAMP=$(date +"%Y-%m-%d_%H.%M.%S")
OUTPUT_DIR="make_submission_forcodabench/my_submissions/pipeline_${TIMESTAMP}"

# ── 多卡 batch 缩放 ──────────────────────────────────────────────────────────
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | grep -c '[0-9]')
ACTUAL_BATCH=${BATCH_SIZE}
ACTUAL_LR=${LR}
if [ "${AUTO_SCALE_BATCH}" = "true" ] && [ "${NUM_GPUS}" -gt 1 ]; then
    ACTUAL_BATCH=$(( BATCH_SIZE * NUM_GPUS ))
    ACTUAL_LR=$(awk "BEGIN {printf \"%.2e\", ${LR} * ${NUM_GPUS}}")
fi

# ── subtrack 目录名映射（与 train.py 里的 SUBTRACK_LOG_DIRS 保持一致）────────
# train.py 把 A-V+P 映射为 A-V-P 作为目录名（加号变短横线）
case "${SUBTRACK}" in
    "A-V+P")   SUBTRACK_DIR="A-V-P"   ;;
    "A-V-G+P") SUBTRACK_DIR="A-V-G+P" ;;
    "G+P")     SUBTRACK_DIR="G-P"     ;;
    *)         SUBTRACK_DIR="${SUBTRACK}" ;;
esac

# ── 实验名称 & checkpoint 目录推导（与 train.py 的 build_experiment_name 一致）
TRACK_LOWER="${TRACK,,}"    # 转小写：Track2 → track2
FEATURE_TAG="${AUDIO_FEATURE}__${VIDEO_FEATURE}"

BINARY_EXP="${TRACK_LOWER}_binary_${SUBTRACK}_${ENCODER_TYPE}_${FEATURE_TAG}"
TERNARY_EXP="${TRACK_LOWER}_ternary_${SUBTRACK}_${ENCODER_TYPE}_${FEATURE_TAG}"
REG_EXP="${TRACK_LOWER}_regression_${REGRESSION_LABEL}_${SUBTRACK}_${ENCODER_TYPE}_${FEATURE_TAG}"

CKPT_ROOT="checkpoints"
BINARY_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/binary/${BINARY_EXP}"
TERNARY_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/ternary/${TERNARY_EXP}"
REG_CKPT_DIR="${CKPT_ROOT}/${TRACK}/${SUBTRACK_DIR}/regression/${REG_EXP}"

# ── 共同训练参数（不含 --task 和 --seed，后面循环中动态填入）──────────────
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
    --val_ratio     "${VAL_RATIO}"
    --device        cuda
)

# ─────────────────────────────────────────────────────────────────────────────
#  打印配置概览
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  GPU         : ${GPU_IDS}  (${NUM_GPUS} card(s))"
echo "  Track       : ${TRACK}  Subtrack: ${SUBTRACK}"
echo "  Encoder     : ${ENCODER_TYPE}  hidden_dim=${HIDDEN_DIM}  dropout=${DROPOUT}"
echo "  Feature     : audio=${AUDIO_FEATURE}  video=${VIDEO_FEATURE}"
echo "  LR=${ACTUAL_LR}  batch=${ACTUAL_BATCH}  epochs=${EPOCHS}  patience=${PATIENCE}"
echo "  weight_decay=${WEIGHT_DECAY}  val_ratio=${VAL_RATIO}"
echo "  Seeds       : ${SEEDS[*]}  (共 ${#SEEDS[@]} 次训练 × 3 任务 = $((${#SEEDS[@]}*3)) 次)"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
#  初始化 checkpoint 收集数组
# ─────────────────────────────────────────────────────────────────────────────
BINARY_CKPTS=()    # 收集每个 seed 的 binary best checkpoint 路径
TERNARY_CKPTS=()   # 收集每个 seed 的 ternary best checkpoint 路径
REG_CKPTS=()       # 收集每个 seed 的 regression best checkpoint 路径

# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: 按 seed 顺序训练三种任务
# ─────────────────────────────────────────────────────────────────────────────
SEED_IDX=0
for SEED in "${SEEDS[@]}"; do
    SEED_IDX=$(( SEED_IDX + 1 ))
    echo ""
    echo "################################################################"
    echo "##  Seed ${SEED_IDX}/${#SEEDS[@]}  (seed=${SEED})               ##"
    echo "################################################################"

    # ── 1a. 训练 Binary 分类（不带回归头，纯分类优化）────────────────────────
    echo ""
    echo "  >> [${SEED_IDX}.A] 训练 Binary（二分类）seed=${SEED}"
    # train.py 检测到 task=binary 时，默认不启用回归头（避免梯度竞争）
    # 若需要保留旧行为（带回归头），可加 --force_regression_head
    CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} python train.py \
        --task   binary \
        --seed   "${SEED}" \
        "${COMMON_ARGS[@]}"

    # 找到这次训练产生的 best checkpoint（ls -t 按修改时间排序，head -1 取最新）
    # 注意：每次训练完成后 best checkpoint 是目录里最新的 .pth 文件
    BINARY_CKPT=$(ls -t "${BINARY_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
    if [ -z "${BINARY_CKPT}" ]; then
        echo "  [ERROR] 未找到 binary checkpoint！请检查训练是否正常结束。"
        echo "  查找路径: ${BINARY_CKPT_DIR}"
        exit 1
    fi
    BINARY_CKPTS+=("${BINARY_CKPT}")
    echo "  ✅ Binary checkpoint: ${BINARY_CKPT}"

    # ── 1b. 训练 Ternary 分类（不带回归头，纯分类优化）──────────────────────
    echo ""
    echo "  >> [${SEED_IDX}.B] 训练 Ternary（三分类）seed=${SEED}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} python train.py \
        --task   ternary \
        --seed   "${SEED}" \
        "${COMMON_ARGS[@]}"

    TERNARY_CKPT=$(ls -t "${TERNARY_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
    if [ -z "${TERNARY_CKPT}" ]; then
        echo "  [ERROR] 未找到 ternary checkpoint！"
        exit 1
    fi
    TERNARY_CKPTS+=("${TERNARY_CKPT}")
    echo "  ✅ Ternary checkpoint: ${TERNARY_CKPT}"

    # ── 1c. 训练 Regression（PHQ-9 回归，带辅助分类头，按 CCC 选最优）────────
    echo ""
    echo "  >> [${SEED_IDX}.C] 训练 Regression（PHQ-9 回归）seed=${SEED}"
    # task=regression 时 train.py 会：
    #   ① 开启回归头（use_regression_head=True）
    #   ② 按 CCC（一致性相关系数）保存最优 checkpoint，而不是按 F1
    #   ③ regression_label 指定辅助分类头的类型（label2=二分类粒度）
    CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} python train.py \
        --task              regression      \
        --regression_label  "${REGRESSION_LABEL}" \
        --seed              "${SEED}"       \
        --mse_warmup_epochs "${MSE_WARMUP_EPOCHS}" \
        "${COMMON_ARGS[@]}"

    REG_CKPT=$(ls -t "${REG_CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -1)
    if [ -z "${REG_CKPT}" ]; then
        echo "  [ERROR] 未找到 regression checkpoint！"
        exit 1
    fi
    REG_CKPTS+=("${REG_CKPT}")
    echo "  ✅ Regression checkpoint: ${REG_CKPT}"

done

# ─────────────────────────────────────────────────────────────────────────────
#  打印收集到的所有 checkpoint 路径（方便 Debug）
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  [总结] 共收集到 $((${#SEEDS[@]}*3)) 个 checkpoint："
echo ""
echo "  Binary checkpoints（${#BINARY_CKPTS[@]} 个）："
for p in "${BINARY_CKPTS[@]}"; do echo "    $p"; done
echo ""
echo "  Ternary checkpoints（${#TERNARY_CKPTS[@]} 个）："
for p in "${TERNARY_CKPTS[@]}"; do echo "    $p"; done
echo ""
echo "  Regression checkpoints（${#REG_CKPTS[@]} 个）："
for p in "${REG_CKPTS[@]}"; do echo "    $p"; done
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: 集成推理 + 打包
#  把三组 checkpoint 列表传给 Python 脚本
#  Python 脚本会：
#    ① 对 binary_ckpts  做 softmax 概率平均 → binary_pred
#    ② 对 ternary_ckpts 做 softmax 概率平均 → ternary_pred
#    ③ 对 reg_ckpts     做 PHQ-9 预测值平均 → phq9_pred
#    ④ 拼接 binary.csv 和 ternary.csv，打包成 submission.zip
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "################################################################"
echo "##  Step 2: 集成推理 + 提交打包                             ##"
echo "################################################################"

CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} python make_submission_forcodabench/ensemble_infer_and_pack.py \
    --binary_ckpts   "${BINARY_CKPTS[@]}"  \
    --ternary_ckpts  "${TERNARY_CKPTS[@]}" \
    --reg_ckpts      "${REG_CKPTS[@]}"     \
    --test_root      "${TEST_DATA_ROOT}"   \
    --personality    "${PERSONALITY_NPY}"  \
    --output_dir     "${OUTPUT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: 汇总报告
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "################################################################"
echo "##  全流程完成！                                             ##"
echo "################################################################"
echo ""
echo "  输出目录：${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "  上传此文件到 CodaBench："
echo "  → ${OUTPUT_DIR}/submission.zip"
echo ""
