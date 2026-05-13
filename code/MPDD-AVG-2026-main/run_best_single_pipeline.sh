#!/usr/bin/env bash
# =============================================================================
#  run_best_single_pipeline.sh
#  MPDD-AVG-2026 全流程脚本（五折交叉验证 + 单最优权重推理）
#
#  流程
#  ─────────────────────────────────────────────────────────────────────────────
#  Step 0: 幂等生成 fold CSV（若已存在则跳过，只需生成一次）
#            binary / ternary / regression 各自独立分层，存于对应子目录
#
#  Step 1: 五折训练（N_FOLDS × 3 任务）
#            每折通过 --fold_csv 读取固定分割，分割完全确定
#
#  Step 2: 从每个任务的 N_FOLDS 个 checkpoint 中挑 val 指标最优的那一个
#            binary  / ternary → val Macro-F1 最高
#            regression        → val phq9_ccc（原始 PHQ-9 空间 CCC）最高
#
#  Step 3: 单一最优权重推理 + 打包提交
#
#  用法：
#    bash run_best_single_pipeline.sh
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1. 显卡设置
# ─────────────────────────────────────────────────────────────────────────────
GPU_IDS="0"
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
EPOCHS=400
BATCH_SIZE=32
LR=5e-4
PATIENCE=400
WEIGHT_DECAY=1e-3
# 注：五折模式下不再使用 VAL_RATIO，由 fold CSV 固定分割

# ─────────────────────────────────────────────────────────────────────────────
# 7. 五折交叉验证设置（替代多 seed 随机训练）
#
#    N_FOLDS       : fold 数，需与 generate_kfold_splits.py 生成时一致
#    KFOLD_BASE_DIR: fold CSV 根目录
#    SEED          : 只影响模型初始化和 DataLoader shuffle，不影响 fold 分割
# ─────────────────────────────────────────────────────────────────────────────
N_FOLDS=5
SEED=42
KFOLD_BASE_DIR="kfold_splits"
# 三任务各自独立分层，fold CSV 存于不同子目录
BINARY_KFOLD_DIR="${KFOLD_BASE_DIR}/Track2_Young_binary"
TERNARY_KFOLD_DIR="${KFOLD_BASE_DIR}/Track2_Young_ternary"
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
OUTPUT_DIR="make_submission_forcodabench/my_submissions/best_single_${TIMESTAMP}"

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

# COMMON_ARGS：五折模式下不包含 --val_ratio
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
echo "  [五折 + 单最优权重模式]"
echo "  GPU         : ${GPU_IDS}  (${NUM_GPUS} card(s))"
echo "  Track       : ${TRACK}  Subtrack: ${SUBTRACK}"
echo "  Encoder     : ${ENCODER_TYPE}  hidden_dim=${HIDDEN_DIM}  dropout=${DROPOUT}"
echo "  Feature     : audio=${AUDIO_FEATURE}  video=${VIDEO_FEATURE}"
echo "  LR=${ACTUAL_LR}  batch=${ACTUAL_BATCH}  epochs=${EPOCHS}  patience=${PATIENCE}"
echo "  seed=${SEED}  (影响模型初始化 & DataLoader shuffle，不影响 fold 分割)"
echo "  N_FOLDS=${N_FOLDS}  (共 ${N_FOLDS} × 3 任务 = $(( N_FOLDS * 3 )) 次训练)"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 0: 幂等生成 fold CSV（若已存在则跳过）
#
#  每个任务分层标准不同：
#    binary   → 按 label2 (0/1) 分层
#    ternary  → 按 label3 (0/1/2) 分层
#    regression → 按 PHQ-9 等频分箱分层
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
#
#  checkpoint 路径规律（由 train.py 的 --experiment_name 决定子目录）：
#    checkpoints/{TRACK}/{SUBTRACK_DIR}/binary/fold_k/best_model_xxx.pth
#    checkpoints/{TRACK}/{SUBTRACK_DIR}/ternary/fold_k/best_model_xxx.pth
#    checkpoints/{TRACK}/{SUBTRACK_DIR}/regression/fold_k/best_model_xxx.pth
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
    echo "  ✅ Binary  checkpoint: ${BINARY_CKPT}"

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
    echo "  ✅ Ternary checkpoint: ${TERNARY_CKPT}"

    # ── Fold N.C: Regression ────────────────────────────────────────────────
    echo ""
    echo "  >> [Fold ${FOLD_IDX}.C] 训练 Regression（PHQ-9 回归）"
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
    echo "  ✅ Regression checkpoint: ${REG_CKPT}"

    echo ""
    echo "  Fold ${FOLD_IDX} 全部任务训练完成 ✓"
done

# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: 从 N_FOLDS 个候选 checkpoint 中，按 val 指标挑出每任务最优的那一个
#
#  挑选标准：
#    binary  / ternary → val Macro-F1 最高
#    regression        → val phq9_ccc（原始 PHQ-9 空间的 CCC）最高
#                        若 phq9_ccc 未记录（旧 checkpoint），退用 val CCC（log1p 空间）
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "################################################################"
echo "##  Step 2: 从 ${N_FOLDS} 个候选 checkpoint 中挑最优单一权重"
echo "################################################################"

# Python 辅助脚本：读 checkpoint 的 best_val_metrics，按指定 key 排序返回最优路径
pick_best_ckpt() {
    local TASK="$1"       # binary / ternary / regression
    local METRIC="$2"     # f1 / phq9_ccc
    local FALLBACK="$3"   # 若 METRIC 不存在时退用的 key（如 ccc）
    shift 3
    local PATHS=("$@")    # checkpoint 路径列表

    python3 - <<PYEOF
import torch, sys

task      = "${TASK}"
metric    = "${METRIC}"
fallback  = "${FALLBACK}"
paths     = [p for p in """${PATHS[*]}""".split() if p.endswith('.pth')]

best_score = -999.0
best_path  = paths[0] if paths else ""

for p in paths:
    try:
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        m    = ckpt.get('best_val_metrics', {})
        score = float(m.get(metric, m.get(fallback, -999.0)))
        print(f"  [{task}] {p.split('/')[-2]}/{p.split('/')[-1]}  {metric}={score:.4f}", file=sys.stderr)
        if score > best_score:
            best_score = score
            best_path  = p
    except Exception as e:
        print(f"  [WARN] 无法读取 {p}: {e}", file=sys.stderr)

print(f"  => 选中: {best_path}  ({metric}={best_score:.4f})", file=sys.stderr)
print(best_path)   # stdout 只输出路径，供 shell 捕获
PYEOF
}

echo ""
echo "  ── Binary 候选（${#BINARY_CKPTS[@]} 个 fold），选 val F1 最高 ──"
BEST_BINARY=$(pick_best_ckpt "binary" "f1" "ccc" "${BINARY_CKPTS[@]}")

echo ""
echo "  ── Ternary 候选（${#TERNARY_CKPTS[@]} 个 fold），选 val F1 最高 ──"
BEST_TERNARY=$(pick_best_ckpt "ternary" "f1" "ccc" "${TERNARY_CKPTS[@]}")

echo ""
echo "  ── Regression 候选（${#REG_CKPTS[@]} 个 fold），选 val phq9_ccc 最高 ──"
BEST_REG=$(pick_best_ckpt "regression" "phq9_ccc" "ccc" "${REG_CKPTS[@]}")

echo ""
echo "================================================================"
echo "  最终使用的单一权重："
echo "    Binary    : ${BEST_BINARY}"
echo "    Ternary   : ${BEST_TERNARY}"
echo "    Regression: ${BEST_REG}"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: 推理 + 打包（只传 1 个 checkpoint，不做集成平均）
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "################################################################"
echo "##  Step 3: 推理 + 提交打包（单权重）"
echo "################################################################"

CUDA_VISIBLE_DEVICES=${GPU_IDS} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
    python make_submission_forcodabench/ensemble_infer_and_pack.py \
        --binary_ckpts   "${BEST_BINARY}"    \
        --ternary_ckpts  "${BEST_TERNARY}"   \
        --reg_ckpts      "${BEST_REG}"       \
        --test_root      "${TEST_DATA_ROOT}" \
        --personality    "${PERSONALITY_NPY}" \
        --output_dir     "${OUTPUT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 4: 汇总
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "################################################################"
echo "##  全流程完成！"
echo "################################################################"
echo ""
echo "  fold CSV 目录（固定，下次运行自动复用）："
echo "    binary   : ${BINARY_KFOLD_DIR}"
echo "    ternary  : ${TERNARY_KFOLD_DIR}"
echo "    regression: ${REG_KFOLD_DIR}"
echo ""
echo "  输出目录：${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "  上传此文件到 CodaBench："
echo "  → ${OUTPUT_DIR}/submission.zip"
echo ""
