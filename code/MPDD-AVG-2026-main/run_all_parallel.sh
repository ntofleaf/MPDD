#!/usr/bin/env bash
# ============================================================
# run_all_parallel.sh
# 多实验并行训练脚本：每个实验独占 1 张 GPU
# 用法：
#   bash run_all_parallel.sh              # 默认用 GPU 0-7（8 张）
#   NUM_GPUS=4 bash run_all_parallel.sh   # 只用 GPU 0-3
#   GPU_LIST="2,3,5,6" bash run_all_parallel.sh  # 指定具体 GPU 编号
# ============================================================
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

# ---- 配置：GPU 列表 ----
if [[ -n "${GPU_LIST:-}" ]]; then
    # 用户手动指定，例如 GPU_LIST="0,1,2,3"
    IFS=',' read -ra GPUS <<< "$GPU_LIST"
else
    NUM_GPUS="${NUM_GPUS:-8}"
    GPUS=()
    for ((i=0; i<NUM_GPUS; i++)); do GPUS+=("$i"); done
fi
echo "[INFO] Using GPUs: ${GPUS[*]} (total ${#GPUS[@]})"

# ---- 日志目录 ----
PARALLEL_LOG_DIR="$PROJECT_ROOT/parallel_logs"
mkdir -p "$PARALLEL_LOG_DIR"

# ---- 定义所有实验 ----
# 格式：每行一个实验配置，用 | 分隔字段
# 字段：TRACK | TASK | SUBTRACK | AUDIO | VIDEO | EPOCHS | BATCH | LR | HIDDEN | DROPOUT | PATIENCE | TARGET_T | SEED | EXTRA_ARGS
declare -a EXPERIMENTS=(
    # === Track1 / Elder ===
    # A-V+P ternary（干净脚本，完全可用）
    "Track1|ternary|A-V+P|mfcc|resnet|140|4|2e-4|160|0.5|30|128|42|"
    # A-V+P binary（忽略 init_checkpoint，其余参数沿用脚本默认）
    "Track1|binary|A-V+P|opensmile|resnet|60|8|3e-5|64|0.4|20|128|42|"
    # A-V-G+P binary
    "Track1|binary|A-V-G+P|mfcc|resnet|140|4|8e-5|128|0.45|35|128|42|"
    # A-V-G+P ternary
    "Track1|ternary|A-V-G+P|wav2vec|openface|160|4|8e-5|160|0.4|40|128|42|"
    # G+P binary（仅步态）
    "Track1|binary|G+P|||320|2|8e-5|128|0.3|90|128|42|"
    # === Track2 / Young ===
    # A-V+P binary（wav2vec + resnet 作为代表组合）
    "Track2|binary|A-V+P|wav2vec|resnet|80|8|5e-4|64|0.4|15|128|3407|"
    # A-V+P ternary
    "Track2|ternary|A-V+P|wav2vec|resnet|80|8|5e-4|64|0.4|15|128|3407|"
)

# ---- 数据路径 ----
ELDER_DATA="MPDD-AVG2026/MPDD-AVG2026-trainval/Elder"
YOUNG_DATA="MPDD-AVG2026/MPDD-AVG2026-trainval/Young"
ELDER_CSV="$ELDER_DATA/split_labels_train.csv"
YOUNG_CSV="$YOUNG_DATA/split_labels_train.csv"
ELDER_NPY="$ELDER_DATA/descriptions_embeddings_with_ids.npy"
YOUNG_NPY="$YOUNG_DATA/descriptions_embeddings_with_ids.npy"

# ---- 如果实验数 > GPU 数，循环使用 GPU ----
GPU_IDX=0
declare -a PIDS=()
declare -a LOGFILES=()
declare -a EXP_NAMES=()

echo ""
echo "=========================================="
echo " 启动 ${#EXPERIMENTS[@]} 个实验"
echo "=========================================="

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r TRACK TASK SUBTRACK AUDIO VIDEO EPOCHS BATCH LR HIDDEN DROPOUT PATIENCE TARGET_T SEED EXTRA <<< "$EXP"

    # 分配 GPU（循环使用）
    GPU="${GPUS[$GPU_IDX]}"
    GPU_IDX=$(( (GPU_IDX + 1) % ${#GPUS[@]} ))

    # 数据路径
    if [[ "$TRACK" == "Track1" ]]; then
        DATA_ROOT="$ELDER_DATA"
        SPLIT_CSV="$ELDER_CSV"
        PERSONALITY_NPY="$ELDER_NPY"
    else
        DATA_ROOT="$YOUNG_DATA"
        SPLIT_CSV="$YOUNG_CSV"
        PERSONALITY_NPY="$YOUNG_NPY"
    fi

    # 实验名
    if [[ "$SUBTRACK" == "G+P" ]]; then
        FEAT_TAG="gait_only"
    else
        FEAT_TAG="${AUDIO}__${VIDEO}"
    fi
    EXP_NAME="${TRACK,,}_${TASK}_${SUBTRACK/+/-}_bilstm_mean_${FEAT_TAG}_parallel"

    # 日志文件（并行日志）
    LOGFILE="$PARALLEL_LOG_DIR/${EXP_NAME}.log"

    echo "[GPU $GPU] $TRACK/$SUBTRACK/$TASK  audio=$AUDIO video=$VIDEO"
    echo "         输出日志: $LOGFILE"
    echo "         checkpoint: checkpoints/$TRACK/$(echo $SUBTRACK | tr '+' '-')/$TASK/$EXP_NAME/"
    echo "         result JSON: logs/$TRACK/$(echo $SUBTRACK | tr '+' '-')/$TASK/$EXP_NAME/train_result_*.json"
    echo ""

    # 构建命令
    CMD="python train.py \
        --track $TRACK \
        --task $TASK \
        --subtrack $SUBTRACK \
        --encoder_type bilstm_mean \
        --data_root $DATA_ROOT \
        --split_csv $SPLIT_CSV \
        --personality_npy $PERSONALITY_NPY \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --lr $LR \
        --hidden_dim $HIDDEN \
        --dropout $DROPOUT \
        --patience $PATIENCE \
        --target_t $TARGET_T \
        --seed $SEED \
        --experiment_name $EXP_NAME \
        --device cuda"

    # 加音频/视频特征（G+P 不需要）
    if [[ "$SUBTRACK" != "G+P" ]]; then
        CMD="$CMD --audio_feature $AUDIO --video_feature $VIDEO"
    fi

    # 后台运行，绑定到指定 GPU
    CUDA_VISIBLE_DEVICES=$GPU eval "$CMD" > "$LOGFILE" 2>&1 &
    PID=$!
    PIDS+=("$PID")
    LOGFILES+=("$LOGFILE")
    EXP_NAMES+=("$EXP_NAME")
    echo "[GPU $GPU] PID=$PID 已启动 -> $EXP_NAME"
done

echo ""
echo "=========================================="
echo " 全部 ${#PIDS[@]} 个进程已启动，开始等待..."
echo " 实时查看某个实验进度："
echo "   tail -f parallel_logs/<实验名>.log"
echo "   tail -f parallel_logs/*.log  (全部)"
echo "=========================================="

# 等待所有进程结束
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    EXP="${EXP_NAMES[$i]}"
    if wait "$PID"; then
        echo "[OK]   $EXP"
    else
        echo "[FAIL] $EXP (PID=$PID, 查看日志: ${LOGFILES[$i]})"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo " 全部实验完成！失败数: $FAILED"
echo "=========================================="
echo ""
echo "汇总结果："
python3 << 'PYEOF'
import json, glob, os

rows = []
for f in sorted(glob.glob("logs/**/train_result_*.json", recursive=True)):
    try:
        d = json.load(open(f))
        m = d.get("best_val_metrics", {})
        rows.append({
            "track":    d.get("track",""),
            "subtrack": d.get("subtrack",""),
            "task":     d.get("task",""),
            "audio":    d.get("audio_feature","gait"),
            "video":    d.get("video_feature","-"),
            "epoch":    d.get("best_epoch", 0),
            "F1":       f"{m.get('f1', 0):.4f}",
            "ACC":      f"{m.get('acc', 0):.4f}",
            "Kappa":    f"{m.get('kappa', 0):.4f}",
            "CCC":      f"{m.get('ccc', 0):.4f}",
            "RMSE":     f"{m.get('rmse', 0):.4f}",
        })
    except Exception:
        pass

if not rows:
    print("(暂无结果文件)")
else:
    header = ["track","subtrack","task","audio","video","epoch","F1","ACC","Kappa","CCC","RMSE"]
    col_w = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in header}
    fmt = "  ".join(f"{{:{col_w[h]}}}" for h in header)
    print(fmt.format(*header))
    print("-" * (sum(col_w.values()) + 2 * len(header)))
    for r in rows:
        print(fmt.format(*[str(r[h]) for h in header]))

    # 保存 CSV
    import csv
    with open("results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"\n结果已保存到: {os.path.abspath('results_summary.csv')}")
PYEOF
