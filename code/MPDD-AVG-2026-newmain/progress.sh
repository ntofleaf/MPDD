#!/usr/bin/env bash
# Real-time view of the grid pipeline. Usage:
#   bash progress.sh               # one-shot snapshot
#   watch -n 5 bash progress.sh    # refresh every 5s
cd "$(dirname "$0")"

TOTAL=228
BASELINE=15  # pre-existing train_result files before this grid started

RUNNING=$(ps -eo cmd | grep -E "train\.py --track" | grep -v grep | wc -l)
DONE=$(find logs -path "*pipeline__*" -name "train_result_*.json" 2>/dev/null | wc -l)
PROGRESS=$((DONE - BASELINE))
PCT=$(awk -v p="$PROGRESS" -v t="$TOTAL" 'BEGIN{printf "%.1f", p*100/t}')
LOAD=$(cut -d' ' -f1-3 /proc/loadavg)

echo "================ MPDD-AVG grid pipeline ================"
echo "running workers : $RUNNING / 12"
echo "completed runs  : +$PROGRESS / $TOTAL  (${PCT}%)"
echo "system load avg : $LOAD     (128 cores)"
echo "current stage   : $(tail -n 1 parallel_logs/grid/run.log 2>/dev/null | sed 's/=====//g')"
echo
echo "GPU usage:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
  | awk -F', ' 'BEGIN{print "  GPU   mem        util"} {printf "  %3s   %-9s  %s\n",$1,$2,$3}'
echo
echo "Latest 5 finished runs:"
find logs -path "*pipeline__*" -name "train_result_*.json" -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -5 \
  | while read TS PATH_; do
      EXP=$(echo "$PATH_" | sed 's|.*/pipeline__|pipeline__|;s|/train_result.*||')
      F1=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));b=d.get('best_val_metrics') or {};v=b.get('f1') or b.get('macro_f1');print(f'{v:.3f}' if isinstance(v,(int,float)) else '?')" "$PATH_" 2>/dev/null)
      KP=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));b=d.get('best_val_metrics') or {};v=b.get('kappa');print(f'{v:+.3f}' if isinstance(v,(int,float)) else '?')" "$PATH_" 2>/dev/null)
      WHEN=$(date -d "@${TS%.*}" "+%H:%M:%S")
      printf "  %s  F1=%s  Kappa=%s  %s\n" "$WHEN" "$F1" "$KP" "$EXP"
    done
echo
echo "Workers in flight (max-progress epoch shown per GPU):"
for GPU in 1 4 5 7; do
  echo "  --- GPU $GPU ---"
  ls -t experiments/grid_baseline/stdout_logs/*__gpu${GPU}.log 2>/dev/null \
    | head -3 \
    | while read L; do
        EPOCH_LINE=$(tail -n 1 "$L" 2>/dev/null | grep -oE "Epoch [0-9]+/[0-9]+" | head -1)
        BNAME=$(basename "$L" .log | sed 's|pipeline__||;s|__bilstm_mean.*||')
        printf "    %-50s %s\n" "$BNAME" "${EPOCH_LINE:-(starting)}"
      done
done