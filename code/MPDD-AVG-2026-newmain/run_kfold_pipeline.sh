#!/usr/bin/env bash
# Step 4-7: 5-fold validation + fold ensemble for the top combos picked from
# the grid baseline. Produces submissions_v2/*.zip independent of submissions/.
set -euo pipefail
cd "$(dirname "$0")"

TOP_N="${TOP_N:-3}"
RECALL_FLOOR="${RECALL_FLOOR:-0.3}"
GPUS="${GPUS:-1,1,1,4,4,4,5,5,5,7,7,7}"
SEED="${SEED:-42}"
N_SPLITS="${N_SPLITS:-5}"

# ---- Step 1: pick top combos per cell with gates -----------------------------
python select_top_combos.py --top_n "$TOP_N" --recall_floor "$RECALL_FLOOR"

# ---- Step 2: run kfold for every selected combo ------------------------------
GPUS="$GPUS" SEED="$SEED" N_SPLITS="$N_SPLITS" bash run_kfold_for_selected.sh

# ---- Step 3+4+5: select best, fold-ensemble test inference, pack -------------
CUDA_VISIBLE_DEVICES=6 python pack_submissions_v2.py --out_dir sub/submissions_v2

echo
echo "===== DONE ====="
ls -la sub/submissions_v2/*.zip 2>/dev/null || echo "(no zips written)"