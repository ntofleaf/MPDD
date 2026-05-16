#!/usr/bin/env bash
# =============================================================================
#  Chain wrapper: 同一张 GPU 上先跑 Track2 全量再跑 Track1 全量
#  用法:  bash run_chain_t2_then_t1.sh <route_shell>
#  示例:  bash run_chain_t2_then_t1.sh run_route1_ordinal_v1.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")" || exit 1

ROUTE_SH="${1:?usage: bash run_chain_t2_then_t1.sh <route_shell>}"
[ -x "${ROUTE_SH}" ] || { echo "ERROR: ${ROUTE_SH} 不可执行"; exit 1; }

echo ">>>>>>>>> [chain] $(date +%H:%M:%S)  ${ROUTE_SH}  Track2 全量开始"
bash "${ROUTE_SH}" Track2
T2_RC=$?
echo ">>>>>>>>> [chain] $(date +%H:%M:%S)  ${ROUTE_SH}  Track2 退出码=${T2_RC}"

if [ "${T2_RC}" -ne 0 ]; then
    echo ">>>>>>>>> [chain] Track2 失败，跳过 Track1"
    exit "${T2_RC}"
fi

echo ">>>>>>>>> [chain] $(date +%H:%M:%S)  ${ROUTE_SH}  Track1 全量开始"
bash "${ROUTE_SH}" Track1
T1_RC=$?
echo ">>>>>>>>> [chain] $(date +%H:%M:%S)  ${ROUTE_SH}  Track1 退出码=${T1_RC}"
exit "${T1_RC}"
