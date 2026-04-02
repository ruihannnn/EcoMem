#!/bin/bash
set -eo pipefail

# ========================================================================
# 超参数扫描：遍历 topics_per_batch (k=1..20)
#
# 固定使用 mincut_d1.01 的主题分割结果（themes.json），
# 仅重新运行 step 3~6（记忆构建 → 回答 → 评估 → 汇总）。
# 同时并行运行 PARALLEL 组实验以加速。
#
# 用法：
#   bash sweep_k.sh              # 并行运行 k=1..20（每批 3 组）
#   bash sweep_k.sh 5 15         # 仅运行 k=5..15
#   bash sweep_k.sh 1 20 4       # k=1..20，每批 4 组并行
# ========================================================================

cd "$(dirname "${BASH_SOURCE[0]}")"

THEMES_SRC="results/mincut_d1.01/step2_segment/themes.json"

if [[ ! -f "$THEMES_SRC" ]]; then
    echo "错误: $THEMES_SRC 不存在"
    exit 1
fi

K_START=${1:-1}
K_END=${2:-20}
PARALLEL=${3:-3}

echo "========================================"
echo "  Sweep: topics_per_batch k=${K_START}..${K_END}"
echo "  并行度: ${PARALLEL}"
echo "  Themes: $THEMES_SRC"
echo "========================================"

run_one() {
    local k=$1
    local TAG="mincut_d1.01.compress_0.7.topic_windows_${k}"
    local RESULT_DIR="results/${TAG}"
    local LOG_FILE="${RESULT_DIR}/sweep.log"

    mkdir -p "${RESULT_DIR}/step2_segment"
    ln -sf "$(realpath "$THEMES_SRC")" "${RESULT_DIR}/step2_segment/themes.json"

    echo "[$(date '+%H:%M:%S')] 启动 k=${k} ..."

    bash run_pipeline.sh \
        --step 3,4,5,6 \
        --tag "$TAG" \
        --batch-k "$k" \
        > "$LOG_FILE" 2>&1

    echo "[$(date '+%H:%M:%S')] k=${k} 完成 ✓"
}

export -f run_one
export THEMES_SRC

PIDS=()
for k in $(seq "$K_START" "$K_END"); do
    run_one "$k" &
    PIDS+=($!)

    if (( ${#PIDS[@]} >= PARALLEL )); then
        for pid in "${PIDS[@]}"; do
            wait "$pid" || echo "警告: 进程 $pid 异常退出"
        done
        PIDS=()
    fi
done

for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "警告: 进程 $pid 异常退出"
done

echo ""
echo "========================================"
echo "  全部完成! k=${K_START}..${K_END}"
echo "========================================"
echo "  结果目录: results/mincut_d1.01_k*/"
echo "  各实验日志: results/mincut_d1.01_k*/sweep.log"
echo "========================================"
