#!/bin/bash
set -eo pipefail

# ========================================================================
# 批量运行：固定压缩率 0.8，GPT-4o-mini，遍历窗口大小
# 每 2 个窗口为一组并行运行
# ========================================================================

WINDOWS=(1 2 3 4 5 7 8 9 10 13 17 20)
RATE=0.8
STEPS="3,4,5,6"

echo "=========================================="
echo "  批量运行: 压缩率=$RATE, 窗口大小=${WINDOWS[*]}"
echo "  步骤: $STEPS"
echo "  并行度: 每组 2 个"
echo "=========================================="
echo ""

run_one() {
    local w=$1
    local logfile="results/mincut_d1.01.compress_${RATE}.topic_windows_${w}/pipeline_batch.log"
    echo "[$(date '+%H:%M:%S')] 启动: window=$w"
    bash run_pipeline.sh \
        --compress-rate "$RATE" \
        --batch-k "$w" \
        --step "$STEPS" \
        > "$logfile" 2>&1
    echo "[$(date '+%H:%M:%S')] 完成: window=$w"
}

for ((i=0; i<${#WINDOWS[@]}; i+=2)); do
    w1=${WINDOWS[$i]}
    w2=${WINDOWS[$((i+1))]:-}

    if [[ -n "$w2" ]]; then
        echo ">>> 并行运行: window=$w1, window=$w2"
        mkdir -p "results/mincut_d1.01.compress_${RATE}.topic_windows_${w1}" \
                 "results/mincut_d1.01.compress_${RATE}.topic_windows_${w2}"
        run_one "$w1" &
        pid1=$!
        run_one "$w2" &
        pid2=$!
        wait $pid1 $pid2
        echo ">>> 完成: window=$w1, window=$w2"
    else
        echo ">>> 单独运行: window=$w1"
        mkdir -p "results/mincut_d1.01.compress_${RATE}.topic_windows_${w1}"
        run_one "$w1"
        echo ">>> 完成: window=$w1"
    fi
    echo ""
done

echo ""
echo "=========================================="
echo "  全部完成!"
echo "  结果目录:"
for w in "${WINDOWS[@]}"; do
    echo "    results/mincut_d1.01.compress_${RATE}.topic_windows_${w}/"
done
echo "=========================================="
