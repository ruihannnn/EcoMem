#!/bin/bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

run_group() {
    local rates=("$@")
    local pids=()
    echo "========================================"
    echo "  启动批次: ${rates[*]}"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    for rate in "${rates[@]}"; do
        local logfile="results/mincut_d1.01.compress_${rate}.topic_windows_15/pipeline_step456.log"
        bash run_pipeline.sh --step 4,5,6 --compress-rate "${rate}" --batch-k 15 \
            2>&1 | tee "${logfile}" &
        pids+=($!)
        echo "  [r${rate}] PID=$! 已启动"
    done
    echo "  等待本批次完成 (PIDs: ${pids[*]})..."
    for pid in "${pids[@]}"; do
        wait "$pid"
        echo "  PID=$pid 已完成 (exit=$?)"
    done
    echo "========================================"
    echo "  批次 [${rates[*]}] 全部完成"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
}

echo "========== 分批运行 Step 4,5,6 =========="
echo ""

run_group 0.3
run_group 0.4
run_group 0.5
run_group 0.6
run_group 0.8
run_group 0.9
run_group 1.0

echo "========== 全部批次完成! =========="
