#!/bin/bash
set -eo pipefail

# ========================================================================
# 批量运行：固定窗口大小 6，GPT-4o-mini，遍历压缩率 0.3~1.0（跳过 0.7）
# ========================================================================

RATES=(0.3 0.4 0.5 0.6 0.8 0.9 1.0)
BATCH_K=6
STEPS="3,4,5,6"

echo "=========================================="
echo "  批量运行: 窗口大小=$BATCH_K, 压缩率=${RATES[*]}"
echo "  步骤: $STEPS"
echo "=========================================="
echo ""

for rate in "${RATES[@]}"; do
    echo ""
    echo "######################################################"
    echo "  开始: compress_rate=$rate, batch_k=$BATCH_K"
    echo "######################################################"
    echo ""

    bash run_pipeline.sh \
        --compress-rate "$rate" \
        --batch-k "$BATCH_K" \
        --step "$STEPS"

    echo ""
    echo "######################################################"
    echo "  完成: compress_rate=$rate, batch_k=$BATCH_K"
    echo "######################################################"
    echo ""
done

echo ""
echo "=========================================="
echo "  全部完成!"
echo "  结果目录:"
for rate in "${RATES[@]}"; do
    echo "    results/mincut_d1.01.compress_${rate}.topic_windows_${BATCH_K}/"
done
echo "=========================================="
