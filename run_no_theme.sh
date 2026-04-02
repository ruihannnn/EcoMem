#!/bin/bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ========================================================================
# compress_0.7, topic_windows = 15, 不划分主题 (step 3,4,5,6)
# 每个 session 作为一个 theme，使用自适应召回
# ========================================================================
echo "============================================================"
echo "  no_theme (each session = one theme)"
echo "============================================================"

TAG="mincut_d1.01.compress_0.7.topic_windows_15.no_theme"
mkdir -p "results/${TAG}"
bash run_pipeline.sh --step 3,4,5,6 --compress-rate 0.7 --batch-k 15 \
    --themes "dataset/themes/no_theme.compressed_r0.7.json" --tag "$TAG" \
    2>&1 | tee "results/${TAG}/pipeline_step3456.log"
