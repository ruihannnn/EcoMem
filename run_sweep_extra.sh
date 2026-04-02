#!/bin/bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

THEMES_SRC="dataset/themes/themes_r0.7.json"

# ========================================================================
# Part 1: compress_0.7, topic_windows = 23, 25, 27, 30  (step 3,4,5,6)
# ========================================================================
echo "============================================================"
echo "  Part 1: compress_0.7, topic_windows = 23, 25, 27, 30"
echo "============================================================"

for K in 23 25 27 30; do
    TAG="mincut_d1.01.compress_0.7.topic_windows_${K}"
    RESULT_DIR="results/${TAG}"

    mkdir -p "${RESULT_DIR}/step2_segment"
    ln -sf "$(realpath "$THEMES_SRC")" "${RESULT_DIR}/step2_segment/themes.json"

    echo "[$(date '+%H:%M:%S')] 启动 k=${K} ..."
    bash run_pipeline.sh --step 3,4,5,6 --compress-rate 0.7 --batch-k "$K" \
        > "${RESULT_DIR}/pipeline_step3456.log" 2>&1 &
    echo "  [k=${K}] PID=$! -> ${RESULT_DIR}/pipeline_step3456.log"
done

echo "等待 Part 1 完成..."
wait
echo "[$(date '+%H:%M:%S')] Part 1 全部完成"
echo ""

# ========================================================================
# Part 2: compress_0.7, topic_windows = 15, 禁用自适应召回 (step 4,5,6)
#         自动复用已有 chroma_db
# ========================================================================
echo "============================================================"
echo "  Part 2: compress_0.7, topic_windows_15, no_adaptive"
echo "============================================================"

TAG="mincut_d1.01.compress_0.7.topic_windows_15.no_adaptive"
mkdir -p "results/${TAG}"
bash run_pipeline.sh --step 4,5,6 --compress-rate 0.7 --batch-k 15 --no-adaptive \
    2>&1 | tee "results/${TAG}/pipeline_step456.log"

# ========================================================================
# Part 3: compress_0.7, topic_windows = 15
#         自适应召回 + 禁用情景记忆 / 禁用语义记忆 (step 4,5,6)
#         自动复用已有 chroma_db
# ========================================================================
echo "============================================================"
echo "  Part 3: adaptive_no_episodic & adaptive_no_semantic"
echo "============================================================"

BASE="mincut_d1.01.compress_0.7.topic_windows_15"

TAG1="${BASE}.adaptive_no_episodic"
mkdir -p "results/${TAG1}"
echo "[$(date '+%H:%M:%S')] 启动 adaptive_no_episodic ..."
bash run_pipeline.sh --step 4,5,6 --compress-rate 0.7 --batch-k 15 \
    --no-episodic --tag "$TAG1" \
    > "results/${TAG1}/pipeline_step456.log" 2>&1 &
echo "  PID=$! -> results/${TAG1}/pipeline_step456.log"

TAG2="${BASE}.adaptive_no_semantic"
mkdir -p "results/${TAG2}"
echo "[$(date '+%H:%M:%S')] 启动 adaptive_no_semantic ..."
bash run_pipeline.sh --step 4,5,6 --compress-rate 0.7 --batch-k 15 \
    --no-semantic --tag "$TAG2" \
    > "results/${TAG2}/pipeline_step456.log" 2>&1 &
echo "  PID=$! -> results/${TAG2}/pipeline_step456.log"

echo "等待 Part 3 完成..."
wait
echo "[$(date '+%H:%M:%S')] Part 3 全部完成"

echo ""
echo "============================================================"
echo "  全部完成!"
echo "============================================================"
