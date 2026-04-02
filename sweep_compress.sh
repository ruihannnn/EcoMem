#!/bin/bash
set -eo pipefail

# ========================================================================
# 超参数扫描：遍历压缩率 compress_rate=0.3..1.0（排除 0.7）
#
# 固定 topic_windows (batch_k) = 15，使用 mincut_d1.01 分割结果。
# 每个压缩率需要先确保对应的 themes.json 存在（由 step2 生成），
# 然后运行 step 3~6（记忆构建 → 回答 → 评估 → 汇总）。
#
# 用法：
#   bash sweep_compress.sh                  # 并行运行所有压缩率（每批 3 组）
#   bash sweep_compress.sh --parallel 4     # 每批 4 组并行
#   bash sweep_compress.sh --rates 0.3,0.5  # 仅运行指定压缩率
# ========================================================================

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONPATH="scripts:${PYTHONPATH:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-m4MWzcZColN3Du7Ab5xpCMDDrU5KboL223Ezv8OqTE8drWLY}"
export BASE_URL="${BASE_URL:-https://chatapi.zjt66.top/v1}"

BATCH_K=15
DELTA=1.01
PARALLEL=3
ALL_RATES="0.3 0.4 0.5 0.6 0.8 0.9 1.0"
RATES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)  PARALLEL="$2"; shift 2 ;;
        --rates)     RATES="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
        --batch-k)   BATCH_K="$2"; shift 2 ;;
        -h|--help)
            cat <<'HELP'
用法: bash sweep_compress.sh [选项]

选项:
  --parallel <n>       每批并行数 (默认: 3)
  --rates <r1,r2,...>  仅运行指定压缩率 (默认: 0.3,0.4,0.5,0.6,0.8,0.9,1.0)
  --batch-k <n>        topics_per_batch (默认: 15)

示例:
  bash sweep_compress.sh                          # 运行全部
  bash sweep_compress.sh --rates 0.3,0.5 --parallel 2
HELP
            exit 0 ;;
        *)  echo "未知参数: $1 (使用 --help 查看用法)"; exit 1 ;;
    esac
done

[[ -z "$RATES" ]] && RATES="$ALL_RATES"

echo "========================================"
echo "  Sweep: compress_rate"
echo "========================================"
echo "  压缩率:        $RATES"
echo "  batch_k:       $BATCH_K"
echo "  delta:         $DELTA"
echo "  并行度:        $PARALLEL"
echo "========================================"
echo ""

get_dataset() {
    local rate=$1
    if [[ "$rate" == "1.0" ]]; then
        echo "dataset/locomo10.json"
    else
        echo "dataset/locomo10_compressed_r${rate}.json"
    fi
}

run_one() {
    local rate=$1
    local TAG="mincut_d1.01.compress_${rate}.topic_windows_${BATCH_K}"
    local RESULT_DIR="results/${TAG}"
    local SEG_DIR="results/mincut_d1.01.compress_${rate}/step2_segment"
    local THEMES="${SEG_DIR}/themes.json"
    local DATA
    DATA=$(get_dataset "$rate")
    local LOG_FILE="${RESULT_DIR}/sweep.log"

    if [[ ! -f "$DATA" ]]; then
        echo "[$(date '+%H:%M:%S')] r=${rate} 跳过: $DATA 不存在"
        return 1
    fi

    # Step 2: 确保 themes.json 存在
    if [[ ! -f "$THEMES" ]]; then
        echo "[$(date '+%H:%M:%S')] r=${rate} 运行 step 2 (mincut 分割)..."
        mkdir -p "$SEG_DIR"
        python3 scripts/step2_segment.py \
            --data "$DATA" --output "$THEMES" \
            --mode mincut --delta "$DELTA" \
            > "${SEG_DIR}/step2_segment.log" 2>&1
        echo "[$(date '+%H:%M:%S')] r=${rate} step 2 完成"
    else
        echo "[$(date '+%H:%M:%S')] r=${rate} themes.json 已存在，跳过 step 2"
    fi

    # 创建结果目录并链接 themes.json
    mkdir -p "${RESULT_DIR}/step2_segment"
    ln -sf "$(realpath "$THEMES")" "${RESULT_DIR}/step2_segment/themes.json"

    echo "[$(date '+%H:%M:%S')] r=${rate} 启动 step 3-6 (k=${BATCH_K})..."

    bash run_pipeline.sh \
        --step 3,4,5,6 \
        --data "$DATA" \
        --tag "$TAG" \
        --batch-k "$BATCH_K" \
        > "$LOG_FILE" 2>&1

    echo "[$(date '+%H:%M:%S')] r=${rate} 完成 ✓"
}

export -f run_one get_dataset
export BATCH_K DELTA

PIDS=()
RATE_MAP=()

for rate in $RATES; do
    run_one "$rate" &
    PIDS+=($!)
    RATE_MAP+=("$rate")

    if (( ${#PIDS[@]} >= PARALLEL )); then
        for i in "${!PIDS[@]}"; do
            wait "${PIDS[$i]}" || echo "警告: r=${RATE_MAP[$i]} 异常退出"
        done
        PIDS=()
        RATE_MAP=()
    fi
done

for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || echo "警告: r=${RATE_MAP[$i]} 异常退出"
done

echo ""
echo "========================================"
echo "  全部完成!"
echo "========================================"
echo "  结果目录:"
for rate in $RATES; do
    TAG="mincut_d1.01.compress_${rate}.topic_windows_${BATCH_K}"
    SCORE_LOG="results/${TAG}/logs/step6_scores.log"
    if [[ -f "$SCORE_LOG" ]]; then
        ACC=$(grep 'Overall' "$SCORE_LOG" | grep -oP '0\.\d+' || echo "N/A")
        echo "    r=${rate}: accuracy=${ACC}  -> results/${TAG}/"
    else
        echo "    r=${rate}: (未完成)  -> results/${TAG}/"
    fi
done
echo "========================================"
