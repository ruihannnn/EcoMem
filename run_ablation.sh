#!/bin/bash
set -eo pipefail

# ========================================================================
# LessMem 消融实验脚本
#
# 基础配置：压缩率 0.8，窗口大小 6，MinCut delta=1.01
#
# 三组消融实验（复用基础实验的 step2 主题分割 + step3 记忆库）：
#   1. --no-episodic   去除情景记忆
#   2. --no-semantic    去除语义记忆
#   3. --no-adaptive    去除自适应门控
#
# 用法：
#   bash run_ablation.sh                 # 运行全部 3 组消融
#   bash run_ablation.sh --only 1        # 仅运行第 1 组 (no_episodic)
#   bash run_ablation.sh --only 2        # 仅运行第 2 组 (no_semantic)
#   bash run_ablation.sh --only 3        # 仅运行第 3 组 (no_adaptive)
#   bash run_ablation.sh --only 1,3      # 运行第 1、3 组
# ========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ====== 基础配置 ======
COMPRESS_RATE=0.8
BATCH_K=6
SEG_DELTA=1.01
BASE_TAG="mincut_d${SEG_DELTA}.compress_${COMPRESS_RATE}.topic_windows_${BATCH_K}"
BASE_DIR="results/${BASE_TAG}"

# ====== 解析参数 ======
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        -h|--help)
            head -18 "$0" | tail -16
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

should_run_exp() { [[ -z "$ONLY" ]] && return 0; echo ",$ONLY," | grep -q ",$1,"; }

# ====== 预检查 ======
if [[ ! -d "${BASE_DIR}/step3_build_memory/chroma_db" ]]; then
    echo "错误: 基础实验记忆库不存在: ${BASE_DIR}/step3_build_memory/chroma_db"
    echo "请先运行基础实验: bash run_pipeline.sh --compress-rate ${COMPRESS_RATE} --batch-k ${BATCH_K}"
    exit 1
fi
echo "基础实验目录: ${BASE_DIR}/"
echo ""

# ====== 消融实验定义 ======
declare -a EXP_NAMES=("no_episodic" "no_semantic" "no_adaptive")
declare -a EXP_FLAGS=("--no-episodic" "--no-semantic" "--no-adaptive")
declare -a EXP_DESCS=(
    "去除情景记忆 (Episodic Memory)"
    "去除语义记忆 (Semantic Memory)"
    "去除自适应门控 (Adaptive Gating)"
)

TOTAL=${#EXP_NAMES[@]}
PASS=0
FAIL=0

for i in "${!EXP_NAMES[@]}"; do
    idx=$((i + 1))
    if ! should_run_exp "$idx"; then
        continue
    fi

    name="${EXP_NAMES[$i]}"
    flag="${EXP_FLAGS[$i]}"
    desc="${EXP_DESCS[$i]}"

    echo "=========================================="
    echo "  消融实验 $idx/$TOTAL: $desc"
    echo "=========================================="

    if bash run_pipeline.sh \
        --compress-rate "$COMPRESS_RATE" \
        --batch-k "$BATCH_K" \
        --step 4,5,6 \
        $flag \
        2>&1 | tee "results/${BASE_TAG}.${name}/pipeline_ablation.log"; then
        PASS=$((PASS + 1))
        echo ">>> 消融实验 $idx ($name) 完成"
    else
        FAIL=$((FAIL + 1))
        echo ">>> 消融实验 $idx ($name) 失败!"
    fi
    echo ""
done

# ====== 汇总结果 ======
echo "=========================================="
echo "  消融实验汇总"
echo "=========================================="
echo ""

echo "基础实验 (${BASE_TAG}):"
cat "${BASE_DIR}/step6_scores/step6_scores.log" 2>/dev/null || echo "  [未找到]"
echo ""

for i in "${!EXP_NAMES[@]}"; do
    idx=$((i + 1))
    if ! should_run_exp "$idx"; then
        continue
    fi
    name="${EXP_NAMES[$i]}"
    desc="${EXP_DESCS[$i]}"
    tag="${BASE_TAG}.${name}"
    echo "消融 $idx — $desc (${tag}):"
    cat "results/${tag}/step6_scores/step6_scores.log" 2>/dev/null || echo "  [未找到]"
    echo ""
done

echo "=========================================="
echo "  完成: $PASS 成功, $FAIL 失败"
echo "=========================================="
