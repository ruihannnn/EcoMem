#!/bin/bash
set -eo pipefail

# ========================================================================
# LessMem 流水线 — Qwen 模型版本
#
# 与 run_pipeline.sh 的区别：
#   - 提取主题 (step 3) 和生成回答 (step 4) 使用 Qwen 模型
#   - Embedding 仍使用原始 API Key
#   - 评估 (step 5) 仍使用原始 API Key 和 gpt-4o-mini
#   - 压缩率 0.8，主题窗口大小 6
#
# 用法示例：
#   bash run_qwen.sh                        # 运行 step 2-6
#   bash run_qwen.sh --step 3,4,5,6         # 跳过分割，仅提取+回答+评估+汇总
# ========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}/scripts:${PYTHONPATH:-}"

# ====== API 配置 ======
# Embedding + 评估：使用原始 Key
export OPENAI_API_KEY="sk-m4MWzcZColN3Du7Ab5xpCMDDrU5KboL223Ezv8OqTE8drWLY"
export BASE_URL="https://chatapi.zjt66.top/v1"

# 提取主题 + 生成回答：使用 Qwen 模型和独立 Key
export LLM_API_KEY="sk-n0ildCfqILSgtU13mqxQCHnACbMcPlYUFBgEwhmGknHPY3jk"
export LLM_MODEL="qwen3-30b-a3b-instruct-2507"

# ====== 固定参数 ======
DATA=""
RAW_DATA="dataset/locomo10.json"
COMPRESS_RATE=0.8
COMPRESS_MODEL="microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
COMPRESS_DEVICE="cuda"
SEG_MODE="mincut"
SEG_DELTA=1.01
WORKERS=2000
BATCH_K=6
STEPS=""
TAG=""
EXT_THEMES=""
NO_ADAPTIVE=false
NO_EPISODIC=false
NO_SEMANTIC=false

# ====== 解析命令行参数 ======
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)             STEPS="$2";                  shift 2 ;;
        --data)             DATA="$2";                   shift 2 ;;
        --raw-data)         RAW_DATA="$2";               shift 2 ;;
        --compress-rate)    COMPRESS_RATE="$2";           shift 2 ;;
        --compress-model)   COMPRESS_MODEL="$2";         shift 2 ;;
        --compress-device)  COMPRESS_DEVICE="$2";        shift 2 ;;
        --segment-mode)     SEG_MODE="$2";               shift 2 ;;
        --delta)            SEG_DELTA="$2";              shift 2 ;;
        --max-workers)      WORKERS="$2";                shift 2 ;;
        --batch-k)          BATCH_K="$2";                shift 2 ;;
        --tag)              TAG="$2";                    shift 2 ;;
        --themes)           EXT_THEMES="$2";             shift 2 ;;
        --no-adaptive)      NO_ADAPTIVE=true;            shift   ;;
        --no-episodic)      NO_EPISODIC=true;            shift   ;;
        --no-semantic)      NO_SEMANTIC=true;            shift   ;;
        -h|--help)
            cat <<'HELP'
用法: bash run_qwen.sh [选项]

Qwen 模型配置:
  提取主题/生成回答: qwen3-30b-a3b-instruct-2507 (独立 API Key)
  Embedding/评估:   原始 API Key + gpt-4o-mini

流程控制:
  --step <1,2,3,4,5,6>    要运行的步骤 (默认: 2,3,4,5,6)
  --tag <name>            输出文件标签

数据与压缩:
  --data <path>           压缩后的数据集路径
  --raw-data <path>       原始数据集路径 (用于 step 1)
  --compress-rate <f>     压缩率 (默认: 0.7)

分割:
  --segment-mode <mode>   mincut 或 llm (默认: mincut)
  --delta <f>             MinCut delta 阈值 (默认: 1.01)

记忆构建与检索:
  --max-workers <n>       最大并发数 (默认: 2000)
  --batch-k <n>           每次 LLM 调用的主题数 (默认: 15)
  --no-adaptive           禁用自适应召回
  --no-episodic           禁用情景记忆
  --no-semantic           禁用语义记忆

主题文件:
  --themes <path>         预计算主题文件路径

所有中间结果保存在 results/<tag>/ 目录下。
HELP
            exit 0 ;;
        *)  echo "未知参数: $1 (使用 --help 查看用法)"; exit 1 ;;
    esac
done

# ====== 根据压缩率自动推断 DATA / TAG / THEMES ======
if [[ -z "$DATA" ]]; then
    if [[ "$COMPRESS_RATE" == "1.0" || "$COMPRESS_RATE" == "1" ]]; then
        DATA="dataset/locomo10.json"
    else
        DATA="dataset/locomo10_compressed_r${COMPRESS_RATE}.json"
    fi
fi
if [[ -z "$TAG" ]]; then
    TAG="mincut_d${SEG_DELTA}.compress_${COMPRESS_RATE}.topic_windows_${BATCH_K}.qwen"
    [[ "$NO_ADAPTIVE" == true ]] && TAG="${TAG}.no_adaptive"
    [[ "$NO_EPISODIC" == true ]] && TAG="${TAG}.no_episodic"
    [[ "$NO_SEMANTIC" == true ]] && TAG="${TAG}.no_semantic"
fi

# ====== 校验 API Key ======
if [[ -z "$OPENAI_API_KEY" || "$OPENAI_API_KEY" == "your-api-key-here" ]]; then
    echo "错误: OPENAI_API_KEY 未设置。"
    exit 1
fi

# ====== 辅助函数 ======
should_run() { [[ -z "$STEPS" ]] && return 0; echo ",$STEPS," | grep -q ",$1,"; }

# ====== 输出路径 ======
RESULT_DIR="results/${TAG}"

DIR_COMPRESS="${RESULT_DIR}/step1_compress"
DIR_SEGMENT="${RESULT_DIR}/step2_segment"
DIR_MEMORY="${RESULT_DIR}/step3_build_memory"
DIR_ANSWER="${RESULT_DIR}/step4_answer"
DIR_EVAL="${RESULT_DIR}/step5_evaluate"
DIR_SCORES="${RESULT_DIR}/step6_scores"
LOG_DIR="${RESULT_DIR}/logs"

mkdir -p "$DIR_COMPRESS" "$DIR_SEGMENT" "$DIR_MEMORY" \
         "$DIR_ANSWER" "$DIR_EVAL" "$DIR_SCORES" "$LOG_DIR"

COMPRESSED="${DIR_COMPRESS}/compressed.json"
if [[ -n "$EXT_THEMES" ]]; then
    THEMES="$EXT_THEMES"
else
    THEMES="${DIR_SEGMENT}/themes.json"
fi
RAW_JSON="${DIR_MEMORY}/raw_memories.json"
CHROMA_DB="${DIR_MEMORY}/chroma_db"
ANSWERS="${DIR_ANSWER}/answers.json"
EVAL="${DIR_EVAL}/evaluation.json"

# ====== 打印配置 ======
echo "=========================================="
echo "  LessMem Pipeline (Qwen)"
echo "=========================================="
echo "  LLM 模型:      $LLM_MODEL"
echo "  评估模型:      gpt-4o-mini (不变)"
echo "  数据集:        $DATA"
echo "  压缩率:        $COMPRESS_RATE"
echo "  分割模式:      $SEG_MODE"
[[ "$SEG_MODE" == "mincut" ]] && echo "  Delta 阈值:    $SEG_DELTA"
echo "  并发数:        $WORKERS"
echo "  批次大小:      $BATCH_K"
echo "  标签:          $TAG"
echo "  主题文件:      $THEMES"
echo "  运行步骤:      ${STEPS:-2,3,4,5,6}"
echo "  结果目录:      $RESULT_DIR/"
echo "=========================================="
echo ""

# ====== Step 1: 预压缩 (可选) ======
if should_run 1; then
    echo ">>> Step 1/6: 预压缩对话文本"
    python scripts/step1_compress.py \
        --input "$RAW_DATA" --output "$COMPRESSED" \
        --model "$COMPRESS_MODEL" --rate "$COMPRESS_RATE" \
        --device "$COMPRESS_DEVICE" \
        2>&1 | tee "$LOG_DIR/step1_compress.log"
    DATA="$COMPRESSED"
    echo ">>> Step 1 完成 -> $COMPRESSED"
    echo ""
else
    if [[ ! -f "$COMPRESSED" && -f "$DATA" ]]; then
        echo ">>> Step 1 跳过: 复制压缩数据 $DATA -> $COMPRESSED"
        cp "$DATA" "$COMPRESSED"
    fi
fi

# ====== Step 2: 主题分割 ======
if should_run 2; then
    echo ">>> Step 2/6: 主题分割 ($SEG_MODE)"
    ARGS="--data $DATA --output $THEMES --mode $SEG_MODE"
    [[ "$SEG_MODE" == "mincut" ]] \
        && ARGS="$ARGS --delta $SEG_DELTA" \
        || ARGS="$ARGS --max-workers $WORKERS"
    python scripts/step2_segment.py $ARGS \
        2>&1 | tee "$LOG_DIR/step2_segment.log"
    echo ">>> Step 2 完成 -> $THEMES"
    echo ""
else
    if [[ ! -f "$THEMES" ]]; then
        PRECOMPUTED_THEMES="dataset/themes/themes_r${COMPRESS_RATE}.json"
        if [[ -f "$PRECOMPUTED_THEMES" ]]; then
            echo ">>> Step 2 跳过: 复制主题文件 $PRECOMPUTED_THEMES -> $THEMES"
            cp "$PRECOMPUTED_THEMES" "$THEMES"
        else
            echo "警告: 预计算主题文件 $PRECOMPUTED_THEMES 不存在，step 2 未执行"
        fi
    fi
fi

# ====== Step 3: LLM 提取 + 构建记忆库 (使用 Qwen) ======
if should_run 3; then
    echo ">>> Step 3/6: 构建记忆库 (LLM: $LLM_MODEL)"
    [[ ! -f "$THEMES" ]] && { echo "错误: $THEMES 不存在，请先运行 step 2"; exit 1; }
    export CHROMA_PERSIST_DIR="$CHROMA_DB"

    echo "--- LLM 提取记忆 (Qwen) ---"
    python scripts/step3_build_memory.py \
        --themes "$THEMES" --max-workers "$WORKERS" \
        -k "$BATCH_K" --raw-json "$RAW_JSON" --stage extract \
        2>&1 | tee "$LOG_DIR/step3a_extract.log"

    echo "--- 存入 ChromaDB ---"
    python scripts/step3_build_memory.py \
        --themes "$THEMES" --max-workers "$WORKERS" \
        --raw-json "$RAW_JSON" --stage store \
        2>&1 | tee "$LOG_DIR/step3b_store.log"

    echo ">>> Step 3 完成 -> $CHROMA_DB"
    echo ""
fi

# ====== Step 4: 向量检索回答 (使用 Qwen) ======
if should_run 4; then
    if [[ ! -d "$CHROMA_DB" ]]; then
        BASE_TAG="mincut_d${SEG_DELTA}.compress_${COMPRESS_RATE}.topic_windows_${BATCH_K}.qwen"
        BASE_CHROMA="results/${BASE_TAG}/step3_build_memory/chroma_db"
        if [[ -d "$BASE_CHROMA" ]]; then
            echo ">>> 复用已有记忆库: $BASE_CHROMA -> $CHROMA_DB"
            ln -sf "$(realpath "$BASE_CHROMA")" "$CHROMA_DB"
        fi
    fi

    STEP4_ARGS="--data $DATA --max-workers $WORKERS --output $ANSWERS --chroma-db $CHROMA_DB"
    [[ "$NO_SEMANTIC" != true ]] && STEP4_ARGS="$STEP4_ARGS --use-semantic"
    [[ "$NO_ADAPTIVE" != true ]] && STEP4_ARGS="$STEP4_ARGS --adaptive"
    [[ "$NO_EPISODIC" == true ]] && STEP4_ARGS="$STEP4_ARGS --no-episodic"

    echo ">>> Step 4/6: 回答问题 (LLM: $LLM_MODEL)"
    export CHROMA_PERSIST_DIR="$CHROMA_DB"
    python scripts/step4_answer.py $STEP4_ARGS \
        2>&1 | tee "$LOG_DIR/step4_answer.log"
    echo ">>> Step 4 完成 -> $ANSWERS"
    echo ""
fi

# ====== Step 5: LLM Judge 评估 (使用原始 gpt-4o-mini) ======
if should_run 5; then
    echo ">>> Step 5/6: LLM Judge 评估 (模型: gpt-4o-mini, 原始 Key)"
    [[ ! -f "$ANSWERS" ]] && { echo "错误: $ANSWERS 不存在，请先运行 step 4"; exit 1; }
    python scripts/step5_evaluate.py \
        --input "$ANSWERS" --output "$EVAL" \
        --max-workers "$WORKERS" \
        2>&1 | tee "$LOG_DIR/step5_evaluate.log"
    echo ">>> Step 5 完成 -> $EVAL"
    echo ""
fi

# ====== Step 6: 汇总分数 ======
if should_run 6; then
    echo ">>> Step 6/6: 汇总分数"
    [[ ! -f "$EVAL" ]] && { echo "错误: $EVAL 不存在，请先运行 step 5"; exit 1; }
    python scripts/step6_scores.py \
        --input "$EVAL" --labels "$TAG" \
        2>&1 | tee "$LOG_DIR/step6_scores.log"
    cp "$LOG_DIR/step6_scores.log" "$DIR_SCORES/step6_scores.log"
    echo ">>> Step 6 完成 -> $DIR_SCORES/step6_scores.log"
fi

# ====== 流水线完成 ======
echo ""
echo "=========================================="
echo "  流水线完成! (Qwen)"
echo "=========================================="
echo "  LLM 模型:      $LLM_MODEL"
echo "  评估模型:      gpt-4o-mini"
echo "  结果目录:      $RESULT_DIR/"
echo "  压缩数据:      $COMPRESSED"
echo "  主题文件:      $THEMES"
echo "  原始记忆:      $RAW_JSON"
echo "  记忆库:        $CHROMA_DB/"
echo "  回答结果:      $ANSWERS"
echo "  评估结果:      $EVAL"
echo "  日志目录:      $LOG_DIR/"
echo "=========================================="
