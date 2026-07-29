<div align="center">
  <img src="assets/remember.png" alt="EcoMem Logo" width="400"/>
  <h1>EcoMem</h1>
  <p><strong>面向长对话智能体的轻量级结构化记忆框架</strong></p>
</div>

## 概述

`EcoMem` 的核心目标不是把整段历史对话直接塞回模型，而是把长对话先压缩、再结构化、最后按问题自适应召回。基于源码实现，系统主链路为：**预压缩** -> **MinCut 主题切分** -> **结构化记忆抽取与索引** -> **自适应召回问答** -> **自动评估**。

这一路径对应 `run_pipeline.sh` 中的 Step 1-6，其中真正构成方法主体的是前三个阶段：压缩、结构化索引、自适应召回。

## 方法介绍

### EcoMem 架构

EcoMem 可以概括为一个三段式记忆流水线。第一段通过预压缩和基于图切分的主题划分，把原始长对话整理成更紧凑、语义更集中的片段；第二段把每个片段进一步转成两种可检索的记忆形态，即保留叙事上下文的情景记忆与可独立检索的语义记忆；第三段在回答问题时不固定检索深度，而是先判断问题类型，再动态调节两类记忆的召回范围，从而在准确率与上下文成本之间取得平衡。

### 模块 1：压缩模块（预压缩 + MinCut）

这一模块对应 `scripts/step1_compress.py`、`scripts/compressor.py` 与 `scripts/step2_segment.py`。其目标不是直接生成摘要，而是在尽量保留对话结构的前提下，先缩短输入，再把混杂的话题切成后续可建模的主题单元。

在预压缩阶段，系统会遍历每条对话中各个 `session` 的每个 turn，并调用 `LlmLingua2Compressor.compress_text()` 对 `text` 字段逐条压缩。这里保留了原始会话层级、说话人角色以及各 session 的时间信息，因此压缩发生在 utterance 级别，而不是把整段对话揉成一个摘要。

在主题切分阶段，默认实现并不是普通阈值切分，而是 `mincut`。系统先对每个 utterance 生成 embedding，再构建 utterance 间的相似度矩阵，并通过 `ncut_dp()` 在 `k = 1 ... MAX_K` 的候选切分数上做动态规划求解 normalized cut。随后，`choose_k_delta()` 不直接固定主题数，而是依据 `delta` 阈值自动决定最终的切分粒度。这使得对话越复杂，主题数可以自然增多；对话越集中，则保留更粗粒度的主题块。

这一阶段的输出是统一格式的 `themes.json`。每个主题都带有 `theme_name`、对应的 `session_key`、主题时间戳以及消息列表，因此它不是单纯的分段结果，而是后续结构化记忆抽取的标准输入接口。

### 模块 2：结构化索引（提取语义记忆和情景记忆）

这一模块对应 `scripts/step3_build_memory.py` 以及 `scripts/EcoMem/` 下的 `memory.py`、`theme_extractor.py`、`chroma.py`、`data_utils.py`、`prompts.py`。它的核心作用是把主题级对话片段转成两套互补的记忆表示。

首先，`EcoMem.extract_all_from_themes()` 会把多个主题按 batch 送入 `ThemeExtractor.extract_memories_batch()`。在 prompt 设计上，模型需要同时输出两部分结果：

- `episodic_summary`：以 `As of [timestamp]` 开头的主题级叙事摘要，保留谁在什么时间说了什么、发生了什么、涉及哪些活动、地点与情绪。
- `semantic_memories`：拆分成多条自包含的原子事实，每条都尽量写成显式主语、显式时间、显式事件的独立句子。

源码里还对抽取结果施加了两类关键约束。第一类是时间消歧：prompt 要求把 `yesterday`、`last year`、`two months ago` 等相对时间转换为绝对日期，随后 `normalize_timestamp()` 再把主题时间戳统一规范到 ISO 格式。第二类是指代消歧：prompt 明确要求把 `he`、`she`、`they` 等代词替换成真实姓名，从而提升后续检索时的可匹配性。

在索引阶段，`EcoMem.store_records_to_chroma()` 会把上述两类结果分别写入 Chroma。具体来说，情景记忆通过 `add_episode()` 写入 `*_episodes` collection，语义记忆通过 `add_semantic_memory()` 写入 `*_semantic` collection；二者共享同一个 `user_id = speaker_a_speaker_b`，但以不同集合分别管理。这样做的好处是，系统既能保留主题级叙事上下文，又能保留更细粒度、可跨主题复用的原子事实。

换言之，模块 2 的重点不是“把摘要存起来”，而是把原始长对话重写成一套双通道、可向量检索、带时间锚点的结构化记忆索引。

### 模块 3：自适应召回

这一模块对应 `scripts/step4_answer.py`、`scripts/EcoMem/vector_search.py` 与 `scripts/EcoMem/chroma.py`。它回答的问题是：面对不同类型的问题，系统应该召回多少情景记忆、多少语义记忆，才能既避免无关上下文，又保留足够的推理证据。

默认情况下，系统会先根据问题在情景记忆库中检索 `ep_k` 条 episode；如果开启 `--use-semantic`，则额外在语义记忆库中检索 `sem_k` 条 semantic memory。真正的自适应能力来自 `VectorSearch._route()`：当 `--adaptive` 打开后，系统会先用 `ROUTER_PROMPT` 对问题进行轻量分类，将其划分为 `multi_hop`、`temporal` 或 `simple` 三类，再自动给出对应的 `ep_k` 与 `sem_k`。

从源码中的路由规则可以看到，这种调节具有明确偏向：`multi_hop` 问题通常需要更高的 `sem_k` 来聚合跨主题事实，`temporal` 问题则会提升 `ep_k` 以保留更多时间线上下文，`simple` 问题使用较小的默认召回范围即可。随后系统把召回到的 `Episodes` 与 `Semantic Memories` 拼接成回答提示词，再调用生成模型输出最终答案。

因此，EcoMem 的召回并不是单一路径的向量搜索，而是一个“先判题型、再调召回、再组织上下文”的查询感知式检索过程。

### 三个模块如何衔接

从代码执行链路来看，三部分是逐层收缩再逐层展开的关系：

1. 压缩模块把长对话从原始 turn 序列变成紧凑、主题清晰的 `themes`。
2. 结构化索引把 `themes` 变成两套长期可复用的记忆库。
3. 自适应召回再根据问题特征，从两套记忆库中动态取回最合适的证据。

这也是 EcoMem 区别于“全量拼接上下文”方案的核心：它不依赖单次超长上下文窗口，而是通过压缩、显式结构化与查询感知检索，把长期记忆变成一个可维护、可扩展、可控成本的系统过程。

## 流水线概览

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1 | `scripts/step1_compress.py` | 使用 LLMLingua-2 对每个对话 turn 做预压缩，降低后续处理成本 |
| 2 | `scripts/step2_segment.py` | 主题分割，默认使用 `mincut`，也支持 LLM 分割 |
| 3 | `scripts/step3_build_memory.py` | 批量抽取 `episodic_summary` 与 `semantic_memories`，并写入 ChromaDB |
| 4 | `scripts/step4_answer.py` | 基于 episode / semantic 双通道记忆进行检索问答，支持自适应路由 |
| 5 | `scripts/step5_evaluate.py` | 使用 LLM Judge 评估回答正确性 |
| 6 | `scripts/step6_scores.py` | 按类别汇总分数 |

## 项目结构

```text
EcoMem/
├── scripts/
│   ├── EcoMem/
│   │   ├── memory.py
│   │   ├── config.py
│   │   ├── client.py
│   │   ├── models.py
│   │   ├── chroma.py
│   │   ├── prompts.py
│   │   ├── theme_extractor.py
│   │   ├── vector_search.py
│   │   └── data_utils.py
│   ├── step1_compress.py
│   ├── step2_segment.py
│   ├── step3_build_memory.py
│   ├── step4_answer.py
│   ├── step5_evaluate.py
│   ├── step6_scores.py
│   └── compressor.py
├── dataset/
├── results/
├── run_pipeline.sh
├── pyproject.toml
└── LICENSE
```

## 安装

```bash
conda create -n EcoMem python=3.10
conda activate EcoMem

git clone https://github.com/YimingShu-teay/EcoMem.git
cd EcoMem

pip install -e .
```

如需启用 Step 1 的 LLMLingua-2 预压缩：

```bash
pip install -e ".[compress]"
```

## 环境配置

在 `run_pipeline.sh` 顶部配置 API 参数：

```bash
export OPENAI_API_KEY="your-api-key"
export BASE_URL="https://api.openai.com/v1"
```

## 快速开始

### 运行默认流程（Step 2-6）

```bash
bash run_pipeline.sh --data dataset/locomo10_compressed_r0.7.json
```

### 运行包含预压缩的全流程

```bash
bash run_pipeline.sh --step 1,2,3,4,5,6 --raw-data dataset/locomo10.json
```

### 只运行问答与评估

```bash
bash run_pipeline.sh --step 4,5,6
```

## 中间结果

所有结果保存在 `results/<tag>/` 下，默认标签会根据 `delta`、压缩率和 `batch-k` 自动构造成：

```text
mincut_d<delta>.compress_<rate>.topic_windows_<batch_k>
```

目录结构如下：

```text
results/<tag>/
├── step1_compress/
│   └── compressed.json
├── step2_segment/
│   └── themes.json
├── step3_build_memory/
│   ├── raw_memories.json
│   └── chroma_db/
├── step4_answer/
│   └── answers.json
├── step5_evaluate/
│   └── evaluation.json
├── step6_scores/
│   └── step6_scores.log
└── logs/
```

## 单步运行示例

```bash
# Step 1: 预压缩
python scripts/step1_compress.py \
    --input dataset/locomo10.json \
    --output results/EcoMem/step1_compress/compressed.json \
    --rate 0.7

# Step 2: MinCut 主题切分
python scripts/step2_segment.py \
    --data results/EcoMem/step1_compress/compressed.json \
    --output results/EcoMem/step2_segment/themes.json \
    --mode mincut \
    --delta 1.01

# Step 3: 结构化记忆抽取 + 入库
export CHROMA_PERSIST_DIR=results/EcoMem/step3_build_memory/chroma_db
python scripts/step3_build_memory.py \
    --themes results/EcoMem/step2_segment/themes.json \
    --max-workers 2000 \
    --topics-per-batch 10 \
    --raw-json results/EcoMem/step3_build_memory/raw_memories.json \
    --stage extract

python scripts/step3_build_memory.py \
    --themes results/EcoMem/step2_segment/themes.json \
    --max-workers 2000 \
    --raw-json results/EcoMem/step3_build_memory/raw_memories.json \
    --stage store

# Step 4: 语义记忆 + 自适应召回
python scripts/step4_answer.py \
    --data results/EcoMem/step1_compress/compressed.json \
    --output results/EcoMem/step4_answer/answers.json \
    --use-semantic \
    --adaptive \
    --chroma-db results/EcoMem/step3_build_memory/chroma_db
```

## 关键配置

### `run_pipeline.sh`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--step` | `2,3,4,5,6` | 指定运行步骤 |
| `--raw-data` | `dataset/locomo10.json` | 原始数据路径 |
| `--data` | 自动推断 | 压缩后数据路径 |
| `--compress-rate` | `0.7` | 预压缩率 |
| `--segment-mode` | `mincut` | 分割模式：`mincut` 或 `llm` |
| `--delta` | `1.01` | MinCut 的 `k` 选择阈值 |
| `--max-workers` | `2000` | 最大并发数 |
| `--batch-k` | `10` | Step 3 每次 LLM 处理的主题数 |
| `--no-adaptive` | `false` | 禁用自适应召回 |
| `--no-episodic` | `false` | 禁用情景记忆 |
| `--no-semantic` | `false` | 禁用语义记忆 |

### 检索模式

- `Episode only`：只检索情景记忆。
- `Episode + Semantic`：同时检索情景记忆和语义记忆。
- `Adaptive`：先路由问题类型，再动态设定 `ep_k` 与 `sem_k`。

## 评估

项目默认在 [LoCoMo](https://github.com/snap-research/locomo) 上评估，并覆盖以下问答类型：

| 类别 | 类型 | 说明 |
|------|------|------|
| 1 | Multi-hop | 需要跨事件聚合的复杂问题 |
| 2 | Temporal | 时间相关问题 |
| 3 | Open-ended | 开放式或主观问题 |
| 4 | Single-hop | 简单事实回忆 |

## 引用

```bibtex
@article{shu2026tracemem,
  title={TraceMem: Weaving Narrative Memory Schemata from User Conversational Traces},
  author={Shu, Yiming and Liu, Pei and Zhang, Tiange and Gao, Ruiyang and Ma, Jun and Sun, Chen},
  journal={arXiv preprint arXiv:2602.09712},
  year={2026}
}
```

## 相关项目

- [A-mem](https://github.com/agiresearch/A-mem)
- [Nemori](https://github.com/nemori-ai/nemori)
- [LightMem](https://github.com/zjunlp/LightMem)

## 许可证

本项目使用 MIT 许可证。详见 [LICENSE](LICENSE)。
