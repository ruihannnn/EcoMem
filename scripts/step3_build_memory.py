#!/usr/bin/env python3
"""
Step 3: 从主题 JSON 构建记忆库。

两阶段流程：
  extract: 调用 LLM 提取 episodic + semantic 记忆，保存为 JSON
  store:   将提取结果存入 ChromaDB 向量数据库

Usage:
    # 运行全部（提取 + 存储）
    python scripts/step3_build_memory.py --themes themes.json

    # 仅提取
    python scripts/step3_build_memory.py --themes themes.json --stage extract

    # 仅存储（需要先有提取结果 JSON）
    python scripts/step3_build_memory.py --themes themes.json --stage store --raw-json raw.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import logging
import threading
import traceback
import concurrent.futures
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

from lessmem.memory import LessMem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


# ======================================================================
# 提取：多线程 LLM 调用 + token 统计 + 保存 JSON
# ======================================================================

def run_extract(
    themes_data: List[Dict],
    max_workers: int,
    outer_workers: int,
    topics_per_batch: int,
    output_json: str,
) -> List[Dict]:
    """调用 LLM 提取所有对话的 episodic + semantic 记忆，保存为 JSON。

    总并发 = outer_workers × inner_workers
    outer_workers: 同时处理的 conversation 数
    inner_workers: 每个 conversation 内部的 batch 并发数
    """
    outer_workers = min(outer_workers, len(themes_data))
    inner_workers = max(1, max_workers // outer_workers)

    total = len(themes_data)
    logger.info("=" * 60)
    logger.info(f"EXTRACT: {total} conversation(s), "
                f"outer={outer_workers}, inner={inner_workers}, "
                f"total≈{outer_workers * inner_workers}, k={topics_per_batch}")
    logger.info("=" * 60)

    all_records: List[Dict] = []
    all_usage: List[Dict] = []
    records_lock = threading.Lock()

    def _extract_conv(idx: int, conv_themes: Dict):
        roles = conv_themes['roles']
        theme_groups = conv_themes.get('themes', [])
        if not theme_groups:
            logger.warning(f"[{roles}] No themes, skipping")
            return

        mem = LessMem()
        records, usage = mem.extract_all_from_themes(
            theme_groups=theme_groups,
            roles=roles,
            topics_per_batch=topics_per_batch,
            max_workers=inner_workers,
        )
        with records_lock:
            all_records.extend(records)
            all_usage.append({"roles": roles, **usage,
                              "themes": len(theme_groups),
                              "records": len(records)})
        logger.info(f"Conversation {idx + 1}/{total} [{roles}] OK: "
                    f"{len(records)} records, {usage['total_tokens']:,} tokens")

    with ThreadPoolExecutor(max_workers=outer_workers) as executor:
        futures = {}
        for i, conv in enumerate(themes_data):
            future = executor.submit(_extract_conv, i, conv)
            futures[future] = i

        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Conversation {idx + 1}/{total} failed: "
                             f"{type(e).__name__}: {e}")
                logger.error(traceback.format_exc())

    _log_usage_summary(all_usage)

    logger.info(f"Saving {len(all_records)} record(s) -> {output_json}")
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    return all_records


def _log_usage_summary(usage_stats: List[Dict]):
    if not usage_stats:
        return
    agg = {k: sum(s[k] for s in usage_stats)
           for k in ('prompt_tokens', 'completion_tokens',
                     'total_tokens', 'api_calls', 'themes', 'records')}
    n = len(usage_stats)
    logger.info("=" * 60)
    logger.info("EXTRACT — TOKEN USAGE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Conversations:       {n}")
    logger.info(f"  Total themes:        {agg['themes']}")
    logger.info(f"  Total records:       {agg['records']}")
    logger.info(f"  Total API calls:     {agg['api_calls']}")
    logger.info(f"  Total prompt tokens: {agg['prompt_tokens']:,}")
    logger.info(f"  Total compl. tokens: {agg['completion_tokens']:,}")
    logger.info(f"  Total tokens:        {agg['total_tokens']:,}")
    logger.info(f"  Avg tokens/conv:     {agg['total_tokens'] // n:,}")
    logger.info("=" * 60)
    for s in sorted(usage_stats, key=lambda x: x['roles']):
        logger.info(f"  [{s['roles']}] themes={s['themes']}, "
                    f"records={s['records']}, tokens={s['total_tokens']:,}, "
                    f"calls={s['api_calls']}")
    logger.info("=" * 60)


# ======================================================================
# 存储：将提取结果写入 ChromaDB
# ======================================================================

def run_store(records: List[Dict], max_workers: int = 10) -> None:
    """将提取的记忆存入 ChromaDB。"""
    logger.info("=" * 60)
    logger.info(f"STORE: {len(records)} record(s) -> ChromaDB, "
                f"max_workers={max_workers}")
    logger.info("=" * 60)
    mem = LessMem()
    ep_count, sem_count = mem.store_records_to_chroma(
        records, max_workers=max_workers)
    logger.info(f"Stored {ep_count} episode(s), {sem_count} semantic(s)")


# ======================================================================
# 更新：对已存储的 semantic 记忆执行离线更新（合并/删除冗余）
# ======================================================================

def run_update(
    themes_data: List[Dict],
    score_threshold: float = 0.8,
    top_k: int = 20,
    keep_top_n: int = 10,
    max_workers: int = 5,
) -> None:
    """对所有 conversation 的 semantic 记忆执行离线更新。"""
    user_ids = sorted({conv['roles'] for conv in themes_data})
    logger.info("=" * 60)
    logger.info(f"UPDATE SEMANTIC MEMORIES: {len(user_ids)} user(s), "
                f"threshold={score_threshold}, top_k={top_k}, "
                f"keep_top_n={keep_top_n}, max_workers={max_workers}")
    logger.info("=" * 60)

    mem = LessMem()
    all_stats: List[Dict] = []
    for uid in user_ids:
        logger.info(f"Updating semantic memories for [{uid}]")
        stats = mem.update_semantic_memories(
            user_id=uid,
            score_threshold=score_threshold,
            top_k=top_k,
            keep_top_n=keep_top_n,
            max_workers=max_workers,
        )
        all_stats.append({"roles": uid, **stats})
        logger.info(f"[{uid}] Result: updated={stats['updated']}, "
                    f"deleted={stats['deleted']}, skipped={stats['skipped']}, "
                    f"api_calls={stats['api_calls']}, "
                    f"tokens={stats['total_tokens']:,}")

    _log_update_summary(all_stats)
    logger.info("UPDATE complete.")


def _log_update_summary(stats_list: List[Dict]):
    if not stats_list:
        return
    agg = {
        k: sum(s.get(k, 0) for s in stats_list)
        for k in ('updated', 'deleted', 'skipped',
                  'prompt_tokens', 'completion_tokens',
                  'total_tokens', 'api_calls')
    }
    n = len(stats_list)
    logger.info("=" * 60)
    logger.info("UPDATE — TOKEN USAGE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Conversations:       {n}")
    logger.info(f"  Total updated:       {agg['updated']}")
    logger.info(f"  Total deleted:       {agg['deleted']}")
    logger.info(f"  Total skipped:       {agg['skipped']}")
    logger.info(f"  Total API calls:     {agg['api_calls']}")
    logger.info(f"  Total prompt tokens: {agg['prompt_tokens']:,}")
    logger.info(f"  Total compl. tokens: {agg['completion_tokens']:,}")
    logger.info(f"  Total tokens:        {agg['total_tokens']:,}")
    if n > 0 and agg['total_tokens'] > 0:
        logger.info(f"  Avg tokens/conv:     {agg['total_tokens'] // n:,}")
    logger.info("=" * 60)
    for s in sorted(stats_list, key=lambda x: x['roles']):
        logger.info(f"  [{s['roles']}] updated={s['updated']}, "
                    f"deleted={s['deleted']}, skipped={s['skipped']}, "
                    f"tokens={s['total_tokens']:,}, calls={s['api_calls']}")
    logger.info("=" * 60)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: 从主题 JSON 构建记忆库（LLM 提取 + ChromaDB 存储）")
    parser.add_argument("--themes", required=True,
                        help="主题 JSON 文件路径（step2 输出）")
    parser.add_argument("--max-workers", type=int, default=2000,
                        help="总并发数 = 外层 × 内层 (默认: 2000)")
    parser.add_argument("--outer-workers", type=int, default=10,
                        help="外层并发数：同时处理的 conversation 数 (默认: 10)")
    parser.add_argument("-k", "--topics-per-batch", type=int, default=5,
                        help="每次 LLM 调用处理的主题数 (默认: 5)")
    parser.add_argument("--raw-json", type=str, default=None,
                        help="提取结果 JSON 路径（默认自动生成）")
    parser.add_argument("--stage", type=str, default=None,
                        help="要运行的阶段: extract / store / update（默认: extract+store）")
    parser.add_argument("--update-threshold", type=float, default=0.8,
                        help="记忆更新相似度阈值 (默认: 0.8)")
    args = parser.parse_args()

    stages = set()
    if args.stage:
        stages = {s.strip() for s in args.stage.split(',')}
    run_all = len(stages) == 0

    db_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    raw_json = args.raw_json
    if raw_json is None:
        raw_json = os.path.join(
            os.path.dirname(db_dir) or ".",
            os.path.basename(db_dir) + "_raw_memories.json")

    inner_workers = max(1, args.max_workers // args.outer_workers)
    print(f"ChromaDB dir:       {db_dir}")
    print(f"Themes input:       {args.themes}")
    print(f"Topics per batch:   {args.topics_per_batch}")
    print(f"Max workers:        {args.max_workers} (outer={args.outer_workers} × inner={inner_workers})")
    print(f"Raw JSON path:      {raw_json}")
    print(f"Stages to run:      {sorted(stages) if stages else 'all (extract, store)'}")
    print()

    themes_data = json.loads(Path(args.themes).read_text(encoding="utf-8"))
    print(f"Loaded {len(themes_data)} conversation(s)")

    records = None

    if run_all or 'extract' in stages:
        records = run_extract(
            themes_data,
            max_workers=args.max_workers,
            outer_workers=args.outer_workers,
            topics_per_batch=args.topics_per_batch,
            output_json=raw_json,
        )

    if run_all or 'store' in stages:
        if records is None:
            if os.path.exists(raw_json):
                logger.info(f"Loading records from {raw_json}")
                records = json.loads(
                    Path(raw_json).read_text(encoding="utf-8"))
            else:
                logger.error(f"Store 需要提取结果，{raw_json} 不存在。请先运行 extract。")
                return
        run_store(records, max_workers=args.max_workers)

    if 'update' in stages:
        run_update(
            themes_data,
            score_threshold=args.update_threshold,
            max_workers=args.max_workers,
        )

    logger.info("Step 3 complete.")


if __name__ == "__main__":
    main()
