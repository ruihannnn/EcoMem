#!/usr/bin/env python3
"""
Step 4: Answer questions using vector retrieval (Episode + optional Semantic).

Supports three modes:
  - Episode only:              default
  - Episode + Semantic:        --use-semantic
  - Adaptive (LLM Router):    --adaptive (auto-decides ep_k/sem_k per question)

Usage:
    # Episode only
    python scripts/step4_answer.py --data dataset/locomo10.json

    # Episode + Semantic
    python scripts/step4_answer.py --data dataset/locomo10.json --use-semantic

    # Adaptive mode
    python scripts/step4_answer.py --data dataset/locomo10.json --use-semantic --adaptive

    # Custom ChromaDB path
    python scripts/step4_answer.py --data dataset/locomo10.json --chroma-db ./chroma_db_custom
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import concurrent.futures
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from lessmem.memory import LessMem


def answer_single_qa(args_tuple):
    """Answer a single QA, return (conv_idx, record, usage)."""
    conv_idx, speaker_a, speaker_b, qa, use_semantic, ep_k, sem_k, adaptive, sem_timestamp, no_episodic = args_tuple

    memory_system = LessMem()
    question = qa.get("question", "")
    response, prompt = memory_system.answer(
        question=question,
        speakers=[speaker_a, speaker_b],
        use_semantic=use_semantic,
        ep_k=ep_k, sem_k=sem_k,
        adaptive=adaptive,
        sem_timestamp=sem_timestamp,
        no_episodic=no_episodic,
    )
    usage = dict(memory_system.llm_client._total_usage)
    record = {
        "question": question,
        "gt_answer": qa.get("answer"),
        "category": qa.get("category"),
        "evidence": qa.get("evidence", []),
        "tracemem_answer": response,
        "input_prompt": prompt,
    }
    return conv_idx, record, usage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Answer LoCoMo questions (vector search)")
    parser.add_argument("--data", required=True,
                        help="Path to LoCoMo JSON dataset")
    parser.add_argument("--max-workers", type=int, default=2000,
                        help="Max concurrent workers (default: 2000)")
    parser.add_argument("--use-semantic", action="store_true", default=False,
                        help="Enable Semantic Memory retrieval")
    parser.add_argument("--output", default=None,
                        help="Output file path (auto-named if omitted)")
    parser.add_argument("--chroma-db", type=str, default=None,
                        help="ChromaDB persistence directory")
    parser.add_argument("--ep-k", type=int, default=1,
                        help="Episode top-k recall count (default: 1)")
    parser.add_argument("--sem-k", type=int, default=20,
                        help="Semantic top-k recall count (default: 20)")
    parser.add_argument("--adaptive", action="store_true", default=False,
                        help="Adaptive retrieval: LLM router decides ep_k/sem_k per question")
    parser.add_argument("--no-episodic", action="store_true", default=False,
                        help="Disable episodic memory (force ep_k=0, compatible with --adaptive)")
    parser.add_argument("--no-sem-timestamp", action="store_true", default=False,
                        help="Do not prepend timestamp to semantic memories")
    args = parser.parse_args()

    if args.chroma_db:
        os.environ["CHROMA_PERSIST_DIR"] = args.chroma_db
        print(f"Using ChromaDB directory: {args.chroma_db}")

    if args.output:
        output_file = args.output
    else:
        if args.adaptive:
            suffix = "_adaptive"
        elif args.use_semantic:
            suffix = "_ep_sem"
        else:
            suffix = "_ep_only"
        output_file = f"conversation_results_vector{suffix}.json"

    dataset = json.loads(Path(args.data).read_text(encoding="utf-8"))

    tasks = []
    for idx, conv_data in enumerate(dataset):
        conv = conv_data.get("conversation", {})
        speaker_a = conv.get("speaker_a", " ")
        speaker_b = conv.get("speaker_b", " ")
        for qa in conv_data.get("qa", []):
            if qa.get("category") == 5:
                continue
            sem_ts = not args.no_sem_timestamp
            tasks.append((idx, speaker_a, speaker_b, qa,
                          args.use_semantic, args.ep_k, args.sem_k,
                          args.adaptive, sem_ts, args.no_episodic))

    if args.adaptive:
        mode = "Adaptive (LLM Router)"
    elif args.use_semantic:
        mode = "Episode + Semantic"
    else:
        mode = "Episode only"
    print(f"Mode: {mode}")
    if not args.adaptive:
        print(f"ep_k: {args.ep_k}, sem_k: {args.sem_k}")
    print(f"Total QA items: {len(tasks)}, max_workers: {args.max_workers}")
    print(f"Output: {output_file}")

    results_by_conv = {}
    completed = 0
    skipped = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(answer_single_qa, t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                conv_idx, record, usage = future.result()
            except Exception as e:
                skipped += 1
                print(f"[SKIP {skipped}] Conversation {task[0]}, "
                      f"question: {task[3].get('question', '?')[:80]}... "
                      f"Error: {type(e).__name__}: {e}")
                continue
            key = f"Conversation_{conv_idx}"
            results_by_conv.setdefault(key, []).append(record)
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
            completed += 1
            if completed % 50 == 0:
                print(f"Progress: {completed}/{len(tasks)}")

    print(f"\n{'='*50}")
    print(f"Token Usage Statistics:")
    print(f"  Total prompt tokens:     {total_usage['prompt_tokens']:,}")
    print(f"  Total completion tokens: {total_usage['completion_tokens']:,}")
    print(f"  Total tokens:            {total_usage['total_tokens']:,}")
    if completed > 0:
        print(f"  Avg prompt tokens/question:     {total_usage['prompt_tokens']/completed:,.1f}")
        print(f"  Avg completion tokens/question: {total_usage['completion_tokens']/completed:,.1f}")
        print(f"  Avg total tokens/question:      {total_usage['total_tokens']/completed:,.1f}")
    print(f"{'='*50}\n")

    print(f"Completed: {completed}, Skipped: {skipped}, Total: {len(tasks)}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_by_conv, f, ensure_ascii=False, indent=2)
    print(f"Done! Saved to {output_file}")


if __name__ == "__main__":
    main()
