#!/usr/bin/env python3
"""
Step 5: Evaluate answers using LLM Judge.

Reads the answer results JSON from step4, evaluates each QA pair using
an LLM judge (GPT-4o-mini), and outputs evaluation metrics.

Usage:
    python scripts/step5_evaluate.py \\
        --input conversation_results.json \\
        --output evaluation_metrics.json \\
        --max-workers 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import concurrent.futures
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from lessmem.config import MemoryConfig
from openai import OpenAI

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ======================================================================
# LLM Judge
# ======================================================================

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                config = MemoryConfig()
                _client = OpenAI(
                    api_key=config.openai_api_key,
                    base_url=config.base_url,
                )
    return _client


def evaluate_llm_judge(question: str, gold_answer: str,
                       generated_answer: str) -> int:
    """Evaluate using LLM judge. Returns 1 for CORRECT, 0 for WRONG."""
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert grader that determines if answers "
                           "to questions match a gold standard answer.",
            },
            {
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question,
                    gold_answer=gold_answer,
                    generated_answer=generated_answer,
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    label = json.loads(response.choices[0].message.content)["label"]
    return 1 if label == "CORRECT" else 0


# ======================================================================
# Evaluation pipeline
# ======================================================================

def process_item(item_data):
    """Evaluate all QAs for one conversation."""
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item["gt_answer"])
        pred_answer = str(item["tracemem_answer"])
        category = str(item["category"])
        question = str(item["question"])

        if category == "5":
            continue

        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

        local_results[k].append({
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "category": category,
            "llm_score": llm_score,
        })

    return local_results


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Evaluate answers using LLM Judge")
    parser.add_argument("--input", required=True,
                        help="Path to answer results JSON (from step4)")
    parser.add_argument("--output", required=True,
                        help="Path to save evaluation metrics JSON")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="Maximum number of worker threads (default: 5)")
    args = parser.parse_args()

    with open(args.input, "r", encoding='utf-8') as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_item, item_data)
                   for item_data in data.items()]

        iterator = futures
        if HAS_TQDM:
            iterator = tqdm(concurrent.futures.as_completed(futures),
                            total=len(futures))
        else:
            iterator = concurrent.futures.as_completed(futures)

        for future in iterator:
            local_results = future.result()
            with results_lock:
                for k, items in local_results.items():
                    results[k].extend(items)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
