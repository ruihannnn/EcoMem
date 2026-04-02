#!/usr/bin/env python3
"""
Step 6: Generate score summary from evaluation metrics.

Reads the evaluation metrics JSON from step5 and computes per-category
and overall accuracy scores.

Usage:
    python scripts/step6_scores.py --input evaluation_metrics.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def summarize_scores(eval_path: str, label: str = "") -> None:
    """Print per-category and overall scores from an evaluation JSON."""
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_items = [item for v in data.values() for item in v]
    if not all_items:
        print(f"  [{label}] No evaluation data found")
        return

    cat_scores = defaultdict(list)
    for item in all_items:
        cat_scores[int(item['category'])].append(item['llm_score'])

    title = f"  === {label} ===" if label else "  === Scores ==="
    print(f"\n{title}")
    print(f"  {'Category':<16} {'Mean':>8} {'Count':>6}")
    print(f"  {'-'*34}")

    total_sum, total_count = 0, 0
    for cat in [1, 2, 3, 4]:
        scores = cat_scores.get(cat, [])
        if scores:
            mean = sum(scores) / len(scores)
            print(f"  cat{cat:<12} {mean:>8.4f} {len(scores):>6}")
            total_sum += sum(scores)
            total_count += len(scores)

    overall = total_sum / total_count if total_count else 0
    print(f"  {'-'*34}")
    print(f"  {'Overall':<16} {overall:>8.4f} {total_count:>6}")

    return {
        "per_category": {
            cat: {"mean": sum(scores) / len(scores), "count": len(scores)}
            for cat, scores in cat_scores.items() if scores
        },
        "overall": {"mean": overall, "count": total_count},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Generate score summary from evaluation metrics")
    parser.add_argument("--input", required=True, nargs="+",
                        help="Path(s) to evaluation metrics JSON file(s)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Optional label(s) for each input file")
    args = parser.parse_args()

    labels = args.labels or [f"Run {i+1}" for i in range(len(args.input))]
    if len(labels) < len(args.input):
        labels.extend([f"Run {i+1}" for i in range(len(labels), len(args.input))])

    for path, label in zip(args.input, labels):
        summarize_scores(path, label)

    print()


if __name__ == "__main__":
    main()
