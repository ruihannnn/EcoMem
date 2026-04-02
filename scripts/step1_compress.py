#!/usr/bin/env python3
"""
Step 1: Pre-compress LoCoMo dataset conversations using LLMLingua-2.

Reads the original LoCoMo JSON, compresses every dialogue turn's text
via LLMLingua-2, and writes a new JSON file with the same structure.

Usage:
    python scripts/step1_compress.py \
        --input  dataset/locomo10.json \
        --output dataset/locomo10_compressed_r0.6.json \
        --model  /path/to/llmlingua-2-model \
        --rate   0.6 \
        --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from compressor import LlmLingua2Compressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("step1_compress")


def _count_text_chars(conversation: Dict[str, Any]) -> int:
    """Count total characters in all 'text' fields of a conversation."""
    total = 0
    special_keys = {"speaker_a", "speaker_b"}
    for key in conversation:
        if key in special_keys or key.endswith("_date_time"):
            continue
        if key.startswith("session_") and isinstance(conversation[key], list):
            for turn in conversation[key]:
                text = turn.get("text", "")
                total += len(text)
    return total


def compress_conversation(
    conversation: Dict[str, Any],
    compressor: LlmLingua2Compressor,
) -> Dict[str, Any]:
    """Compress every session's dialogue turns within one conversation dict."""
    compressed_conv = copy.deepcopy(conversation)
    special_keys = {"speaker_a", "speaker_b"}

    for key in list(compressed_conv.keys()):
        if key in special_keys or key.endswith("_date_time"):
            continue
        if key.startswith("session_"):
            session_data: List[Dict[str, Any]] = compressed_conv[key]
            if not isinstance(session_data, list):
                continue
            for turn in session_data:
                text = turn.get("text", "")
                if text and text.strip():
                    compressed_text = compressor.compress_text(text)
                    turn["text"] = compressed_text

    return compressed_conv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Pre-compress LoCoMo conversations using LLMLingua-2")
    parser.add_argument("--input", required=True,
                        help="Path to the original LoCoMo JSON dataset")
    parser.add_argument("--output", required=True,
                        help="Path to write the compressed JSON dataset")
    parser.add_argument("--model",
                        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                        help="LLMLingua-2 model name or local path")
    parser.add_argument("--rate", type=float, default=0.6,
                        help="Compression rate (0~1, lower = more aggressive)")
    parser.add_argument("--device", default="cuda",
                        help="Device for the compressor model (cuda / cpu)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Only process the first k conversations (default: all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    logger.info(f"Loading dataset from {input_path}")
    dataset: List[Dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(dataset)} conversations")

    if args.top_k is not None:
        dataset = dataset[: args.top_k]
        logger.info(f"Processing first {args.top_k} conversations only")

    logger.info(f"Initializing LLMLingua-2 compressor "
                f"(model={args.model}, rate={args.rate}, device={args.device})")
    compressor = LlmLingua2Compressor(
        model_name=args.model,
        device_map=args.device,
        use_llmlingua2=True,
        compress_rate=args.rate,
    )
    logger.info("Compressor ready")

    compressed_dataset: List[Dict[str, Any]] = []
    total_original_chars = 0
    total_compressed_chars = 0

    for idx, sample in enumerate(dataset):
        conversation = sample.get("conversation")
        if conversation is None:
            logger.warning(f"[{idx}] No 'conversation' key, skipping")
            compressed_dataset.append(copy.deepcopy(sample))
            continue

        speaker_a = conversation.get("speaker_a", "?")
        speaker_b = conversation.get("speaker_b", "?")
        roles = f"{speaker_a}_{speaker_b}"
        logger.info(f"[{idx + 1}/{len(dataset)}] Compressing conversation: {roles}")

        t0 = time.time()
        orig_chars = _count_text_chars(conversation)
        total_original_chars += orig_chars

        compressed_conv = compress_conversation(conversation, compressor)

        comp_chars = _count_text_chars(compressed_conv)
        total_compressed_chars += comp_chars

        elapsed = time.time() - t0
        ratio = comp_chars / orig_chars * 100 if orig_chars > 0 else 100
        logger.info(
            f"[{idx + 1}/{len(dataset)}] {roles} done in {elapsed:.1f}s  "
            f"({orig_chars} -> {comp_chars} chars, {ratio:.1f}%)")

        out_sample = copy.deepcopy(sample)
        out_sample["conversation"] = compressed_conv
        compressed_dataset.append(out_sample)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(compressed_dataset, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    logger.info(f"Saved compressed dataset to {output_path}")

    overall_ratio = (
        total_compressed_chars / total_original_chars * 100
        if total_original_chars > 0 else 100
    )
    logger.info(
        f"Overall: {total_original_chars} -> {total_compressed_chars} chars "
        f"({overall_ratio:.1f}%)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
