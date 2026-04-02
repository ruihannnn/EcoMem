#!/usr/bin/env python3
"""
Step 2: Topic segmentation — split conversations into theme-based topics.

Supports two segmentation modes:
  --mode mincut : Normalized-Cut DP segmentation with delta threshold (default δ=1.01)
  --mode llm    : LLM-based segmentation using TOPIC_SEGMENT_PROMPT

Both modes output the same themes JSON format for downstream consumption by step3.

Usage:
    # MinCut mode (default)
    python scripts/step2_segment.py \\
        --data dataset/locomo10_compressed_r0.7.json \\
        --output logs/themes.json \\
        --mode mincut --delta 1.01

    # LLM mode
    python scripts/step2_segment.py \\
        --data dataset/locomo10_compressed_r0.7.json \\
        --output logs/themes.json \\
        --mode llm --max-workers 5
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
import logging
import traceback
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("step2_segment")


# ======================================================================
# MinCut (Normalized Cut DP) mode
# ======================================================================

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_DELTA = 1.01
MAX_K = 20


class EmbeddingCache:
    def __init__(self, path: str):
        self.store: Dict[str, List[float]] = {}
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.store = pickle.load(f)
            print(f"  [cache] {len(self.store)} vectors loaded")

    def get(self, text: str) -> Optional[List[float]]:
        return self.store.get(text)


def get_embedding(text: str, cache: EmbeddingCache,
                  client: openai.OpenAI) -> List[float]:
    v = cache.get(text)
    if v is not None:
        return v
    for attempt in range(5):
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL, input=[text.replace("\n", " ")], timeout=60)
            emb = resp.data[0].embedding
            cache.store[text] = emb
            return emb
        except Exception:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                raise
    return []


def format_utterance_content(utt: dict) -> str:
    text = utt.get("text", "")
    blip = utt.get("blip_caption", "")
    if blip:
        text += f" Image: {blip}"
    return text


def build_sim_matrix(embeddings: List[List[float]]) -> np.ndarray:
    E = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    sim = (E / norms) @ (E / norms).T
    np.fill_diagonal(sim, 0.0)
    return np.maximum(sim, 0.0)


def ncut_dp(sim: np.ndarray, max_k: int) -> Dict:
    """Normalized Cut DP: returns {k: (ncut_val, segment_starts)} for k=1..max_k."""
    n = sim.shape[0]
    max_k = min(max_k, n)

    degree = sim.sum(axis=1)
    deg_prefix = np.concatenate(([0.0], np.cumsum(degree)))
    p2 = np.zeros((n + 1, n + 1))
    p2[1:, 1:] = np.cumsum(np.cumsum(sim, axis=0), axis=1)

    def vol(a, b):
        return deg_prefix[b + 1] - deg_prefix[a]

    def w_int(a, b):
        return p2[b + 1][b + 1] - p2[a][b + 1] - p2[b + 1][a] + p2[a][a]

    def ncut_term(a, b):
        v = vol(a, b)
        return (1.0 - w_int(a, b) / v) if v > 1e-10 else 0.0

    INF = float("inf")
    C = np.full((n, max_k + 1), INF)
    P = np.full((n, max_k + 1), -1, dtype=int)
    for i in range(n):
        C[i][1] = ncut_term(0, i)
    for k in range(2, max_k + 1):
        for i in range(k - 1, n):
            for j in range(k - 2, i):
                val = C[j][k - 1] + ncut_term(j + 1, i)
                if val < C[i][k]:
                    C[i][k] = val
                    P[i][k] = j

    results = {}
    for k in range(1, max_k + 1):
        if C[n - 1][k] >= INF:
            continue
        seg_starts = []
        cur, cur_k = n - 1, k
        while cur_k > 1:
            j = int(P[cur][cur_k])
            seg_starts.append(j + 1)
            cur, cur_k = j, cur_k - 1
        seg_starts.reverse()
        results[k] = (float(C[n - 1][k]), seg_starts)
    return results


def choose_k_delta(ncut_results: Dict, delta_thr: float) -> int:
    """Pick k by incrementing until Ncut(k) - Ncut(k-1) >= delta_thr."""
    chosen_k = 1
    for k in range(2, max(ncut_results.keys()) + 1):
        if k in ncut_results and (k - 1) in ncut_results:
            delta = ncut_results[k][0] - ncut_results[k - 1][0]
            if delta < delta_thr:
                chosen_k = k
            else:
                break
        else:
            break
    return chosen_k


def _process_session_mincut(
    skey: str, utts: list, timestamp: str,
    cache: EmbeddingCache, client: openai.OpenAI, delta: float,
) -> Tuple[str, List[Dict], int, int]:
    """Process one session: embed → sim matrix → ncut → segments.

    Returns (skey, theme_dicts_without_name, n_topics, n_utts).
    """
    n_utts = len(utts)
    if not utts:
        return (skey, [], 0, 0)

    texts = [format_utterance_content(u) for u in utts]
    embeddings = [get_embedding(t, cache, client) for t in texts]

    if any(not e for e in embeddings):
        messages = [{"role": u["speaker"], "content": t}
                    for u, t in zip(utts, texts)]
        return (skey, [{
            "topic_count": 1,
            "topics": [{"session_key": skey, "timestamp": timestamp,
                        "messages": messages}],
        }], 1, n_utts)

    sim = build_sim_matrix(embeddings)
    mk = min(MAX_K, n_utts)
    ncut_results = ncut_dp(sim, mk)
    k = choose_k_delta(ncut_results, delta)

    if k in ncut_results:
        _, seg_starts = ncut_results[k]
    else:
        seg_starts = []

    boundaries = sorted(set(seg_starts))
    segments = []
    prev = 0
    for b in boundaries:
        if b > prev:
            segments.append((prev, b))
        prev = b
    segments.append((prev, n_utts))

    themes = []
    for start, end in segments:
        messages = [{"role": utts[i]["speaker"], "content": texts[i]}
                    for i in range(start, end)]
        if messages:
            themes.append({
                "topic_count": 1,
                "topics": [{"session_key": skey, "timestamp": timestamp,
                            "messages": messages}],
            })
    return (skey, themes, k, n_utts)


def run_mincut(dataset, args) -> List[Dict]:
    """Run Normalized-Cut DP segmentation on all conversations."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    cache_path = os.path.join(os.path.dirname(args.output), "embedding_cache.pkl")
    cache = EmbeddingCache(cache_path)

    all_themes_data = []
    total_topics = 0

    for entry in dataset:
        conv = entry["conversation"]
        speaker_a, speaker_b = conv["speaker_a"], conv["speaker_b"]
        roles = f"{speaker_a}_{speaker_b}"
        print(f"\nProcessing {roles} ...")

        sessions = []
        session_idx = 1
        while f"session_{session_idx}" in conv:
            skey = f"session_{session_idx}"
            utts = conv[skey]
            timestamp = conv.get(f"{skey}_date_time", "")
            sessions.append((skey, utts, timestamp))
            session_idx += 1

        results_by_skey: Dict[str, Tuple[List[Dict], int, int]] = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    _process_session_mincut,
                    skey, utts, ts, cache, client, args.delta,
                ): skey
                for skey, utts, ts in sessions
            }
            for future in concurrent.futures.as_completed(futures):
                skey_tag = futures[future]
                try:
                    skey_r, themes, n_topics, n_utts = future.result()
                    results_by_skey[skey_r] = (themes, n_topics, n_utts)
                except Exception as e:
                    logger.error(f"  {skey_tag} failed: {e}")
                    logger.error(traceback.format_exc())

        themes_list = []
        global_topic_idx = 0
        for skey, utts, _ in sessions:
            if skey not in results_by_skey:
                continue
            themes, n_topics, n_utts = results_by_skey[skey]
            for theme in themes:
                theme["theme_name"] = f"topic_{global_topic_idx}"
                global_topic_idx += 1
            themes_list.extend(themes)
            print(f"  {skey}: {n_utts} utts -> {n_topics} topics")

        all_themes_data.append({"roles": roles, "themes": themes_list})
        total_topics += global_topic_idx
        print(f"  Total topics for {roles}: {global_topic_idx}")

    return all_themes_data


# ======================================================================
# LLM segmentation mode
# ======================================================================

def _format_segment_prompt(messages):
    """Format messages as <D1>...</D1> etc."""
    if not messages:
        return "", []

    first_msg_metadata = messages[0].get('metadata', {})
    msgs_time = first_msg_metadata.get('dataset_timestamp', 'Unknown Time')

    formatted_lines = [f"Current Time: {msgs_time}"]
    speakers = []

    for i, msg in enumerate(messages):
        speaker = msg.get('role', 'Unknown')
        speakers.append(speaker)
        content = msg.get('content', '')
        text_only = re.sub(r'\[.*?\]', '', content).strip()
        metadata = msg.get('metadata', {})
        blip = metadata.get('blip_caption')

        display_parts = []
        if text_only:
            display_parts.append(text_only)
        if blip:
            display_parts.append(f"Image: {blip}")

        if display_parts:
            final_content = " ".join(display_parts)
            formatted_lines.append(f"<D{i+1}>{speaker}: {final_content}</D{i+1}>")

    return "\n".join(formatted_lines), speakers


def _extract_topic_boundaries(result_text: str, num_messages: int) -> List[Dict]:
    """Parse TOPIC_SEGMENT_PROMPT output — only <intent>, no <semantic>."""
    pattern = r'<D(\d+)>.*?<intent>(.*?)</intent>.*?</D\1>'
    matches = re.findall(pattern, result_text, re.DOTALL)

    if not matches:
        return [{'range': (0, num_messages - 1)}] if num_messages > 0 else []

    matches = sorted([(int(idx), intent.strip()) for idx, intent in matches],
                     key=lambda x: x[0])
    topics = []
    current_start = None

    for idx, intent in matches:
        idx_0based = idx - 1
        if intent == "CHANGE_TOPIC" and current_start is not None:
            topics.append({'range': (current_start, idx_0based - 1)})
            current_start = idx_0based
        if current_start is None:
            current_start = idx_0based

    if current_start is not None:
        last_idx = matches[-1][0] - 1
        topics.append({'range': (current_start, last_idx)})

    return topics


def _format_message_content(msg: Dict) -> str:
    """Produce clean display text from a raw message."""
    content = msg.get('content', '')
    text_only = re.sub(r'\[.*?\]', '', content).strip()
    metadata = msg.get('metadata', {})
    blip = metadata.get('blip_caption') if metadata else None
    parts = []
    if text_only:
        parts.append(text_only)
    if blip:
        parts.append(f"Image: {blip}")
    return " ".join(parts)


def run_llm_segmentation(dataset, args) -> List[Dict]:
    """Run LLM-based topic segmentation on all conversations."""
    from lessmem.client import Client
    from lessmem.config import MemoryConfig
    from lessmem.prompts import TOPIC_SEGMENT_PROMPT
    from lessmem.data_utils import build_sessions_locomo

    config = MemoryConfig()

    def _segment_session(llm_client, messages):
        input_prompt, speakers = _format_segment_prompt(messages)
        if not input_prompt:
            return []
        result = llm_client.client_response(
            system_prompt=TOPIC_SEGMENT_PROMPT,
            input_prompt=input_prompt,
        )
        return _extract_topic_boundaries(result, len(messages))

    def _process_conversation(conv_data):
        conversation = conv_data['conversation']
        speaker_a = conversation['speaker_a']
        speaker_b = conversation['speaker_b']
        roles = f"{speaker_a}_{speaker_b}"

        llm_client = Client(api_key=config.openai_api_key,
                            base_url=config.base_url,
                            model=config.llm_model)

        logger.info(f"[{roles}] Start theme extraction")
        sessions = build_sessions_locomo(conversation)
        logger.info(f"[{roles}] Built {len(sessions)} session(s)")

        session_keys = sorted(sessions.keys())
        all_topics = []

        with ThreadPoolExecutor(max_workers=len(session_keys)) as executor:
            futures = {}
            for key in session_keys:
                messages = sessions[key]
                future = executor.submit(_segment_session, llm_client, messages)
                futures[future] = (key, messages)

            for future in concurrent.futures.as_completed(futures):
                key, messages = futures[future]
                try:
                    topics = future.result()
                    if not topics:
                        continue
                    timestamp = (messages[0].get('metadata', {})
                                 .get('dataset_timestamp', '') if messages else '')
                    for topic in topics:
                        start, end = topic['range']
                        all_topics.append({
                            'session_key': key,
                            'timestamp': timestamp,
                            'messages': messages[start:end + 1],
                        })
                except Exception as e:
                    logger.error(f"[{roles}] Session {key} segmentation failed: {e}")

        if not all_topics:
            logger.warning(f"[{roles}] No topics found")
            return {'roles': roles, 'themes': []}

        logger.info(f"[{roles}] Collected {len(all_topics)} topic(s)")

        themes_out = []
        for i, topic in enumerate(all_topics):
            themes_out.append({
                'theme_name': f"topic_{i}",
                'topic_count': 1,
                'topics': [{
                    'session_key': topic['session_key'],
                    'timestamp': topic['timestamp'],
                    'messages': [
                        {'role': m['role'], 'content': _format_message_content(m)}
                        for m in topic['messages']
                    ],
                }],
            })

        logger.info(f"[{roles}] Output {len(themes_out)} topic(s)")
        return {'roles': roles, 'themes': themes_out}

    all_results = []
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_conversation, conv): i
            for i, conv in enumerate(dataset)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                logger.info(f"Conversation {idx+1}/{len(dataset)} OK "
                            f"(done={completed}, fail={failed})")
            except Exception as e:
                failed += 1
                logger.error(f"Conversation {idx+1}/{len(dataset)} failed: "
                             f"{type(e).__name__}: {e}")
                logger.error(traceback.format_exc())

    return all_results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Topic segmentation (MinCut or LLM)")
    parser.add_argument("--data", required=True,
                        help="Path to LoCoMo JSON dataset")
    parser.add_argument("--output", required=True,
                        help="Output themes JSON file path")
    parser.add_argument("--mode", choices=["mincut", "llm"],
                        default="mincut",
                        help="Segmentation mode (default: mincut)")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                        help="Delta threshold for MinCut k selection (default: 1.01)")
    parser.add_argument("--max-workers", type=int, default=2000,
                        help="Max concurrent workers for session/conversation parallelism (default: 2000)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Only process first k conversations")
    args = parser.parse_args()

    print(f"Loading dataset: {args.data}")
    dataset = json.loads(Path(args.data).read_text(encoding="utf-8"))

    if args.top_k is not None:
        dataset = dataset[:args.top_k]
        print(f"Processing first {args.top_k} conversations only")

    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")

    if args.mode == "mincut":
        print(f"Delta threshold: {args.delta}")
        all_themes_data = run_mincut(dataset, args)
    else:
        print(f"Max workers: {args.max_workers}")
        all_themes_data = run_llm_segmentation(dataset, args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_themes_data, f, ensure_ascii=False, indent=2)

    total_topics = sum(len(r.get('themes', [])) for r in all_themes_data)
    print(f"\nDone! {total_topics} topics across {len(all_themes_data)} conversations")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
