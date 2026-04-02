#!/usr/bin/env python3
"""
TextTiling 参数扫描：基于已缓存的 embedding，快速评估不同参数组合下的主题数量。

可调参数：
  threshold:      余弦相似度阈值，低于阈值则切分（越低 → 越少主题）
  window_size:    滑动窗口大小，将窗口内 embedding 平均后再比较（越大 → 越平滑 → 越少噪声切分）
  min_topic_size: 最小主题长度，短于此的主题合并到相邻主题

用法：
  # 仅扫描打印表格
  python texttiling/sweep_themes.py

  # 对指定参数生成 themes JSON
  python texttiling/sweep_themes.py --save --threshold 0.30 --window 3 --min-size 3 \
      --output results/sweep_t0.30_w3_m3/step2_segment/themes.json
"""
import argparse
import json
import os
import pickle
import sys
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = "/root/autodl-tmp/LessMem"
DATASET_PATH = os.path.join(ROOT, "dataset", "locomo10_compressed_r0.7.json")
CACHE_PATH = os.path.join(ROOT, "texttiling", "embedding_cache.pkl")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_utterance_content(utt: dict) -> str:
    text = utt.get("text", "")
    blip = utt.get("blip_caption", "")
    if blip:
        text += f" Image: {blip}"
    return text


def cosine_sim(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def window_cosine_sims(embeddings: List[List[float]], window: int) -> List[float]:
    """计算相邻位置间的余弦相似度，使用滑动窗口平均。"""
    n = len(embeddings)
    if n <= 1:
        return []
    embs = np.array(embeddings, dtype=np.float64)
    half = window // 2

    def avg_emb(center: int) -> np.ndarray:
        lo = max(0, center - half)
        hi = min(n, center + half + 1)
        return embs[lo:hi].mean(axis=0)

    sims = []
    for i in range(n - 1):
        a = avg_emb(i)
        b = avg_emb(i + 1)
        d = np.linalg.norm(a) * np.linalg.norm(b)
        sims.append(float(np.dot(a, b) / d) if d > 0 else 0.0)
    return sims


def segment(sims: List[float], threshold: float,
            min_topic_size: int) -> List[int]:
    """根据相似度序列、阈值和最小主题长度，返回每个 utterance 的 topic label。"""
    n = len(sims) + 1
    if n <= 1:
        return [0] * n

    labels = [0] * n
    topic_id = 0
    for i in range(len(sims)):
        if sims[i] < threshold:
            topic_id += 1
        labels[i + 1] = topic_id

    if min_topic_size <= 1:
        return labels

    changed = True
    while changed:
        changed = False
        topic_sizes: Dict[int, int] = {}
        for lbl in labels:
            topic_sizes[lbl] = topic_sizes.get(lbl, 0) + 1
        for lbl, sz in topic_sizes.items():
            if sz < min_topic_size:
                first_idx = labels.index(lbl)
                if first_idx > 0:
                    merge_to = labels[first_idx - 1]
                else:
                    last_idx = len(labels) - 1 - labels[::-1].index(lbl)
                    merge_to = labels[last_idx + 1] if last_idx + 1 < n else lbl
                if merge_to != lbl:
                    labels = [merge_to if l == lbl else l for l in labels]
                    changed = True

    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    return [remap[l] for l in labels]


def count_topics_for_dataset(
    dataset, cache: Dict[str, List[float]],
    threshold: float, window: int, min_topic_size: int,
) -> Tuple[int, Dict[str, int]]:
    """返回 (总主题数, {roles: 主题数})。"""
    total = 0
    per_roles = {}
    for entry in dataset:
        conv = entry["conversation"]
        roles = f"{conv['speaker_a']}_{conv['speaker_b']}"
        conv_topics = 0
        idx = 1
        while f"session_{idx}" in conv:
            utts = conv[f"session_{idx}"]
            idx += 1
            if not utts:
                continue
            texts = [format_utterance_content(u) for u in utts]
            embeddings = [cache[t] for t in texts if t in cache]
            if len(embeddings) != len(texts):
                miss = sum(1 for t in texts if t not in cache)
                print(f"  WARNING: {roles}/session_{idx-1}: {miss} uncached, skipping")
                conv_topics += 1
                continue
            if window > 1:
                sims = window_cosine_sims(embeddings, window)
            else:
                sims = [cosine_sim(embeddings[i], embeddings[i+1])
                        for i in range(len(embeddings)-1)]
            labels = segment(sims, threshold, min_topic_size)
            n_topics = max(labels) + 1 if labels else 1
            conv_topics += n_topics
        per_roles[roles] = conv_topics
        total += conv_topics
    return total, per_roles


def generate_themes_json(
    dataset, cache: Dict[str, List[float]],
    threshold: float, window: int, min_topic_size: int,
    output_path: str,
):
    """生成 themes JSON 文件。"""
    all_themes = []
    total = 0
    for entry in dataset:
        conv = entry["conversation"]
        roles = f"{conv['speaker_a']}_{conv['speaker_b']}"
        themes_list = []
        global_idx = 0
        idx = 1
        while f"session_{idx}" in conv:
            skey = f"session_{idx}"
            utts = conv[skey]
            timestamp = conv.get(f"{skey}_date_time", "")
            idx += 1
            if not utts:
                continue
            texts = [format_utterance_content(u) for u in utts]
            embeddings = [cache.get(t) for t in texts]
            if any(e is None for e in embeddings):
                themes_list.append({
                    "theme_name": f"topic_{global_idx}",
                    "topic_count": 1,
                    "topics": [{"session_key": skey, "timestamp": timestamp,
                                "messages": [{"role": u["speaker"], "content": t}
                                             for u, t in zip(utts, texts)]}],
                })
                global_idx += 1
                continue
            if window > 1:
                sims = window_cosine_sims(embeddings, window)
            else:
                sims = [cosine_sim(embeddings[i], embeddings[i+1])
                        for i in range(len(embeddings)-1)]
            labels = segment(sims, threshold, min_topic_size)
            n_topics = max(labels) + 1 if labels else 1
            for local_t in range(n_topics):
                messages = []
                for i, utt in enumerate(utts):
                    if labels[i] == local_t:
                        messages.append({"role": utt["speaker"], "content": texts[i]})
                if messages:
                    themes_list.append({
                        "theme_name": f"topic_{global_idx}",
                        "topic_count": 1,
                        "topics": [{"session_key": skey, "timestamp": timestamp,
                                    "messages": messages}],
                    })
                    global_idx += 1
        all_themes.append({"roles": roles, "themes": themes_list})
        total += global_idx
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_themes, f, ensure_ascii=False, indent=2)
    print(f"Saved {total} topics -> {output_path}")
    return total


def main():
    parser = argparse.ArgumentParser(description="TextTiling 参数扫描")
    parser.add_argument("--save", action="store_true",
                        help="生成指定参数的 themes JSON")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save-promising", action="store_true",
                        help="自动保存主题数在 1200-1600 范围的参数组合")
    args = parser.parse_args()

    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_json(DATASET_PATH)

    print(f"Loading embedding cache: {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache)} vectors loaded")

    if args.save:
        t, w, m = args.threshold, args.window, args.min_size
        out = args.output or f"results/sweep_t{t}_w{w}_m{m}/step2_segment/themes.json"
        print(f"\nGenerating themes: threshold={t}, window={w}, min_size={m}")
        generate_themes_json(dataset, cache, t, w, m, out)
        return

    thresholds = [0.20, 0.25, 0.30, 0.32, 0.34, 0.35, 0.36, 0.38,
                  0.40, 0.42, 0.45, 0.50, 0.55, 0.60]
    windows = [1, 3, 5]
    min_sizes = [1, 3, 5]

    print(f"\n{'='*80}")
    print(f"参数扫描: {len(thresholds)} thresholds × {len(windows)} windows × {len(min_sizes)} min_sizes"
          f" = {len(thresholds)*len(windows)*len(min_sizes)} combinations")
    print(f"{'='*80}")

    results = []
    for thr, win, msz in product(thresholds, windows, min_sizes):
        total, per_roles = count_topics_for_dataset(dataset, cache, thr, win, msz)
        results.append((thr, win, msz, total, per_roles))

    print(f"\n{'threshold':>10} {'window':>7} {'min_sz':>7} {'topics':>8}  {'status':>12}")
    print("-" * 55)
    promising = []
    for thr, win, msz, total, per_roles in sorted(results, key=lambda x: x[3]):
        mark = ""
        if 1200 <= total <= 1600:
            mark = " *** 目标范围"
            promising.append((thr, win, msz, total, per_roles))
        elif total < 1200:
            mark = " (过少)"
        elif total > 2000:
            mark = ""
        print(f"{thr:>10.2f} {win:>7d} {msz:>7d} {total:>8d}{mark}")

    if promising:
        print(f"\n{'='*80}")
        print(f"目标范围内的参数组合 (1200-1600 themes):")
        print(f"{'='*80}")
        print(f"{'threshold':>10} {'window':>7} {'min_sz':>7} {'topics':>8}")
        print("-" * 40)
        for thr, win, msz, total, per_roles in promising:
            print(f"{thr:>10.2f} {win:>7d} {msz:>7d} {total:>8d}")
            for roles in sorted(per_roles):
                print(f"{'':>36} {roles}: {per_roles[roles]}")

    if args.save_promising and promising:
        print(f"\n自动保存目标范围内的参数组合...")
        for thr, win, msz, total, _ in promising:
            tag = f"sweep_t{thr}_w{win}_m{msz}"
            out = f"results/{tag}/step2_segment/themes.json"
            generate_themes_json(dataset, cache, thr, win, msz, out)


if __name__ == "__main__":
    main()
