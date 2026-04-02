#!/usr/bin/env python3
"""
MinCut 参数扫描：基于已缓存的 embedding，快速评估不同参数组合下的主题数量。

使用 Normalized Cut DP 对每个 session 预计算 k=1..MAX_K 的最优分割，
然后扫描两种选 k 策略：
  1. delta threshold: 逐步增加 k，直到 Ncut(k)-Ncut(k-1) >= θ
  2. ratio: k = max(2, round(n * ratio))

用法：
  # 扫描打印表格
  python texttiling/sweep_mincut.py

  # 生成指定参数的 themes JSON
  python texttiling/sweep_mincut.py --save --method delta --param 1.05 \
      --output results/sweep_mincut_delta1.05/step2_segment/themes.json
"""
import argparse
import json
import os
import pickle
import sys
import time
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


def build_sim_matrix(embeddings: List[List[float]]) -> np.ndarray:
    E = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    sim = (E / norms) @ (E / norms).T
    np.fill_diagonal(sim, 0.0)
    return np.maximum(sim, 0.0)


def ncut_dp(sim: np.ndarray, max_k: int):
    """返回 {k: (ncut_val, segment_starts)} for k=1..max_k"""
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


def choose_k_delta(ncut_results: dict, delta_thr: float) -> int:
    """根据 delta 阈值选择 k。"""
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


def choose_k_ratio(n_utt: int, ratio: float, max_available_k: int) -> int:
    k = max(2, round(n_utt * ratio))
    return min(k, max_available_k)


MAX_K = 20


def compute_all_sessions(dataset, cache):
    """对所有 session 计算 MinCut DP，返回 session 列表。"""
    sessions_data = []
    for entry in dataset:
        conv = entry["conversation"]
        roles = f"{conv['speaker_a']}_{conv['speaker_b']}"
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
                sessions_data.append({
                    "roles": roles, "session_key": skey,
                    "timestamp": timestamp, "utts": utts, "texts": texts,
                    "n_utt": len(utts), "ncut_results": {1: (0.0, [])},
                })
                continue
            sim = build_sim_matrix(embeddings)
            mk = min(MAX_K, len(utts))
            ncut_results = ncut_dp(sim, mk)
            sessions_data.append({
                "roles": roles, "session_key": skey,
                "timestamp": timestamp, "utts": utts, "texts": texts,
                "n_utt": len(utts), "ncut_results": ncut_results,
            })
    return sessions_data


def count_topics_delta(sessions_data, delta_thr):
    total = 0
    per_roles = {}
    for s in sessions_data:
        k = choose_k_delta(s["ncut_results"], delta_thr)
        roles = s["roles"]
        per_roles[roles] = per_roles.get(roles, 0) + k
        total += k
    return total, per_roles


def count_topics_ratio(sessions_data, ratio):
    total = 0
    per_roles = {}
    for s in sessions_data:
        max_k = max(s["ncut_results"].keys())
        k = choose_k_ratio(s["n_utt"], ratio, max_k)
        roles = s["roles"]
        per_roles[roles] = per_roles.get(roles, 0) + k
        total += k
    return total, per_roles


def generate_themes_json(sessions_data, method, param, output_path):
    """生成 themes JSON。"""
    roles_themes = {}
    total = 0
    for s in sessions_data:
        roles = s["roles"]
        if roles not in roles_themes:
            roles_themes[roles] = {"roles": roles, "themes": [], "global_idx": 0}
        entry = roles_themes[roles]

        if method == "delta":
            k = choose_k_delta(s["ncut_results"], param)
        else:
            max_k = max(s["ncut_results"].keys())
            k = choose_k_ratio(s["n_utt"], param, max_k)

        if k in s["ncut_results"]:
            _, seg_starts = s["ncut_results"][k]
        else:
            seg_starts = []

        boundaries = sorted(set(seg_starts))
        segments = []
        prev = 0
        for b in boundaries:
            if b > prev:
                segments.append((prev, b))
            prev = b
        segments.append((prev, s["n_utt"]))

        for start, end in segments:
            messages = []
            for i in range(start, end):
                utt = s["utts"][i]
                messages.append({"role": utt["speaker"], "content": s["texts"][i]})
            if messages:
                entry["themes"].append({
                    "theme_name": f"topic_{entry['global_idx']}",
                    "topic_count": 1,
                    "topics": [{
                        "session_key": s["session_key"],
                        "timestamp": s["timestamp"],
                        "messages": messages,
                    }],
                })
                entry["global_idx"] += 1

    all_themes = []
    for roles in sorted(roles_themes):
        e = roles_themes[roles]
        all_themes.append({"roles": e["roles"], "themes": e["themes"]})
        total += e["global_idx"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_themes, f, ensure_ascii=False, indent=2)
    print(f"Saved {total} topics -> {output_path}")
    return total


def main():
    parser = argparse.ArgumentParser(description="MinCut 参数扫描")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--method", choices=["delta", "ratio"], default="delta")
    parser.add_argument("--param", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save-promising", action="store_true")
    args = parser.parse_args()

    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_json(DATASET_PATH)

    print(f"Loading embedding cache: {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache)} vectors loaded")

    print("\nComputing MinCut DP for all sessions...")
    t0 = time.time()
    sessions_data = compute_all_sessions(dataset, cache)
    elapsed = time.time() - t0
    print(f"  Done: {len(sessions_data)} sessions in {elapsed:.1f}s")

    if args.save:
        tag = f"mincut_{args.method}{args.param}"
        out = args.output or f"results/{tag}/step2_segment/themes.json"
        print(f"\nGenerating themes: method={args.method}, param={args.param}")
        generate_themes_json(sessions_data, args.method, args.param, out)
        return

    # ── Delta sweep ──
    print(f"\n{'='*80}")
    print("Method: Delta threshold (Ncut(k)-Ncut(k-1) < θ)")
    print(f"{'='*80}")
    delta_values = [round(x * 0.05, 2) for x in range(1, 41)]
    print(f"\n{'delta':>8} {'topics':>8} {'status':>15}")
    print("-" * 35)
    delta_promising = []
    for d in delta_values:
        total, per_roles = count_topics_delta(sessions_data, d)
        mark = ""
        if 800 <= total <= 1600:
            mark = " *** 目标范围"
            delta_promising.append((d, total, per_roles))
        elif total < 800:
            mark = " (过少)"
        print(f"{d:>8.2f} {total:>8d}{mark}")

    # ── Ratio sweep ──
    print(f"\n{'='*80}")
    print("Method: Ratio (k = max(2, round(n * ratio)))")
    print(f"{'='*80}")
    ratio_values = [round(x * 0.02, 2) for x in range(1, 51)]
    print(f"\n{'ratio':>8} {'topics':>8} {'status':>15}")
    print("-" * 35)
    ratio_promising = []
    for r in ratio_values:
        total, per_roles = count_topics_ratio(sessions_data, r)
        mark = ""
        if 800 <= total <= 1600:
            mark = " *** 目标范围"
            ratio_promising.append((r, total, per_roles))
        elif total < 800:
            mark = " (过少)"
        print(f"{r:>8.2f} {total:>8d}{mark}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("目标范围内的参数组合 (800-1600 themes):")
    print(f"{'='*80}")
    if delta_promising:
        print("\n  Delta threshold:")
        for d, total, per_roles in delta_promising:
            print(f"    delta={d:.2f}  topics={total}")
    if ratio_promising:
        print("\n  Ratio:")
        for r, total, per_roles in ratio_promising:
            print(f"    ratio={r:.2f}  topics={total}")

    if args.save_promising:
        print(f"\n自动保存目标范围内的参数组合...")
        for d, total, _ in delta_promising:
            tag = f"mincut_delta{d}"
            out = f"results/{tag}/step2_segment/themes.json"
            generate_themes_json(sessions_data, "delta", d, out)
        for r, total, _ in ratio_promising:
            tag = f"mincut_ratio{r}"
            out = f"results/{tag}/step2_segment/themes.json"
            generate_themes_json(sessions_data, "ratio", r, out)


if __name__ == "__main__":
    main()
