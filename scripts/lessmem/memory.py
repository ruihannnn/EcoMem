import logging
import threading
import concurrent.futures
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from .client import Client, Embedding
from .config import MemoryConfig
from .models import Episode, SemanticMemory
from .vector_search import VectorSearch
from .data_utils import normalize_timestamp

logger = logging.getLogger(__name__)


class LessMem:
    _GLOBAL_DB_LOCK = threading.Lock()
    _SHARED_BACKEND = None

    def __init__(self):
        self.config = MemoryConfig()
        self.llm_client = Client(
            api_key=self.config.llm_api_key,
            base_url=self.config.base_url,
            model=self.config.llm_model)
        self.embedding_client = Embedding(
            api_key=self.config.openai_api_key,
            base_url=self.config.base_url,
            model=self.config.embedding_model)
        self.vector_search = VectorSearch(
            llm_client=self.llm_client,
            config=self.config,
            backend=self.backend)

    @property
    def backend(self):
        if LessMem._SHARED_BACKEND is None:
            with LessMem._GLOBAL_DB_LOCK:
                if LessMem._SHARED_BACKEND is None:
                    from .chroma import ChromaEngine
                    LessMem._SHARED_BACKEND = ChromaEngine(
                        embedding_client=self.embedding_client,
                        config=self.config)
        return LessMem._SHARED_BACKEND

    # ------------------------------------------------------------------
    # 创建记忆对象（时间戳在此处标准化为 ISO 格式）
    # ------------------------------------------------------------------

    def create_episode_memory(self, roles: str, time_stamp: str,
                              topic: Dict) -> Episode:
        return Episode(
            user_id=roles,
            timestamp=normalize_timestamp(time_stamp),
            summary=topic['summary'])

    def create_semantic_memory(self, roles: str,
                               semantic_memories: List[str],
                               time_stamp: str,
                               episode_id: str) -> List[SemanticMemory]:
        ts = normalize_timestamp(time_stamp)
        return [
            SemanticMemory(content=c, user_id=roles,
                           timestamp=ts, source_episode=episode_id)
            for c in semantic_memories
        ]

    # ==================================================================
    # LLM 提取 episodic + semantic（多线程）
    # ==================================================================

    def extract_all_from_themes(
        self,
        theme_groups: List[Dict],
        roles: str,
        topics_per_batch: int = 5,
        max_workers: int = 100,
    ) -> Tuple[List[Dict], Dict]:
        from .theme_extractor import ThemeExtractor

        theme_extractor = ThemeExtractor(llm_client=self.llm_client)
        batches = [
            (i, theme_groups[i:i + topics_per_batch])
            for i in range(0, len(theme_groups), topics_per_batch)
        ]
        logger.info(f"[{roles}] Extract: {len(theme_groups)} theme(s), "
                    f"{len(batches)} batch(es)")

        with self.llm_client._usage_lock:
            usage_before = dict(self.llm_client._total_usage)
            calls_before = self.llm_client._call_count

        records: List[Dict] = []
        lock = threading.Lock()

        def _do_batch(batch_idx: int, start_id: int, batch_tgs: List[Dict]):
            extracted = theme_extractor.extract_memories_batch(batch_tgs, start_id)
            batch_records = []
            for topic_id, ep_summary, sem_list in extracted:
                tg = batch_tgs[topic_id - start_id]
                timestamp = tg['topics'][0]['timestamp'] if tg['topics'] else ''
                batch_records.append({
                    "roles": roles,
                    "theme_name": tg.get('theme_name', ''),
                    "timestamp": timestamp,
                    "episodic_summary": ep_summary,
                    "semantic_memories": sem_list,
                })
            with lock:
                records.extend(batch_records)

        with ThreadPoolExecutor(
                max_workers=min(len(batches), max_workers)) as executor:
            futures = {
                executor.submit(_do_batch, bi, sid, btg): bi
                for bi, (sid, btg) in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(futures):
                bi = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[{roles}] Batch#{bi} failed: {e}")

        with self.llm_client._usage_lock:
            usage = {
                'prompt_tokens': self.llm_client._total_usage['prompt_tokens'] - usage_before['prompt_tokens'],
                'completion_tokens': self.llm_client._total_usage['completion_tokens'] - usage_before['completion_tokens'],
                'total_tokens': self.llm_client._total_usage['total_tokens'] - usage_before['total_tokens'],
                'api_calls': self.llm_client._call_count - calls_before,
            }

        logger.info(f"[{roles}] Extract done: {len(records)} records, "
                    f"tokens={usage['total_tokens']:,}")
        return records, usage

    # ==================================================================
    # 存入 ChromaDB（多线程）
    # ==================================================================

    def store_records_to_chroma(self, records: List[Dict],
                                max_workers: int = 1000) -> Tuple[int, int]:
        total = len(records)
        logger.info(f"Store: {total} record(s) -> ChromaDB")

        ep_count, sem_count = 0, 0
        counter_lock = threading.Lock()

        def _store_one(rec: Dict):
            nonlocal ep_count, sem_count
            roles = rec['roles']
            timestamp = rec['timestamp']
            episode = self.create_episode_memory(
                roles=roles, time_stamp=timestamp,
                topic={'summary': rec['episodic_summary']})
            semantics = self.create_semantic_memory(
                roles=roles, semantic_memories=rec['semantic_memories'],
                time_stamp=timestamp, episode_id=episode.episode_id)
            self.backend.add_episode(user_id=roles, episode=episode)
            self.backend.add_semantic_memories(semantics)
            with counter_lock:
                ep_count += 1
                sem_count += len(semantics)

        pbar = tqdm(total=total, desc="Store", unit="rec")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_store_one, r): i
                       for i, r in enumerate(records)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Store failed: {e}")
                pbar.update(1)
        pbar.close()

        logger.info(f"Store done: {ep_count} episodes, {sem_count} semantics")
        return ep_count, sem_count

    # ==================================================================
    # 语义记忆离线更新
    # ==================================================================

    def update_semantic_memories(
        self,
        user_id: str,
        score_threshold: float = 0.8,
        top_k: int = 20,
        keep_top_n: int = 10,
        max_workers: int = 5,
    ) -> Dict:
        """对某用户的 semantic 记忆执行离线更新（合并 / 删除冗余）。

        流程：
          1. 取出所有 semantic 记忆及其 embedding
          2. 对每条记忆，用 embedding 检索最相似的 top_k 条作为候选
          3. 对相似度 >= score_threshold 的候选对，调用 LLM 判定
             update / delete / ignore

        返回包含 updated/deleted/skipped 计数以及 token 消耗统计。
        """
        from .prompts import UPDATE_PROMPT

        all_entries = self.backend.get_all_semantic(user_id)
        if not all_entries:
            logger.info(f"[{user_id}] No semantic memories, skipping update")
            return {"updated": 0, "deleted": 0, "skipped": 0,
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "api_calls": 0}

        logger.info(f"[{user_id}] Update: {len(all_entries)} semantic memories, "
                    f"threshold={score_threshold}")

        with self.llm_client._usage_lock:
            usage_before = dict(self.llm_client._total_usage)
            calls_before = self.llm_client._call_count

        id_to_entry = {e["id"]: e for e in all_entries}

        # --- 1. 为每条记忆构建 update_queue ---
        update_queues: Dict[str, list] = {}
        for entry in all_entries:
            eid = entry["id"]
            emb = entry.get("embedding")
            if emb is None:
                continue
            hits = self.backend.search_semantic_by_embedding(
                user_id, emb, top_k=top_k)
            candidates = [
                {"id": h["memory_id"], "score": h["score"]}
                for h in hits if h["memory_id"] != eid
            ]
            candidates.sort(key=lambda x: x["score"], reverse=True)
            update_queues[eid] = candidates[:keep_top_n]

        # --- 2. 反向索引：谁的 update_queue 里包含当前 entry ---
        updated, deleted, skipped = 0, 0, 0
        lock = threading.Lock()

        def _process_entry(entry):
            nonlocal updated, deleted, skipped
            eid = entry["id"]
            target_content = entry["content"]

            candidate_sources = []
            for other_id, queue in update_queues.items():
                if other_id == eid:
                    continue
                for c in queue:
                    if c["id"] == eid and c["score"] >= score_threshold:
                        if other_id in id_to_entry:
                            candidate_sources.append(id_to_entry[other_id])
                        break

            if not candidate_sources:
                with lock:
                    skipped += 1
                return

            candidate_texts = [c["content"] for c in candidate_sources]
            user_prompt = (
                f"Target memory: {target_content}\n"
                f"Candidate memories:\n"
                + "\n".join(f"- {m}" for m in candidate_texts)
            )
            try:
                raw = self.llm_client.client_response(UPDATE_PROMPT, user_prompt)
                import json as _json, re as _re
                _m = _re.search(r"\{[\s\S]*\}", raw)
                result = _json.loads(_m.group()) if _m else {"action": "ignore"}
            except Exception as e:
                logger.warning(f"[{user_id}] Update LLM error for {eid}: {e}")
                result = {"action": "ignore"}

            action = result.get("action", "ignore")
            if action == "delete":
                self.backend.delete_semantic(user_id, eid)
                with lock:
                    deleted += 1
                logger.debug(f"[{user_id}] Deleted {eid}")
            elif action == "update":
                new_mem = result.get("new_memory", target_content)
                self.backend.update_semantic_content(user_id, eid, new_mem)
                with lock:
                    updated += 1
                logger.debug(f"[{user_id}] Updated {eid}")
            else:
                with lock:
                    skipped += 1

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers) as executor:
            executor.map(_process_entry, all_entries)

        with self.llm_client._usage_lock:
            usage = {
                'prompt_tokens': self.llm_client._total_usage['prompt_tokens'] - usage_before['prompt_tokens'],
                'completion_tokens': self.llm_client._total_usage['completion_tokens'] - usage_before['completion_tokens'],
                'total_tokens': self.llm_client._total_usage['total_tokens'] - usage_before['total_tokens'],
                'api_calls': self.llm_client._call_count - calls_before,
            }

        stats = {
            "updated": updated, "deleted": deleted, "skipped": skipped,
            **usage,
        }
        logger.info(f"[{user_id}] Update done: updated={updated}, deleted={deleted}, "
                    f"skipped={skipped}, api_calls={usage['api_calls']}, "
                    f"tokens={usage['total_tokens']:,}")
        return stats

    # ------------------------------------------------------------------
    # 回答问题
    # ------------------------------------------------------------------

    def answer(self, question: str, speakers: List[str],
               use_semantic: bool = False,
               ep_k: int = 1, sem_k: int = 20,
               adaptive: bool = False,
               sem_timestamp: bool = True,
               no_episodic: bool = False):
        return self.vector_search.answer(
            question=question, speakers=speakers,
            use_semantic=use_semantic,
            ep_k=ep_k, sem_k=sem_k,
            adaptive=adaptive,
            sem_timestamp=sem_timestamp,
            no_episodic=no_episodic)
