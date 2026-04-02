import os
import logging
from typing import List, Dict, Any
from collections import defaultdict
import threading

import chromadb
from chromadb.config import Settings

from .models import Episode, SemanticMemory
from .client import Embedding
from .config import MemoryConfig

logger = logging.getLogger(__name__)


class ChromaEngine:
    def __init__(self, embedding_client: Embedding, config: MemoryConfig):
        self.embedding_client = embedding_client
        self.config = config
        self.persist_directory = config.chroma_persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection_prefix = config.chroma_collection_prefix
        self._collection_locks = defaultdict(threading.RLock)

    def _lock(self, name: str) -> threading.RLock:
        return self._collection_locks[name]

    def _get_or_create(self, name: str, meta: dict):
        with self._lock(name):
            try:
                return self.client.get_collection(name=name)
            except Exception:
                try:
                    return self.client.create_collection(name=name, metadata=meta)
                except Exception:
                    try:
                        self.client.delete_collection(name=name)
                    except Exception:
                        pass
                    return self.client.create_collection(name=name, metadata=meta)

    # ----- 写入 -----

    def add_episode(self, user_id: str, episode: Episode):
        col_name = f"{self.collection_prefix}_{user_id}_episodes"
        with self._lock(col_name):
            col = self._get_or_create(col_name, {"user_id": user_id, "type": "episodes"})
            if col.get(ids=[episode.episode_id])['ids']:
                return
            emb = self.embedding_client.embed_texts([episode.summary]).embeddings[0]
            col.add(
                ids=[episode.episode_id],
                documents=[episode.summary],
                metadatas=[{
                    "episode_id": episode.episode_id,
                    "user_id": episode.user_id,
                    "created_at": episode.created_at.isoformat(),
                    "timestamp": episode.timestamp,
                    "type": "episode",
                }],
                embeddings=[emb])

    def add_semantic_memory(self, user_id: str, memory: SemanticMemory):
        col_name = f"{self.collection_prefix}_{user_id}_semantic"
        with self._lock(col_name):
            col = self._get_or_create(col_name, {"user_id": user_id, "type": "semantic"})
            if col.get(ids=[memory.memory_id])['ids']:
                return
            metadata = {
                "memory_id": memory.memory_id,
                "user_id": memory.user_id,
                "created_at": memory.created_at.isoformat(),
                "timestamp": memory.timestamp,
                "source_episode": memory.source_episode,
                "revision_count": memory.revision_count,
                "type": "semantic",
            }
            if memory.updated_at:
                metadata["updated_at"] = memory.updated_at.isoformat()
            emb = self.embedding_client.embed_texts([memory.content]).embeddings[0]
            col.add(ids=[memory.memory_id], documents=[memory.content],
                    metadatas=[metadata], embeddings=[emb])

    def add_semantic_memories(self, memories: List[SemanticMemory]):
        for m in memories:
            self.add_semantic_memory(user_id=m.user_id, memory=m)

    # ----- 记忆更新相关 -----

    def get_all_semantic(self, user_id: str) -> List[Dict[str, Any]]:
        """获取某用户所有 semantic 记忆（含 embedding）。"""
        col_name = f"{self.collection_prefix}_{user_id}_semantic"
        with self._lock(col_name):
            col = self._get_or_create(col_name, {"user_id": user_id, "type": "semantic"})
            total = col.count()
            if total == 0:
                return []
            res = col.get(
                include=["documents", "metadatas", "embeddings"],
                limit=total,
            )
            entries = []
            for i in range(len(res["ids"])):
                entries.append({
                    "id": res["ids"][i],
                    "content": res["documents"][i],
                    "metadata": res["metadatas"][i],
                    "embedding": res["embeddings"][i] if res.get("embeddings") is not None else None,
                })
            return entries

    def update_semantic_content(self, user_id: str, memory_id: str,
                                new_content: str):
        """更新某条 semantic 记忆的文本内容和 embedding。"""
        col_name = f"{self.collection_prefix}_{user_id}_semantic"
        with self._lock(col_name):
            col = self._get_or_create(col_name, {"user_id": user_id, "type": "semantic"})
            existing = col.get(ids=[memory_id], include=["metadatas"])
            if not existing["ids"]:
                return
            meta = existing["metadatas"][0]
            from datetime import datetime
            meta["updated_at"] = datetime.now().isoformat()
            meta["revision_count"] = meta.get("revision_count", 1) + 1
            emb = self.embedding_client.embed_texts([new_content]).embeddings[0]
            col.update(
                ids=[memory_id],
                documents=[new_content],
                metadatas=[meta],
                embeddings=[emb],
            )

    def delete_semantic(self, user_id: str, memory_id: str):
        """删除某条 semantic 记忆。"""
        col_name = f"{self.collection_prefix}_{user_id}_semantic"
        with self._lock(col_name):
            col = self._get_or_create(col_name, {"user_id": user_id, "type": "semantic"})
            try:
                col.delete(ids=[memory_id])
            except Exception:
                pass

    def search_semantic_by_embedding(self, user_id: str,
                                     embedding: List[float],
                                     top_k: int = 20) -> List[Dict[str, Any]]:
        """用 embedding 向量检索 semantic 记忆，返回带距离的结果。"""
        col_name = f"{self.collection_prefix}_{user_id}_semantic"
        with self._lock(col_name):
            col = self._get_or_create(col_name, {"user_id": user_id, "type": "semantic"})
            if col.count() == 0:
                return []
            res = col.query(
                query_embeddings=[embedding],
                n_results=min(top_k, col.count()),
            )
            return [
                {"memory_id": res["ids"][0][i],
                 "content": res["documents"][0][i],
                 "distance": res["distances"][0][i],
                 "score": 1 - res["distances"][0][i],
                 "metadata": res["metadatas"][0][i]}
                for i in range(len(res["ids"][0]))
            ]

    # ----- 检索 -----

    def search_episodes(self, user_id: str, query: str,
                        top_k: int = 10) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        col_name = f"{self.collection_prefix}_{user_id}_episodes"
        with self._lock(col_name):
            col = self._get_or_create(col_name, {"user_id": user_id, "type": "episodes"})
            if col.count() == 0:
                return []
            qe = self.embedding_client.embed_text(query)
            res = col.query(query_embeddings=[qe],
                            n_results=min(top_k, col.count()))
            return [
                {"episode_id": res['ids'][0][i],
                 "content": res['documents'][0][i],
                 "distance": res['distances'][0][i],
                 "score": 1 - res['distances'][0][i],
                 "metadata": res['metadatas'][0][i],
                 "type": "episode"}
                for i in range(len(res['ids'][0]))
            ]

    def search_semantic_memories(self, user_id: str, query: str,
                                 top_k: int = 10) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        col_name = f"{self.collection_prefix}_{user_id}_semantic"
        with self._lock(col_name):
            col = self._get_or_create(col_name, {"user_id": user_id, "type": "semantic"})
            if col.count() == 0:
                return []
            qe = self.embedding_client.embed_text(query)
            res = col.query(query_embeddings=[qe],
                            n_results=min(top_k, col.count()))
            return [
                {"memory_id": res['ids'][0][i],
                 "content": res['documents'][0][i],
                 "distance": res['distances'][0][i],
                 "score": 1 - res['distances'][0][i],
                 "metadata": res['metadatas'][0][i],
                 "type": "semantic"}
                for i in range(len(res['ids'][0]))
            ]
