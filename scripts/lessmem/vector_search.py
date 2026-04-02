import json
import re
import logging
from typing import List

from .prompts import ANSWER_PROMPT, ROUTER_PROMPT
from .client import Client
from .config import MemoryConfig
from .chroma import ChromaEngine

logger = logging.getLogger(__name__)

_EP_K_RANGE = (1, 10)
_SEM_K_RANGE = (1, 80)
_FALLBACK = {"ep_k": 1, "sem_k": 20}
_JSON_RE = re.compile(r'\{[^{}]*\}')


class VectorSearch:
    def __init__(self, llm_client: Client, config: MemoryConfig,
                 backend: ChromaEngine):
        self.llm_client = llm_client
        self.config = config
        self.backend = backend

    def _route(self, question: str) -> dict:
        try:
            raw = self.llm_client.client_response(
                system_prompt=ROUTER_PROMPT,
                input_prompt=f"Q: {question}")
            m = _JSON_RE.search(raw)
            if not m:
                raise ValueError(f"No JSON found in: {raw[:200]}")
            result = json.loads(m.group())
            ep_k = max(_EP_K_RANGE[0], min(_EP_K_RANGE[1], int(result["ep_k"])))
            sem_k = max(_SEM_K_RANGE[0], min(_SEM_K_RANGE[1], int(result["sem_k"])))
            cfg = {"ep_k": ep_k, "sem_k": sem_k}
        except Exception as e:
            cfg = _FALLBACK
            logger.warning(f"Router failed ({e}), using fallback values")
        return cfg

    def answer(self, question: str, speakers: List[str],
               use_semantic: bool = False,
               ep_k: int = 1, sem_k: int = 20,
               adaptive: bool = False,
               sem_timestamp: bool = True,
               no_episodic: bool = False) -> str:
        if adaptive:
            cfg = self._route(question)
            ep_k, sem_k = cfg["ep_k"], cfg["sem_k"]
        if no_episodic:
            ep_k = 0

        roles = f"{speakers[0]}_{speakers[1]}"

        episode_results = self.backend.search_episodes(
            user_id=roles, query=question, top_k=ep_k)
        episode_prompt = "\n".join(
            r['content'] for r in episode_results if r['content'])

        answer_prompt = (f"Question: {question}\n\n"
                         f"Contents:\nEpisodes:\n{episode_prompt}\n")

        if use_semantic:
            semantic_parts = []
            sem_results = self.backend.search_semantic_memories(
                user_id=roles, query=question, top_k=sem_k)
            for r in sem_results:
                if r.get('content'):
                    ts = r.get('metadata', {}).get('timestamp', '') if sem_timestamp else ''
                    prefix = f"[{ts}]\n" if ts else ""
                    semantic_parts.append(f"{prefix}{r['content']}")
            if semantic_parts:
                answer_prompt += f"\nSemantic Memories:\n" + "\n".join(semantic_parts) + "\n"

        response = self.llm_client.client_response(
            system_prompt=ANSWER_PROMPT, input_prompt=answer_prompt)
        print(f'question: {question} has been answered')
        return response, answer_prompt
