import time
import logging
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


# ======================================================================
# LLM Client (带 token 统计)
# ======================================================================

class Client:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = ""):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.max_retries = 3
        self.retry_delay = 1.0

        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._call_count = 0
        self._usage_lock = threading.Lock()

    def client_response(self, system_prompt: str, input_prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_prompt},
                    ],
                    max_tokens=16000,
                    temperature=0.1,
                    stream=False,
                )
                if hasattr(response, "usage") and response.usage:
                    u = response.usage
                    with self._usage_lock:
                        self._total_usage["prompt_tokens"] += u.prompt_tokens
                        self._total_usage["completion_tokens"] += u.completion_tokens
                        self._total_usage["total_tokens"] += u.total_tokens
                        self._call_count += 1
                else:
                    with self._usage_lock:
                        self._call_count += 1
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM call failed ({attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise


# ======================================================================
# Embedding Client
# ======================================================================

@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    usage: Dict[str, Any]
    model: str
    response_time: float


class Embedding:
    def __init__(self, api_key: str, base_url: Optional[str] = "",
                 model: str = "text-embedding-3-small"):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = 3
        self.retry_delay = 1.0
        self.batch_size = 100

    def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(embeddings=[], usage={},
                                     model=self.model, response_time=0.0)
        start = time.time()
        all_embeddings = []
        total_usage = {"prompt_tokens": 0, "total_tokens": 0}

        for i in range(0, len(texts), self.batch_size):
            batch = [str(t).replace("\n", " ") for t in texts[i:i + self.batch_size]]
            for attempt in range(self.max_retries):
                try:
                    resp = self.client.embeddings.create(
                        model=self.model, input=batch, timeout=30.0)
                    all_embeddings.extend(d.embedding for d in resp.data)
                    if resp.usage:
                        total_usage["prompt_tokens"] += getattr(resp.usage, "prompt_tokens", 0)
                        total_usage["total_tokens"] += getattr(resp.usage, "total_tokens", 0)
                    break
                except Exception as e:
                    logger.warning(f"Embedding call failed ({attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                    else:
                        raise

        return EmbeddingResponse(embeddings=all_embeddings, usage=total_usage,
                                 model=self.model, response_time=time.time() - start)

    def embed_text(self, text: str) -> List[float]:
        resp = self.embed_texts([text])
        return resp.embeddings[0] if resp.embeddings else []
