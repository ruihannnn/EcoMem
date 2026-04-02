import os
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = "text-embedding-3-small"

    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_collection_prefix: str = "lessmem"

    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    base_url: str = os.getenv("BASE_URL")

    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.llm_api_key:
            self.llm_api_key = self.openai_api_key
