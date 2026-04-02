from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import uuid


@dataclass
class Episode:
    summary: str
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    timestamp: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class SemanticMemory:
    content: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    timestamp: str = ""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_episode: str = field(default_factory=str)
    updated_at: Optional[datetime] = None
    revision_count: int = 1
