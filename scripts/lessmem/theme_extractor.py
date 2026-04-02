import re
import logging
from typing import Dict, List, Tuple

from .client import Client
from .prompts import THEME_EXTRACT_PROMPT, THEME_BATCH_EXTRACT_PROMPT

logger = logging.getLogger(__name__)


class ThemeExtractor:
    """批量提取 episodic summary + semantic memories。"""

    def __init__(self, llm_client: Client):
        self.llm_client = llm_client

    @staticmethod
    def _format_single_theme(theme_group: Dict) -> str:
        parts = []
        for i, topic in enumerate(theme_group['topics']):
            timestamp = topic.get('timestamp', 'Unknown')
            parts.append(f"\n--- Conversation Segment {i + 1} ---")
            parts.append(f"Current Time: {timestamp}")
            for msg in topic['messages']:
                role = msg.get('role', 'Unknown')
                content = msg.get('content', '')
                if content:
                    parts.append(f"{role}: {content}")
        return "\n".join(parts)

    @staticmethod
    def _format_batch_themes(theme_groups: List[Dict],
                             start_id: int = 0) -> str:
        blocks = []
        for idx, tg in enumerate(theme_groups):
            topic_id = start_id + idx
            inner = []
            for i, topic in enumerate(tg['topics']):
                timestamp = topic.get('timestamp', 'Unknown')
                if i > 0:
                    inner.append(f"--- Segment {i + 1} ---")
                inner.append(f"Current Time: {timestamp}")
                for msg in topic['messages']:
                    role = msg.get('role', 'Unknown')
                    content = msg.get('content', '')
                    if content:
                        inner.append(f"{role}: {content}")
            blocks.append(
                f'<topic id="{topic_id}">\n' + "\n".join(inner) + "\n</topic>")
        return "\n\n".join(blocks)

    def extract_memories(self, theme_group: Dict) -> Tuple[str, List[str]]:
        input_text = self._format_single_theme(theme_group)
        result = self.llm_client.client_response(
            system_prompt=THEME_EXTRACT_PROMPT, input_prompt=input_text)
        return self._parse_episodic(result), self._parse_semantics(result)

    def extract_memories_batch(self, theme_groups: List[Dict],
                               start_id: int = 0) -> List[Tuple[int, str, List[str]]]:
        input_text = self._format_batch_themes(theme_groups, start_id)
        result = self.llm_client.client_response(
            system_prompt=THEME_BATCH_EXTRACT_PROMPT, input_prompt=input_text)
        return self._parse_batch_result(result, theme_groups, start_id)

    def _parse_batch_result(self, result: str, theme_groups: List[Dict],
                            start_id: int) -> List[Tuple[int, str, List[str]]]:
        outputs = []
        pattern = r'<result\s+id="(\d+)">(.*?)</result>'
        matches = re.findall(pattern, result, re.DOTALL)

        parsed_ids = set()
        for rid_str, content in matches:
            rid = int(rid_str)
            parsed_ids.add(rid)
            outputs.append((rid, self._parse_episodic(content),
                            self._parse_semantics(content)))

        missing = set(range(start_id, start_id + len(theme_groups))) - parsed_ids
        if missing:
            logger.warning(f"Batch missing topic ids: {sorted(missing)}, retrying")
            for mid in sorted(missing):
                try:
                    fallback = self.extract_memories_batch(
                        [theme_groups[mid - start_id]], start_id=mid)
                    outputs.extend(fallback)
                except Exception as e:
                    logger.error(f"Fallback for topic #{mid} failed: {e}")

        outputs.sort(key=lambda x: x[0])
        return outputs

    @staticmethod
    def _parse_episodic(result: str) -> str:
        match = re.search(
            r'<episodic_summary>(.*?)</episod(?:ic)?_summary>', result, re.DOTALL)
        if not match:
            match = re.search(
                r'<episodic_summary>(.*?)(?=<semantic_memories>|$)', result, re.DOTALL)
        if not match:
            match = re.search(
                r'Summary:\s*(.*?)(?=<semantic_memories>|$)', result, re.DOTALL)
        text = match.group(1).strip() if match else result.strip()
        if text.startswith('Summary:'):
            text = text[len('Summary:'):].strip()
        return re.sub(r'</?[a-zA-Z_][^>]*>', '', text).strip()

    @staticmethod
    def _parse_semantics(result: str) -> List[str]:
        match = re.search(
            r'<semantic_memories>(.*?)</semantic_memories>', result, re.DOTALL)
        if not match:
            match = re.search(r'<semantic_memories>(.*?)$', result, re.DOTALL)
        if not match:
            return []
        content = re.sub(r'</?speaker[^>]*>', '', match.group(1))
        memories = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if line and line != '-':
                cleaned = re.sub(r'^[-*\d.]+\s*', '', line).strip()
                cleaned = re.sub(r'</?[a-zA-Z_][^>]*>', '', cleaned).strip()
                if cleaned:
                    memories.append(cleaned)
        return memories
