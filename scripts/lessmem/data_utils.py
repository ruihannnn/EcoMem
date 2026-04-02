import re
from datetime import datetime
from typing import Any, Dict, List

# ======================================================================
# 时间戳解析与标准化
# ======================================================================

_TS_PATTERN = re.compile(
    r'(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})',
    re.IGNORECASE,
)
_MONTHS = {m: i for i, m in enumerate([
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
], 1)}


def normalize_timestamp(ts: str) -> str:
    """将自然语言时间戳标准化为 ISO 格式。

    '12:48 am on 1 February, 2023' -> '2023-02-01 00:48'
    已经是 ISO 格式或无法识别的原样返回。
    """
    if not ts:
        return ''
    m = _TS_PATTERN.search(ts)
    if not m:
        return ts
    hour, minute, ampm, day, month_name, year = (
        int(m.group(1)), int(m.group(2)), m.group(3).lower(),
        int(m.group(4)), m.group(5).lower(), int(m.group(6)),
    )
    if ampm == 'am' and hour == 12:
        hour = 0
    elif ampm == 'pm' and hour != 12:
        hour += 12
    mon = _MONTHS.get(month_name, 1)
    return f"{year}-{mon:02d}-{day:02d} {hour:02d}:{minute:02d}"


def parse_timestamp(value: str) -> datetime:
    """解析数据集中的时间戳为 datetime 对象。"""
    value = " ".join(value.split())
    if " on " not in value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now()
    time_part, date_part = value.split(" on ")
    time_part = time_part.lower().strip()
    hour, minute = 0, 0
    is_pm = "pm" in time_part
    time_part = time_part.replace("pm", "").replace("am", "").strip()
    if ":" in time_part:
        hour_str, minute_str = time_part.split(":", 1)
        hour, minute = int(hour_str), int(minute_str)
    else:
        hour = int(time_part)
    if is_pm and hour != 12:
        hour += 12
    if not is_pm and hour == 12:
        hour = 0

    parts = date_part.replace(",", "").split()
    day, month, year = 1, 1, datetime.now().year
    for part in parts:
        lower = part.lower()
        if lower in _MONTHS:
            month = _MONTHS[lower]
        elif part.isdigit():
            num = int(part)
            year = num if num > 31 else day
            if num <= 31:
                day = num
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute)


# ======================================================================
# LoCoMo 数据集解析
# ======================================================================

def build_sessions_locomo(conversation: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """将 LoCoMo 对话数据解析为按 session 分组的消息列表。"""
    speaker_a = conversation.get("speaker_a", "speaker_a")
    speaker_b = conversation.get("speaker_b", "speaker_b")
    special_keys = {"speaker_a", "speaker_b"}
    sessions: Dict[str, List[Dict[str, Any]]] = {}

    for key, chats in conversation.items():
        if key in special_keys or key.endswith("_date_time"):
            continue
        if not key.startswith("session_"):
            continue

        timestamp_raw = conversation.get(f"{key}_date_time")
        timestamp = parse_timestamp(timestamp_raw) if timestamp_raw else datetime.now()

        session_messages: List[Dict[str, Any]] = []
        for chat in chats or []:
            speaker = chat.get("speaker", speaker_a)
            text = chat.get("text", "")
            parts = [text]
            if chat.get("blip_caption"):
                parts.append(f"[Image: {chat['blip_caption']}]")
            if chat.get("query"):
                parts.append(f"[Search: {chat['query']}]")
            role = speaker if speaker in (speaker_a, speaker_b) else "user"
            session_messages.append({
                "role": role,
                "content": " ".join(parts),
                "timestamp": timestamp.isoformat(),
                "metadata": {
                    "original_speaker": speaker,
                    "dataset_timestamp": timestamp_raw,
                    "blip_caption": chat.get("blip_caption"),
                    "search_query": chat.get("query"),
                    "session": key,
                    "dia_id": chat.get("dia_id"),
                },
            })
        sessions[key] = session_messages

    return sessions
