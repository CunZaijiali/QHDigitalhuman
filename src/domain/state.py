from enum import Enum
import re


class EventPoint(Enum):
    """系统状态枚举"""
    REAL_START = "REAL_START"
    REAL_READING = "REAL_READING"
    REAL_INITIALIZED = "REAL_INITIALIZED"
    TTS_REQUEST = "TTS_REQUEST"
    TTS_RESPONSE = "TTS_RESPONSE"
    TTS_AUDIO_START = "TTS_AUDIO_START"
    TTS_AUDIO_END = "TTS_AUDIO_END"
    ACTION_REQUEST = "ACTION_REQUEST"
    ACTION_END = "ACTION_END"
    INTERRUPTED = "INTERRUPTED"


class BehaviorState(Enum):
    """固定的资源行为维度；动作名称由资源上传方自定义。"""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTRODUCE = "INTRODUCE"
    FAREWELL = "FAREWELL"

    INTERRUPTED = "INTERRUPTED"


_ACTION_RE = re.compile(r"^[A-Z0-9][A-Z0-9_&-]{0,63}$")


def normalize_action_name(value: str) -> str:
    normalized = (value or "").strip().upper()
    if not _ACTION_RE.fullmatch(normalized):
        raise ValueError("action must contain 1-64 letters, numbers, underscores, hyphens, or ampersands")
    return normalized


def parse_behavior_action(value: str) -> tuple[BehaviorState, str]:
    behavior_name, separator, action_name = (value or "").strip().upper().partition(".")
    if not separator or not action_name or "." in action_name:
        raise ValueError("action must use the <behavior>.<action> format")
    try:
        behavior = BehaviorState(behavior_name)
    except ValueError as exc:
        raise ValueError(f"unknown behavior: {behavior_name}") from exc
    return behavior, normalize_action_name(action_name)
