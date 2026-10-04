"""Speech synthesis (TTS)."""

from .tts import TTSError, TTSBackend, VolcengineTTS, create_tts, synthesize_to_wav

__all__ = [
    "TTSError",
    "TTSBackend",
    "VolcengineTTS",
    "create_tts",
    "synthesize_to_wav",
]
