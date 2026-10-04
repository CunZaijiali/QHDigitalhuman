"""Text to speech.

Currently one backend: **Volcengine Seed TTS V3** (unidirectional HTTP chunked).
The protocol is ported from the reference implementation; the surrounding
session/queue machinery is not, because this project synthesises a clip up front
instead of streaming TTS into a live renderer.

Secrets never live in ``config.toml``: the API key is read from the environment,
with ``.env`` loaded through ``python-dotenv``. See ``.env.example``.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)

# Volcengine V3 reports success with these codes (and sometimes omits it).
_SUCCESS_CODES = {0, "0", 20000000, "20000000", None}
_RESOURCE_MISMATCH_CODE = "55000000"


class TTSError(RuntimeError):
    """Raised when synthesis cannot be attempted or the service rejects it."""


class TTSBackend(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray: ...


@dataclass
class VolcengineTTS:
    """Volcengine Seed TTS V3 client.

    ``synthesize`` returns mono float32 PCM in ``[-1, 1]`` at ``sample_rate``.
    """

    api_key: str
    speaker: str
    name: str = "volcengine"
    endpoint: str = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    resource_id: str = "seed-tts-2.0"
    model: str = "seed-tts-2.0-standard"
    sample_rate: int = 16000
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise TTSError(
                "Volcengine TTS needs an API key. Put "
                "VOLCENGINE_TTS_API_KEY=... in .env (see .env.example)."
            )
        if not self.speaker:
            raise TTSError("Volcengine TTS needs a speaker, see [tts.volcengine].speaker")
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise TTSError(f"Volcengine TTS endpoint is not a valid https URL: {self.endpoint}")

    def build_request(self, text: str) -> tuple[dict, dict]:
        """Return ``(headers, json_body)`` for one synthesis call."""
        headers = {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }
        params: dict[str, Any] = {
            "text": text,
            "model": self.model,
            "speaker": self.speaker,
            "audio_params": {"format": "pcm", "sample_rate": self.sample_rate},
        }
        if text.lstrip().startswith("<speak"):
            params["text_type"] = "ssml"
        return headers, {"req_params": params}

    def synthesize(self, text: str) -> np.ndarray:
        import requests

        text = (text or "").strip()
        if not text:
            raise TTSError("nothing to synthesize: the text is empty")

        headers, body = self.build_request(text)
        logger.info(
            "volcengine tts: resource_id=%s model=%s speaker=%s chars=%d",
            self.resource_id,
            self.model,
            self.speaker,
            len(text),
        )
        chunks: list[np.ndarray] = []
        try:
            with requests.post(
                self.endpoint,
                headers=headers,
                json=body,
                stream=True,
                timeout=(10, self.timeout),
            ) as response:
                if response.status_code != 200:
                    raise TTSError(
                        f"Volcengine TTS returned HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    pcm = self._decode_chunk(raw_line)
                    if pcm is not None:
                        chunks.append(pcm)
        except requests.RequestException as exc:
            raise TTSError(f"Volcengine TTS request failed: {exc}") from exc

        if not chunks:
            raise TTSError("Volcengine TTS returned no audio")
        return np.concatenate(chunks)

    def _decode_chunk(self, raw_line: str) -> np.ndarray | None:
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            logger.debug("volcengine tts: ignoring non-JSON line: %s", raw_line[:120])
            return None

        code = payload.get("code", 0)
        if code not in _SUCCESS_CODES:
            message = payload.get("message", "")
            hint = ""
            if str(code) == _RESOURCE_MISMATCH_CODE:
                hint = (
                    " (resource/speaker mismatch: this speaker does not belong to "
                    f"resource_id={self.resource_id}; change [tts.volcengine].speaker "
                    "or .resource_id)"
                )
            raise TTSError(f"Volcengine TTS error code={code} message={message}{hint}")

        encoded = payload.get("data")
        if not encoded:
            return None
        pcm16 = np.frombuffer(base64.b64decode(encoded), dtype="<i2")
        if pcm16.size == 0:
            return None
        return pcm16.astype(np.float32) / 32768.0


def create_tts(config: dict | None = None) -> TTSBackend:
    """Build the backend selected by ``[digitalhuman].tts_engine``."""
    from src.utils import get_config

    config = config if config is not None else get_config()
    engine = str((config.get("digitalhuman", {}) or {}).get("tts_engine") or "volcengine")
    tts_section = dict(config.get("tts", {}) or {})
    sample_rate = int(tts_section.get("sample_rate", 16000))

    normalized = engine.strip().lower()
    if normalized in {"volcengine", "volc", "seed"}:
        section = dict(tts_section.get("volcengine", {}) or {})
        import os

        key_env = str(section.get("api_key_env") or "VOLCENGINE_TTS_API_KEY")
        return VolcengineTTS(
            api_key=os.getenv(key_env, "").strip(),
            speaker=str(section.get("speaker") or ""),
            endpoint=str(
                section.get("endpoint")
                or "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
            ),
            resource_id=str(section.get("resource_id") or "seed-tts-2.0"),
            model=str(section.get("model") or "seed-tts-2.0-standard"),
            sample_rate=sample_rate,
            timeout=float(tts_section.get("request_timeout", 120)),
        )
    raise TTSError(f"unsupported [digitalhuman].tts_engine: {engine!r}")


def synthesize_to_wav(
    text: str,
    output_path: str | Path,
    *,
    backend: TTSBackend | None = None,
    config: dict | None = None,
) -> Path:
    """Synthesize ``text`` and write a mono 16-bit WAV; returns the path."""
    import wave

    from src.utils import get_config

    config = config if config is not None else get_config()
    backend = backend or create_tts(config)
    samples = np.asarray(backend.synthesize(text), dtype=np.float32).reshape(-1)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(int(backend.sample_rate))
        stream.writeframes(pcm16.tobytes())
    logger.info(
        "tts wrote %.2fs of audio to %s", samples.size / backend.sample_rate, path
    )
    return path


__all__ = [
    "TTSError",
    "TTSBackend",
    "VolcengineTTS",
    "create_tts",
    "synthesize_to_wav",
]
