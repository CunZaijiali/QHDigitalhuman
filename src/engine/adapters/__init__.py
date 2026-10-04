"""Model adapters: loaded model objects -> the engine's ``infer`` contract."""

from __future__ import annotations

from src.domain.models.loaders import MUSETALK_ALIASES, normalize_engine

from .musetalk_adapter import MuseTalkAdapter
from .wav2lip_adapter import Wav2LipAdapter


def create_adapter(engine: str, models: object):
    """Pick the adapter matching ``engine`` for an already-loaded model bundle."""
    normalized = normalize_engine(engine)
    if normalized in MUSETALK_ALIASES:
        return MuseTalkAdapter(models)
    if normalized == "wav2lip":
        return Wav2LipAdapter(models)
    raise ValueError(f"unsupported engine: {engine}")


__all__ = ["MuseTalkAdapter", "Wav2LipAdapter", "create_adapter"]
