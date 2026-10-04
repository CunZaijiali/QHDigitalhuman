from .asr import MuseTalkASR, Wav2LipASR
from .musetalk import MuseTalk
from .pipeline import LipSyncInput, LipSyncOutput
from .wav2lip import Wav2Lip


def create_engine(name: str, model=None, feature_extractor=None):
    """Create a model pipeline without exposing model/session internals."""
    normalized = name.strip().lower().replace("_", "")
    if normalized in {"musetalk", "musetalkv15"}:
        return MuseTalk(model=model, feature_extractor=feature_extractor)
    if normalized == "wav2lip":
        return Wav2Lip(model=model, feature_extractor=feature_extractor)
    raise ValueError(f"Unsupported lip-sync engine: {name}")

__all__ = [
    "LipSyncInput",
    "LipSyncOutput",
    "MuseTalkASR",
    "Wav2LipASR",
    "MuseTalk",
    "Wav2Lip",
    "create_engine",
]
