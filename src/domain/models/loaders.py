"""Model loading for the lip-sync engines.

Every weight path comes from ``config.toml`` under ``[models.*]``; this module is
the only place that turns those paths into ready objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

MUSETALK_ALIASES = {"musetalk", "musetalkv15"}


@dataclass
class MuseTalkModels:
    """Everything the MuseTalk pipeline needs, already on the target device."""

    unet: Any
    positional_encoding: Any
    vae: Any
    whisper: Any
    feature_extractor: Any
    device: str = "cpu"


def normalize_engine(engine: str) -> str:
    return (engine or "").strip().lower().replace("_", "")


def resolve_model_path(value: str | Path, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path(project_root) / path


def _required(section: Mapping[str, Any], key: str, project_root: Path, where: str) -> Path:
    value = (section or {}).get(key)
    if not value:
        raise ValueError(f"[{where}].{key} is required in config.toml")
    path = resolve_model_path(value, project_root)
    if not path.exists():
        raise FileNotFoundError(f"[{where}].{key} points at a missing path: {path}")
    return path


def load_whisper(
    model_dir: str | Path, device: str = "cpu", use_float16: bool = False
) -> tuple[Any, Any]:
    """Load the MuseTalk Whisper encoder and its feature extractor from local weights."""
    directory = Path(model_dir)
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(f"Whisper directory has no config.json: {directory}")
    if not (directory / "preprocessor_config.json").is_file():
        raise FileNotFoundError(
            f"Whisper directory has no preprocessor_config.json: {directory}"
        )
    from transformers import AutoFeatureExtractor, WhisperModel

    whisper = WhisperModel.from_pretrained(str(directory), local_files_only=True)
    whisper = whisper.to(device)
    if use_float16:
        whisper = whisper.half()
    whisper.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        str(directory), local_files_only=True
    )
    logger.info("whisper loaded: dir=%s device=%s fp16=%s", directory, device, use_float16)
    return whisper, feature_extractor


def load_musetalk_models(
    *,
    unet_path: str | Path,
    unet_config: str | Path,
    vae_path: str | Path,
    whisper_path: str | Path,
    device: str = "cpu",
    use_float16: bool = False,
) -> MuseTalkModels:
    """Build the UNet + VAE + positional encoding + Whisper bundle."""
    from src.domain.models.unet import PositionalEncoding, UNet
    from src.domain.models.vae import VAE

    vae_dir = Path(vae_path)
    if not (vae_dir / "config.json").is_file():
        raise FileNotFoundError(f"VAE directory has no config.json: {vae_dir}")

    vae = VAE(model_path=str(vae_dir), device=device, use_float16=use_float16)
    unet = UNet(str(unet_config), str(unet_path), use_float16=use_float16, device=device)
    positional_encoding = PositionalEncoding(d_model=384).to(device)
    positional_encoding.eval()
    whisper, feature_extractor = load_whisper(
        whisper_path, device=device, use_float16=use_float16
    )
    logger.info("musetalk models loaded: unet=%s vae=%s device=%s", unet_path, vae_dir, device)
    return MuseTalkModels(
        unet=unet,
        positional_encoding=positional_encoding,
        vae=vae,
        whisper=whisper,
        feature_extractor=feature_extractor,
        device=device,
    )


def load_wav2lip_model(checkpoint_path: str | Path, device: str = "cpu") -> Any:
    """Build the Wav2Lip network and load its checkpoint."""
    from src.domain.models.wav2lip import build_wav2lip_model

    return build_wav2lip_model(str(checkpoint_path), device=device)


def load_models(
    engine: str,
    config: Mapping[str, Any],
    project_root: str | Path,
    device: str | None = None,
) -> Any:
    """Load every model the given engine needs, driven by ``config.toml``."""
    project_root = Path(project_root)
    normalized = normalize_engine(engine)
    device = device or (config.get("digitalhuman", {}) or {}).get("device", "cpu")
    use_float16 = str(device).startswith("cuda")
    models_section = config.get("models", {}) or {}

    if normalized in MUSETALK_ALIASES:
        section = models_section.get("musetalk", {}) or {}
        return load_musetalk_models(
            unet_path=_required(section, "unet", project_root, "models.musetalk"),
            unet_config=_required(section, "unet_config", project_root, "models.musetalk"),
            vae_path=_required(section, "vae", project_root, "models.musetalk"),
            whisper_path=_required(section, "whisper", project_root, "models.musetalk"),
            device=str(device),
            use_float16=use_float16,
        )

    if normalized == "wav2lip":
        section = models_section.get("wav2lip", {}) or {}
        return load_wav2lip_model(
            _required(section, "checkpoint", project_root, "models.wav2lip"),
            device=str(device),
        )

    raise ValueError(f"unsupported engine: {engine}")


def describe_models(engine: str, models: Any) -> dict:
    """Small JSON-friendly summary used by the CLI."""
    normalized = normalize_engine(engine)
    if normalized in MUSETALK_ALIASES:
        unet_model = getattr(models.unet, "model", None)
        return {
            "engine": "musetalkv15",
            "device": models.device,
            "unet": {
                "parameters": sum(p.numel() for p in unet_model.parameters())
                if unet_model is not None
                else None,
                "dtype": str(next(unet_model.parameters()).dtype) if unet_model is not None else None,
            },
            "vae": {"scaling_factor": getattr(models.vae, "scaling_factor", None)},
            "whisper": {
                "parameters": sum(p.numel() for p in models.whisper.parameters()),
                "dtype": str(next(models.whisper.parameters()).dtype),
            },
        }
    return {
        "engine": "wav2lip",
        "parameters": sum(p.numel() for p in models.parameters()),
        "dtype": str(next(models.parameters()).dtype),
    }


__all__ = [
    "MuseTalkModels",
    "describe_models",
    "load_models",
    "load_musetalk_models",
    "load_wav2lip_model",
    "load_whisper",
    "normalize_engine",
    "resolve_model_path",
]
