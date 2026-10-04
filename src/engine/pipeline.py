"""Shared input/output objects for offline lip-sync engines.

The engines deliberately accept plain data and a model adapter. They do not
know about users, HTTP requests, TTS, ASR, or a digital-human session object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.domain.avatar import Avatar
from src.engine.blending import blend_generated_region, get_image_blending


@dataclass(frozen=True)
class LipSyncInput:
    """Input to one offline synthesis operation.

    ``audio_features`` are model-ready features. Feature extraction belongs to
    the caller or to an engine-specific adapter and is intentionally not tied
    to a TTS/ASR implementation here.
    """

    resource_dir: str | Path
    audio: str | Path | np.ndarray | None = None
    audio_features: np.ndarray | None = None
    action_key: str | None = None
    output_path: str | Path | None = None
    fps: int = 25
    sample_rate: int = 16000
    frame_count: int | None = None
    #: Global timeline position of local frame 0. The audio feature index is
    #: ``(index + frame_offset) % len(features)`` so that rendering one action in
    #: several chunks (rotation, or skipping frames to stay on the audio clock)
    #: keeps the audio in step with the wall clock.
    frame_offset: int = 0
    #: Action-local frame counter for the *assets*. Keeping this separate lets the
    #: audio follow the clock (skip frames) while the pose still advances one frame
    #: per rendered frame instead of jumping.
    asset_offset: int = 0
    #: Frames per inference batch; None uses the engine's own limit.
    batch_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LipSyncOutput:
    """Rendered frames and optional encoded video path."""

    frames: list[np.ndarray]
    audio: str | Path | np.ndarray | None = None
    output_path: Path | None = None
    fps: int = 25
    sample_rate: int = 16000
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def load_action_resource(resource_dir: str | Path) -> Avatar:
    """Load the persisted preprocessing contract.

    The contract lives in :class:`src.domain.avatar.Avatar`: ``full_imgs``,
    ``face_imgs``, ``mask``, ``coords.pkl`` ``(y1, y2, x1, x2)`` and
    ``mask_coords.pkl`` ``(x1, y1, x2, y2)``. ``coords`` feeds ``paste_face``
    while ``mask_coords`` feeds ``get_image_blending``, matching MuseTalk.
    """
    return Avatar.load(resource_dir)


def resolve_action_resource(resource_root: str | Path, action_key: str | None) -> Avatar:
    """Resolve an action from an avatar root, or load an action directory directly."""
    root = Path(resource_root)
    if action_key:
        behavior, separator, action = action_key.strip().upper().partition(".")
        if not separator or not behavior or not action:
            raise ValueError("action_key must use the <behavior>.<action> format")
        action_root = root / behavior / action
        if not (action_root / "coords.pkl").is_file() and action_root.is_dir():
            # Engines live in a child directory; resolve it instead of hardcoding names.
            engines = sorted(
                child
                for child in action_root.iterdir()
                if child.is_dir() and (child / "coords.pkl").is_file()
            )
            if len(engines) > 1:
                raise ValueError(
                    f"Multiple engines found under {action_root}: "
                    f"{[path.name for path in engines]}"
                )
            if engines:
                action_root = engines[0]
        root = action_root
    return load_action_resource(root)


def render_output(output: LipSyncOutput) -> LipSyncOutput:
    """Encode frames when the caller requested an output video path."""
    if output.output_path is None:
        return output
    if not output.frames:
        raise ValueError("Cannot encode an empty output")
    path = output.output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = output.frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), output.fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"Unable to create output video: {path}")
    try:
        for frame in output.frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("All output frames must have the same dimensions")
            writer.write(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.release()
    return output


def paste_face(
    frame: np.ndarray,
    generated: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    mask: np.ndarray | None = None,
    mask_coords: tuple[int, int, int, int] | None = None,
    unsharp: float = 0.0,
) -> np.ndarray:
    """Paste a generated face ROI into its source frame, MuseTalk style.

    With ``mask``/``mask_coords`` (an action that has ``mask/`` artifacts) the
    ROI is composited through the jaw mask; otherwise a feathered lower-face mask
    is used, matching MuseTalk's fallback. ``unsharp`` applies the restrained
    sharpening MuseTalk uses to restore detail lost in VAE decode.
    """
    y1, y2, x1, x2 = box
    if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
        raise ValueError(f"Face ROI is outside source frame: {box}")
    generated = np.clip(generated, 0, 255).astype(np.uint8)
    if unsharp:
        softened = cv2.GaussianBlur(generated, (0, 0), 0.5)
        generated = cv2.addWeighted(generated, 1.0 + unsharp, softened, -unsharp, 0)
    generated = cv2.resize(generated, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    if mask is not None and mask_coords is not None:
        return get_image_blending(frame, generated, (x1, y1, x2, y2), mask, mask_coords)
    result = frame.copy()
    result[y1:y2, x1:x2] = blend_generated_region(frame[y1:y2, x1:x2], generated)
    return result


def cycle_index(length: int, index: int) -> int:
    if length <= 0:
        raise ValueError("Cannot cycle an empty resource")
    turn, offset = divmod(index, length)
    return offset if turn % 2 == 0 else length - offset - 1


__all__ = [
    "LipSyncInput",
    "LipSyncOutput",
    "load_action_resource",
    "resolve_action_resource",
    "render_output",
]
