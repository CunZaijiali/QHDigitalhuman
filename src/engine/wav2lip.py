"""Wav2Lip input/output pipeline.

Like MuseTalk, ``process`` renders the whole clip while ``iter_frames`` yields
frames batch by batch for streaming.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .pipeline import (
    LipSyncInput,
    LipSyncOutput,
    cycle_index,
    paste_face,
    render_output,
    resolve_action_resource,
)

BATCH_LIMIT = 16


@dataclass
class _Plan:
    resource: Any
    features: np.ndarray
    count: int
    batch: int
    cycle_length: int
    offset: int = 0
    asset_offset: int = 0


class Wav2Lip:
    """Run Wav2Lip without a realtime/session dependency.

    The model adapter receives ``(mel_batch, image_batch)``. ``image_batch`` is
    a float32 BGR batch in ``[0, 1]`` with shape ``[batch, height, width, 3]``;
    it must return generated BGR face images.
    """

    name = "wav2lip"

    def __init__(self, model: Any | None = None, feature_extractor: Any | None = None):
        if feature_extractor is None:
            from .asr import Wav2LipASR

            feature_extractor = Wav2LipASR()
        self.model = model
        self.feature_extractor = feature_extractor

    def _plan(self, request: LipSyncInput) -> _Plan:
        resource = resolve_action_resource(request.resource_dir, request.action_key)
        features = request.audio_features
        if features is None and request.audio is not None and self.feature_extractor is not None:
            if hasattr(self.feature_extractor, "extract"):
                features = self.feature_extractor.extract(request.audio, request.frame_count)
            else:
                features = self.feature_extractor(request.audio)
        if features is None:
            raise ValueError("Wav2Lip requires model-ready audio_features")
        features = np.asarray(features)
        if features.ndim == 0 or len(features) == 0:
            raise ValueError("audio_features must contain at least one frame")
        count = request.frame_count or len(features)
        if count <= 0:
            raise ValueError("frame_count must be positive")

        # A single ping-pong index drives face, frame and coords, matching
        # MuseTalk's mirror_index; mixing it with a modulo wrap desynchronises
        # the face from its background once the render outlasts the clip.
        cycle_length = min(
            len(resource.full_paths), len(resource.face_paths), len(resource.coords)
        )
        if cycle_length <= 0:
            raise ValueError(f"action resource has no usable frames: {resource.root}")

        batch_limit = int(request.batch_size or BATCH_LIMIT)
        return _Plan(
            resource=resource,
            features=features,
            count=count,
            batch=max(1, min(max(1, batch_limit), count)),
            cycle_length=cycle_length,
            offset=int(request.frame_offset or 0),
            asset_offset=int(request.asset_offset or 0),
        )

    def _apply(self, plan: _Plan) -> Iterator[np.ndarray]:
        resource, features = plan.resource, plan.features
        for start in range(0, plan.count, plan.batch):
            indexes = list(range(start, min(start + plan.batch, plan.count)))
            slots = [
                cycle_index(plan.cycle_length, index + plan.asset_offset)
                for index in indexes
            ]
            images = np.asarray(
                [resource.face(slot) for slot in slots], dtype=np.float32
            ) / 255.0
            mel_batch = np.asarray(
                [features[(index + plan.offset) % len(features)] for index in indexes]
            )
            predicted = self._infer(mel_batch, images)
            if len(predicted) != len(indexes):
                raise ValueError("Wav2Lip model output count does not match input batch")
            for slot, generated in zip(slots, predicted):
                yield paste_face(
                    resource.frame(slot), generated, resource.coords[slot]
                )

    def iter_frames(self, request: LipSyncInput) -> Iterator[np.ndarray]:
        """Yield finished frames as soon as they exist (streaming entry point)."""
        return self._apply(self._plan(request))

    def process(self, request: LipSyncInput) -> LipSyncOutput:
        plan = self._plan(request)
        frames = list(self._apply(plan))
        resource = plan.resource
        return render_output(
            LipSyncOutput(
                frames=frames,
                audio=request.audio,
                output_path=Path(request.output_path) if request.output_path else None,
                fps=request.fps,
                sample_rate=request.sample_rate,
                metadata={
                    **request.metadata,
                    "engine": self.name,
                    "action_key": request.action_key,
                    "action_root": str(resource.root),
                    "blended": bool(resource.mask_paths and resource.mask_coords),
                },
            )
        )

    def _infer(self, mel_batch: np.ndarray, images: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Wav2Lip model is not configured")
        if hasattr(self.model, "infer"):
            result = self.model.infer(mel_batch, images)
        elif callable(self.model):
            result = self.model(mel_batch, images)
        else:
            raise TypeError("Wav2Lip model must be callable or expose infer()")
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        result = np.asarray(result)
        if result.ndim != 4 or result.shape[-1] != 3:
            raise ValueError("Wav2Lip model output must have shape [batch, height, width, 3]")
        if np.issubdtype(result.dtype, np.floating) and result.max(initial=0) <= 1.0:
            result = result * 255.0
        return result


__all__ = ["Wav2Lip"]
