"""MuseTalk pipeline.

This module exposes a stateless input/output API. Model loading is injected so
the engine does not depend on a session object, TTS implementation, or ASR
implementation.

Two entry points share one render loop:

* :meth:`MuseTalk.process` renders the whole clip and returns a ``LipSyncOutput``
  (offline file rendering),
* :meth:`MuseTalk.iter_frames` yields frames as each batch finishes, so a session
  can push them while the rest is still rendering.

The model adapter receives ``(face_batch, feature_batch, latent_batch)``:

* ``face_batch`` is unused by MuseTalk itself (kept so the frame count and face
  assets stay validated here); it is ``[batch, H, W, 3]`` BGR.
* ``feature_batch`` is the Whisper prompt batch, ``[batch, 50, 384]``.
* ``latent_batch`` is the cached VAE latent batch, ``[batch, 8, 32, 32]``.

It returns BGR face images with shape ``[batch, height, width, 3]``.
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
# MuseTalk restores edge definition lost in VAE decode with a restrained unsharp pass.
UNSHARP_AMOUNT = 0.08


@dataclass
class _Plan:
    """Everything needed to render one request, resolved once."""

    resource: Any
    features: np.ndarray
    latents: Any
    count: int
    batch: int
    cycle_length: int
    offset: int = 0
    asset_offset: int = 0


class MuseTalk:
    """Run MuseTalk with a model adapter and persisted avatar resources."""

    name = "musetalk"

    def __init__(self, model: Any | None = None, feature_extractor: Any | None = None):
        self.model = model
        self.feature_extractor = feature_extractor

    # ------------------------------------------------------------------ planning
    def _plan(self, request: LipSyncInput) -> _Plan:
        resource = resolve_action_resource(request.resource_dir, request.action_key)
        features = request.audio_features
        if features is None and request.audio is not None and self.feature_extractor is not None:
            if hasattr(self.feature_extractor, "extract"):
                features = self.feature_extractor.extract(request.audio, request.frame_count)
            else:
                features = self.feature_extractor(request.audio)
        if features is None:
            raise ValueError("MuseTalk requires model-ready audio_features")
        features = np.asarray(features)
        if features.ndim == 0 or len(features) == 0:
            raise ValueError("audio_features must contain at least one frame")

        latents = resource.latents
        if latents is None:
            raise ValueError(
                f"MuseTalk needs latens.pt, which {resource.root} does not have "
                f"(was this action preprocessed for wav2lip?)"
            )
        count = request.frame_count or len(features)
        if count <= 0:
            raise ValueError("frame_count must be positive")

        # One index drives every asset, exactly like MuseTalk's
        # ``length = min(len(latents), len(faces))``. Using different lengths per
        # asset would paste a late face onto an early background frame.
        lengths = [
            len(resource.full_paths),
            len(resource.face_paths),
            len(resource.coords),
            len(latents),
        ]
        if resource.mask_paths and resource.mask_coords:
            lengths.append(min(len(resource.mask_paths), len(resource.mask_coords)))
        cycle_length = min(lengths)
        if cycle_length <= 0:
            raise ValueError(f"action resource has no usable frames: {resource.root}")

        batch_limit = int(request.batch_size or BATCH_LIMIT)
        return _Plan(
            resource=resource,
            features=features,
            latents=latents,
            count=count,
            batch=max(1, min(max(1, batch_limit), count)),
            cycle_length=cycle_length,
            offset=int(request.frame_offset or 0),
            asset_offset=int(request.asset_offset or 0),
        )

    # ------------------------------------------------------------------ rendering
    def _apply(self, plan: _Plan) -> Iterator[np.ndarray]:
        """Yield pasted BGR frames, one batch of inference at a time."""
        resource, features, latents = plan.resource, plan.features, plan.latents
        for start in range(0, plan.count, plan.batch):
            indexes = list(range(start, min(start + plan.batch, plan.count)))
            slots = [
                cycle_index(plan.cycle_length, index + plan.asset_offset)
                for index in indexes
            ]
            faces = np.asarray([resource.face(slot) for slot in slots])
            latent_batch = np.asarray(
                [np.asarray(latents[slot]) for slot in slots], dtype=np.float32
            )
            feature_batch = np.asarray(
                [features[(index + plan.offset) % len(features)] for index in indexes]
            )
            predicted = self._infer(faces, feature_batch, latent_batch)
            if len(predicted) != len(indexes):
                raise ValueError("MuseTalk model output count does not match input batch")
            for slot, generated in zip(slots, predicted):
                yield self._paste(resource, slot, generated)

    def _paste(self, resource, slot: int, generated: np.ndarray) -> np.ndarray:
        """Paste onto frame ``slot``; the same slot indexes face, coords and mask."""
        mask = None
        mask_coords = None
        if resource.mask_paths and resource.mask_coords:
            mask = resource.mask(slot)
            mask_coords = resource.mask_coords[slot]
        return paste_face(
            resource.frame(slot),
            generated,
            resource.coords[slot],
            mask=mask,
            mask_coords=mask_coords,
            unsharp=UNSHARP_AMOUNT,
        )

    # -------------------------------------------------------------------- public
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

    # ---------------------------------------------------------------------- model
    def _infer(
        self, faces: np.ndarray, features: np.ndarray, latents: np.ndarray
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MuseTalk model is not configured")
        if hasattr(self.model, "infer"):
            result = self.model.infer(faces, features, latents)
        elif callable(self.model):
            result = self.model(faces, features, latents)
        else:
            raise TypeError("MuseTalk model must be callable or expose infer()")
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        result = np.asarray(result)
        if result.ndim != 4 or result.shape[-1] != 3:
            raise ValueError("MuseTalk model output must have shape [batch, height, width, 3]")
        return result


__all__ = ["MuseTalk"]
