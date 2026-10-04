"""Frame -> face box, ported from the reference MuseTalk preprocessing.

Backend chain, selected by ``[preprocessing].landmarks``:

``mediapipe``  MediaPipe FaceLandmarker -> head bounding box from all landmarks
``dwpose``     RTMPose whole-body keypoints (MuseTalk's own backend); falls back
               to MediaPipe, then to the optional S3FD detector

Every backend returns the **head/face box** in ``(y1, y2, x1, x2)`` source-frame
pixels, matching what MuseTalk stores in ``coords.pkl``. The caller crops that box
and resizes it to ``output_shape``; MuseTalk never uses a fixed window in source
space.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DWPOSE_BOTTOM_MARGIN = 10

FaceBox = tuple[int, int, int, int]


def fixed_box(face_box: Sequence[float], image_shape: Sequence[int]) -> FaceBox:
    """Clip an (x, y, x, y) box and add MuseTalk 1.5's 10px lower-face margin."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = (float(value) for value in face_box)
    x1 = int(np.clip(np.floor(x1), 0, width - 1))
    y1 = int(np.clip(np.floor(y1), 0, height - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, width))
    y2 = int(np.clip(np.ceil(y2 + DWPOSE_BOTTOM_MARGIN), y1 + 1, height))
    return (y1, y2, x1, x2)


class LandmarkBackend(Protocol):
    name: str

    def box(self, frame: np.ndarray) -> FaceBox | None: ...

    def close(self) -> None: ...


class MediaPipeBackend:
    """Head box from MediaPipe FaceLandmarker landmarks.

    MuseTalk's MediaPipe fallback uses the bounding box of every landmark
    (``_mediapipe_box``), not a mouth-centred fixed window: the crop is the head,
    which the caller then resizes to ``output_shape``.
    """

    name = "mediapipe"

    def __init__(self, model_path: str | Path, confidence: float = 0.5):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MediaPipe model file does not exist: {path}")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        self._landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=confidence,
                min_face_presence_confidence=confidence,
                min_tracking_confidence=confidence,
            )
        )

    def box(self, frame: np.ndarray) -> FaceBox | None:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        )
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        xs = np.asarray([point.x * width for point in landmarks], dtype=np.float32)
        ys = np.asarray([point.y * height for point in landmarks], dtype=np.float32)
        return fixed_box(
            (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
            frame.shape,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


class DwposeBackend:
    """MuseTalk RTMPose whole-body keypoints."""

    name = "dwpose"

    def __init__(self, config_file: str | Path, weights: str | Path, device: str = "cpu"):
        missing = [str(p) for p in (config_file, weights) if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError("DWPose model files are missing: " + ", ".join(missing))
        from mmpose.apis import init_model

        self._model = init_model(str(config_file), str(weights), device=str(device))

    def box(self, frame: np.ndarray) -> FaceBox | None:
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples

        try:
            result = merge_data_samples(inference_topdown(self._model, frame))
            keypoints = result.pred_instances.keypoints[0][23:91].astype(np.float32)
        except Exception as exc:
            logger.warning("DWPose inference failed for this frame: %s", exc)
            return None
        if keypoints.shape[0] <= 29:
            logger.warning("DWPose returned too few keypoints: %s", keypoints.shape)
            return None

        half_face = keypoints[29].copy()
        half_face_distance = float(np.max(keypoints[:, 1]) - half_face[1])
        top = max(0.0, half_face[1] - half_face_distance)
        return fixed_box(
            (
                float(np.min(keypoints[:, 0])),
                top,
                float(np.max(keypoints[:, 0])),
                float(np.max(keypoints[:, 1])),
            ),
            frame.shape,
        )

    def close(self) -> None:
        self._model = None


class SfdBackend:
    """Last-resort S3FD detector; needs the optional face-alignment package."""

    name = "sfd"

    def __init__(self, weights: str | Path, device: str = "cpu"):
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"SFD weights do not exist: {path}")
        from face_alignment.detection.sfd.sfd_detector import SFDDetector

        self._detector = SFDDetector(device=str(device), path_to_detector=str(path))

    def box(self, frame: np.ndarray) -> FaceBox | None:
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch = torch.from_numpy(rgb[None]).permute(0, 3, 1, 2)
        detections = self._detector.detect_from_batch(batch)[0]
        if len(detections) == 0:
            return None
        x1, y1, x2, y2 = np.clip(detections[0][:4], 0, None)
        return fixed_box((x1, y1, x2, y2), frame.shape)

    def close(self) -> None:
        self._detector = None


class FaceDetector:
    """Backend chain. Only raw per-frame results; policy lives in resolve_boxes."""

    def __init__(self, backends: Sequence[LandmarkBackend]):
        if not backends:
            raise ValueError("FaceDetector needs at least one backend")
        self.backends = list(backends)

    @property
    def name(self) -> str:
        return "+".join(backend.name for backend in self.backends)

    def raw_boxes(self, frames: Sequence[np.ndarray]) -> list[FaceBox | None]:
        boxes: list[FaceBox | None] = []
        for frame in frames:
            box: FaceBox | None = None
            for backend in self.backends:
                try:
                    box = backend.box(frame)
                except Exception as exc:
                    logger.warning("%s backend failed on this frame: %s", backend.name, exc)
                    box = None
                if box is not None:
                    break
            boxes.append(box)
        return boxes

    def close(self) -> None:
        for backend in self.backends:
            try:
                backend.close()
            except Exception:
                logger.debug("closing backend failed", exc_info=True)
        self.backends = []


def resolve_boxes(
    raw: Sequence[FaceBox | None],
    *,
    smoothing_window: int = 0,
    max_missed_ratio: float = 0.2,
) -> list[FaceBox]:
    """Apply MuseTalk's reuse/backfill policy to raw detections."""
    total = len(raw)
    if total == 0:
        raise ValueError("no frames to detect faces in")
    missed = sum(1 for box in raw if box is None)
    if missed == total:
        raise ValueError(f"no face detected in any frame (frames={total})")
    if missed / total > max_missed_ratio:
        raise ValueError(
            f"face detection missed {missed}/{total} frames, "
            f"above max_missed_ratio={max_missed_ratio}"
        )

    boxes: list[FaceBox | None] = []
    pending: list[int] = []
    last_box: FaceBox | None = None
    for box in raw:
        if box is None:
            if last_box is None:
                pending.append(len(boxes))
                boxes.append(None)
                continue
            box = last_box
        else:
            last_box = box
            if pending:
                for position in pending:
                    boxes[position] = box
                pending.clear()
        boxes.append(box)
    if pending:
        raise ValueError("no valid face box found to backfill the leading frames")

    resolved = [box for box in boxes if box is not None]
    if smoothing_window > 1:
        resolved = _smooth_boxes(resolved, min(int(smoothing_window), len(resolved)))
    return resolved


def _smooth_boxes(boxes: Sequence[FaceBox], window: int) -> list[FaceBox]:
    array = np.asarray(boxes, dtype=np.float32).copy()
    for index in range(len(array)):
        chunk = array[index:index + window]
        if len(chunk) < window:
            chunk = array[-window:]
        array[index] = np.mean(chunk, axis=0)
    return [(int(v[0]), int(v[1]), int(v[2]), int(v[3])) for v in np.rint(array)]


def configured_path(
    preprocessing: Mapping[str, Any], section: str, key: str, project_root: Path
) -> Path:
    value = (preprocessing.get(section) or {}).get(key)
    if not value:
        raise ValueError(f"[preprocessing.{section}].{key} is required")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path(project_root) / path


def create_detector(
    preprocessing: Mapping[str, Any], project_root: Path, device: str
) -> FaceDetector:
    """Build the backend chain selected by ``[preprocessing].landmarks``."""
    backend = str(preprocessing.get("landmarks") or "mediapipe").strip().lower()
    if backend not in {"mediapipe", "dwpose"}:
        raise ValueError(
            f"[preprocessing].landmarks must be 'mediapipe' or 'dwpose', got {backend!r}"
        )
    confidence = float(preprocessing.get("min_face_confidence", 0.5))

    def mediapipe_backend() -> MediaPipeBackend:
        return MediaPipeBackend(
            configured_path(preprocessing, "mediapipe", "model", project_root), confidence
        )

    backends: list[LandmarkBackend] = []
    if backend == "dwpose":
        try:
            backends.append(
                DwposeBackend(
                    configured_path(preprocessing, "dwpose", "config_file", project_root),
                    configured_path(preprocessing, "dwpose", "weights", project_root),
                    device=device,
                )
            )
        except Exception as exc:
            # A broken mmpose/mmcv install must not take the whole pipeline down.
            logger.warning("DWPose backend unavailable, continuing without it: %s", exc)

    try:
        backends.append(mediapipe_backend())
    except Exception as exc:
        logger.warning("MediaPipe backend unavailable: %s", exc)

    if bool((preprocessing.get("sfd") or {}).get("sfd_fallback", False)):
        try:
            backends.append(
                SfdBackend(
                    configured_path(preprocessing, "sfd", "weights", project_root),
                    device="cpu",
                )
            )
        except Exception as exc:
            logger.warning("SFD fallback disabled: %s", exc)

    if not backends:
        raise RuntimeError("no face detection backend could be initialized")
    return FaceDetector(backends)
