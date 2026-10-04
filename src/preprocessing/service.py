"""Offline avatar preprocessing.

The engine chosen in ``config.toml`` (``[digitalhuman].lip_sync_engine``) decides
which artifacts are produced; see :data:`src.storage.paths.ASSETS_BY_ENGINE`.
Common to every engine::

    full_imgs/       源帧，偶数尺寸
    face_imgs/       output_shape 方形人脸裁剪
    coords.pkl       [(y1, y2, x1, x2)]   人脸框，源图坐标

Additional artifacts for ``musetalkv15``::

    mask/            扩展裁剪区上的融合 mask
    mask_coords.pkl  [(x1, y1, x2, y2)]   扩展裁剪框，源图坐标，与 mask/ 配对
    latens.pt        VAE latents [N, 8, 32, 32]

``coords`` 与 ``mask_coords`` 的字段顺序故意**不同**，与 MuseTalk 上游一致：
``coords`` 供 ``paste_face`` 使用，``mask_coords`` 供 ``get_image_blending`` 使用。
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch

from src.storage import (
    LATENTS_FILE,
    MASK_DIR,
    ActionPaths,
    allocate_index,
    assets_for,
    build_action_dir,
)
from src.utils import find_project_root, get_config

from .blend_mask import FaceBlendMasker
from .detector import FaceDetector, create_detector, resolve_boxes

logger = logging.getLogger(__name__)

MODEL_FACE_SIZE = 256


@dataclass(frozen=True)
class PreprocessResult:
    root: Path
    frame_count: int
    face_count: int
    missed_frames: int
    output_shape: int


class AvatarPreprocessor:
    """Builds one action asset. Heavy models are created lazily and reused."""

    def __init__(self, config: dict | None = None, project_root: Path | str | None = None):
        self.config = config or get_config()
        self.project_root = Path(project_root or find_project_root())
        self.pre = dict(self.config.get("preprocessing", {}) or {})
        self._detector: FaceDetector | None = None
        self._masker: FaceBlendMasker | None = None
        self._vae = None

    # ------------------------------------------------------------------ models
    @property
    def device(self) -> str:
        return str(self.pre.get("device") or "cpu")

    @property
    def output_shape(self) -> int:
        return int(self.pre.get("output_shape", MODEL_FACE_SIZE))

    @property
    def frame_name_width(self) -> int:
        return int(self.pre.get("frame_name_width", 8))

    def resolve_path(self, value) -> Path:
        if not value:
            raise ValueError("missing model path in [preprocessing] config")
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else self.project_root / path

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = create_detector(self.pre, self.project_root, self.device)
        return self._detector

    @property
    def masker(self) -> FaceBlendMasker:
        if self._masker is None:
            from src.domain.models.bisent import FaceParsing

            parser_section = self.pre.get("face_parse", {}) or {}
            blend = self.pre.get("blend_mask", {}) or {}
            parser = FaceParsing(
                resnet_path=self.resolve_path(parser_section.get("resnet")),
                model_pth=self.resolve_path(parser_section.get("model")),
                device=self.device,
            )
            self._masker = FaceBlendMasker(
                parser,
                mode=str(blend.get("mode", "jaw")),
                expand=float(blend.get("expand", 1.5)),
                upper_boundary_ratio=float(blend.get("upper_boundary_ratio", 0.5)),
                blur_ratio=float(blend.get("blur_ratio", 0.1)),
            )
        return self._masker

    @property
    def vae(self):
        if self._vae is None:
            from src.domain.models.vae import VAE

            section = self.pre.get("vae", {}) or {}
            self._vae = VAE(
                model_path=str(self.resolve_path(section.get("model"))),
                device=torch.device(self.device),
                use_float16=bool(section.get("use_float16", False)),
            )
        return self._vae

    def load_models(self) -> None:
        """Eagerly build every model, so startup can report failures early."""
        _ = self.detector
        _ = self.masker
        _ = self.vae

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        self._masker = None
        self._vae = None

    # ------------------------------------------------------------------ frames
    @staticmethod
    def _ensure_even_frame(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        frame = frame[: height - height % 2, : width - width % 2]
        if frame.shape[0] < 2 or frame.shape[1] < 2:
            raise ValueError(
                f"frame is too small after even-size normalization: {width}x{height}"
            )
        return frame

    @staticmethod
    def _reset_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for item in path.iterdir():
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)

    def _extract_frames(
        self, video_path: Path, paths: ActionPaths
    ) -> tuple[list[Path], tuple[int, int]]:
        source = Path(video_path)
        if not source.is_file():
            raise FileNotFoundError(f"Video not found: {source}")
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {source}")
        self._reset_dir(paths.full_dir)
        frame_paths: list[Path] = []
        size = (0, 0)
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = self._ensure_even_frame(frame)
                size = (frame.shape[1], frame.shape[0])
                target = paths.frame(paths.full_dir, index)
                if not cv2.imwrite(str(target), frame):
                    raise OSError(f"Unable to write frame: {target}")
                frame_paths.append(target)
                index += 1
        finally:
            capture.release()
        if not frame_paths:
            raise ValueError(f"Video contains no readable frames: {source}")
        return frame_paths, size

    def _read_image(
        self, image_path: Path, paths: ActionPaths
    ) -> tuple[list[Path], tuple[int, int]]:
        source = Path(image_path)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {source}")
        image = self._ensure_even_frame(image)
        self._reset_dir(paths.full_dir)
        target = paths.frame(paths.full_dir, 0)
        if not cv2.imwrite(str(target), image):
            raise OSError(f"Unable to write frame: {target}")
        return [target], (image.shape[1], image.shape[0])

    def _pad_frames(self, paths: ActionPaths, frame_paths: list[Path]) -> list[Path]:
        """MuseTalk needs image context: repeat the first frame up to batch_size."""
        if not bool(self.pre.get("pad_to_batch_size", True)):
            return frame_paths
        minimum = max(1, int(self.pre.get("batch_size", 1)))
        if len(frame_paths) >= minimum:
            return frame_paths
        first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f"Unable to read frame for context padding: {frame_paths[0]}")
        padded = list(frame_paths)
        for index in range(len(padded), minimum):
            target = paths.frame(paths.full_dir, index)
            if not cv2.imwrite(str(target), first):
                raise OSError(f"Unable to write padded frame: {target}")
            padded.append(target)
        logger.info("padded short asset: frames=%d -> %d", len(frame_paths), len(padded))
        return padded

    # ------------------------------------------------------------------ detect
    def _detect_boxes(
        self, frame_paths: list[Path]
    ) -> tuple[list[tuple[int, int, int, int]], int]:
        batch = max(1, int(self.pre.get("batch_size", 16)))
        raw: list[tuple[int, int, int, int] | None] = []
        missed = 0
        for start in range(0, len(frame_paths), batch):
            frames = []
            for path in frame_paths[start : start + batch]:
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Unable to read extracted frame: {path}")
                frames.append(frame)
            batch_boxes = self.detector.raw_boxes(frames)
            missed += sum(1 for box in batch_boxes if box is None)
            raw.extend(batch_boxes)
        boxes = resolve_boxes(
            raw,
            smoothing_window=int(self.pre.get("box_smoothing_window", 0)),
            max_missed_ratio=float(self.pre.get("max_missed_ratio", 0.2)),
        )
        return boxes, missed

    # ----------------------------------------------------------------- prepare
    def _prepare(
        self, paths: ActionPaths, frame_paths: list[Path], assets: tuple[str, ...]
    ) -> tuple[int, int]:
        need_mask = MASK_DIR in assets
        need_latents = LATENTS_FILE in assets
        if need_mask:
            # Build the parser up front so a broken model fails before writing frames.
            _ = self.masker

        boxes, missed = self._detect_boxes(frame_paths)
        self._reset_dir(paths.face_dir)
        if need_mask:
            self._reset_dir(paths.mask_dir)

        coords: list[tuple[int, int, int, int]] = []
        mask_coords: list[tuple[int, int, int, int]] = []
        latents: list[torch.Tensor] = []
        for index, (path, box) in enumerate(zip(frame_paths, boxes)):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Unable to read extracted frame: {path}")
            y1, y2, x1, x2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError(f"face crop is empty for frame {index}")
            crop = cv2.resize(
                crop,
                (self.output_shape, self.output_shape),
                interpolation=cv2.INTER_LANCZOS4,
            )
            face_path = paths.frame(paths.face_dir, index)
            if not cv2.imwrite(str(face_path), crop):
                raise OSError(f"Unable to write face frame: {face_path}")

            if need_mask:
                mask, crop_box = self.masker.build(frame, box)
                mask_path = paths.frame(paths.mask_dir, index)
                if not cv2.imwrite(str(mask_path), mask):
                    raise OSError(f"Unable to write blend mask: {mask_path}")
                mask_coords.append(tuple(int(value) for value in crop_box))
            if need_latents:
                latents.append(self.vae.get_latents_for_unet(crop).cpu())

            coords.append((int(y1), int(y2), int(x1), int(x2)))
            logger.debug("preprocessed frame %d/%d", index + 1, len(frame_paths))

        with paths.coords.open("wb") as stream:
            pickle.dump(coords, stream)
        if need_mask:
            with paths.mask_coords.open("wb") as stream:
                pickle.dump(mask_coords, stream)
        if need_latents:
            torch.save(torch.cat(latents, dim=0), paths.latents)
        return len(coords), missed

    # ------------------------------------------------------------------- public
    def preprocess(
        self, source: Path, action_dir: Path, *, is_image: bool | None = None
    ) -> PreprocessResult:
        source = Path(source)
        if is_image is None:
            extensions = {
                str(value).lower()
                for value in (self.pre.get("video_extensions") or [])
            }
            is_image = source.suffix.lower() not in extensions

        paths = ActionPaths(
            root=Path(action_dir), frame_name_width=self.frame_name_width
        ).create()
        logger.info(
            "preprocessing started: source=%s action=%s image=%s", source, paths.root, is_image
        )
        if is_image:
            frame_paths, size = self._read_image(source, paths)
        else:
            frame_paths, size = self._extract_frames(source, paths)
        frame_paths = self._pad_frames(paths, frame_paths)
        assets = assets_for(paths.root.name)
        logger.info(
            "engine %s assets: %s", paths.root.name, ", ".join(assets)
        )
        face_count, missed = self._prepare(paths, frame_paths, assets)
        if bool(self.pre.get("write_metadata", True)):
            self._write_metadata(
                paths, source, size, frame_paths, face_count, missed, is_image, assets
            )
        logger.info(
            "preprocessing finished: frames=%d faces=%d missed=%d action=%s",
            len(frame_paths),
            face_count,
            missed,
            paths.root,
        )
        return PreprocessResult(
            root=paths.root,
            frame_count=len(frame_paths),
            face_count=face_count,
            missed_frames=missed,
            output_shape=self.output_shape,
        )

    def preprocess_video(self, video_path: Path, action_dir: Path) -> PreprocessResult:
        return self.preprocess(video_path, action_dir, is_image=False)

    def preprocess_image(self, image_path: Path, action_dir: Path) -> PreprocessResult:
        return self.preprocess(image_path, action_dir, is_image=True)

    def _write_metadata(
        self,
        paths: ActionPaths,
        source: Path,
        size: tuple[int, int],
        frame_paths: list[Path],
        face_count: int,
        missed: int,
        is_image: bool,
        assets: tuple[str, ...],
    ) -> None:
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "source_type": "image" if is_image else "video",
            "engine": paths.root.name,
            "assets": list(assets),
            "frame_count": len(frame_paths),
            "face_count": face_count,
            "missed_frames": missed,
            "source_size": {"width": size[0], "height": size[1]},
            "output_shape": self.output_shape,
            "landmarks": self.pre.get("landmarks", "mediapipe"),
            "detector": self.detector.name,
            "device": self.device,
            "batch_size": int(self.pre.get("batch_size", 16)),
            "box_smoothing_window": int(self.pre.get("box_smoothing_window", 0)),
            "coords_order": "(y1, y2, x1, x2)",
            "mask_coords_order": "(x1, y1, x2, y2)",
            "blend_mask": dict(self.pre.get("blend_mask", {}) or {}),
            "artifacts": {
                "full_imgs": len(frame_paths),
                "face_imgs": face_count,
                "coords": paths.coords.name,
                **(
                    {"mask": face_count, "mask_coords": paths.mask_coords.name}
                    if MASK_DIR in assets
                    else {}
                ),
                **({"latents": paths.latents.name} if LATENTS_FILE in assets else {}),
            },
        }
        paths.metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )


_default: AvatarPreprocessor | None = None


def default_preprocessor() -> AvatarPreprocessor:
    global _default
    if _default is None:
        _default = AvatarPreprocessor()
    return _default


def preprocess_action(
    source: Path, action_dir: Path, *, is_image: bool | None = None
) -> PreprocessResult:
    return default_preprocessor().preprocess(source, action_dir, is_image=is_image)


def preprocess_avatar_video(video_path: Path, action_dir: Path) -> int:
    return default_preprocessor().preprocess_video(Path(video_path), Path(action_dir)).frame_count


def preprocess_avatar_image(image_path: Path, action_dir: Path) -> int:
    return default_preprocessor().preprocess_image(Path(image_path), Path(action_dir)).frame_count


def create_action(
    source: Path,
    *,
    avatar_root: Path | str | None = None,
    index: int | None = None,
    avatar_id: str | None = None,
    behavior: str = "IDLE",
    action: str = "DEFAULT",
    engine: str | None = None,
    is_image: bool | None = None,
) -> PreprocessResult:
    """Resolve the resource path, then preprocess one source into that action."""
    preprocessor = default_preprocessor()
    config = preprocessor.config
    if avatar_root is None:
        avatar_root = (config.get("paths", {}) or {}).get("avatar_root", "avatar")
    root = preprocessor.resolve_path(avatar_root)
    root.mkdir(parents=True, exist_ok=True)
    if index is None:
        index = allocate_index(root)
    if avatar_id is None:
        avatar_id = str(uuid4())
    if engine is None:
        engine = (config.get("digitalhuman", {}) or {}).get("lip_sync_engine") or "musetalkv15"

    paths = build_action_dir(
        root,
        index,
        avatar_id,
        behavior,
        action,
        engine,
        frame_name_width=preprocessor.frame_name_width,
    )
    return preprocessor.preprocess(source, paths.root, is_image=is_image)
