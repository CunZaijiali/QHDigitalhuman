"""One avatar action asset, read-only.

Layout (see :mod:`src.storage.paths`)::

    avatar/<index>_<uuid4>/<BEHAVIOR>/<ACTION>/<engine>/
        full_imgs/ face_imgs/ mask/ coords.pkl mask_coords.pkl latens.pt preprocess.json

The two pickles deliberately use different field orders, matching MuseTalk:
``coords`` is ``(y1, y2, x1, x2)`` and ``mask_coords`` is ``(x1, y1, x2, y2)``.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np

from src.storage import (
    COORDS_FILE,
    FACE_DIR,
    FULL_DIR,
    LATENTS_FILE,
    MASK_COORDS_FILE,
    MASK_DIR,
    METADATA_FILE,
    assets_for,
)

_AVATAR_DIR_RE = re.compile(r"^\d+_")


def _frame_number(path: Path) -> int | str:
    return int(path.stem) if path.stem.isdigit() else path.stem


def _pngs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.png"), key=_frame_number)


def _pickle(path: Path) -> list:
    return pickle.loads(path.read_bytes()) if path.is_file() else []


class Avatar:
    """Frames, face crops, blend masks, boxes and latents for one action."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.engine = self.root.name
        self.action = self.root.parent.name
        self.behavior = self.root.parent.parent.name
        self.avatar_key = self.root.parent.parent.parent.name
        self.full_paths = _pngs(self.root / FULL_DIR)
        self.face_paths = _pngs(self.root / FACE_DIR)
        self.mask_paths = _pngs(self.root / MASK_DIR)
        self.coords = [tuple(int(v) for v in box) for box in _pickle(self.root / COORDS_FILE)]
        self.mask_coords = [
            tuple(int(v) for v in box) for box in _pickle(self.root / MASK_COORDS_FILE)
        ]
        self._latents = None
        self._metadata = None
        self._images: dict[str, list[np.ndarray]] = {}

    # ---------------------------------------------------------------- preload
    def preload(
        self, *, full: bool = True, face: bool = True, mask: bool = True
    ) -> "Avatar":
        """Decode images into memory.

        Decoding one 768x1344 PNG per frame costs ~20 ms, which is half of a
        40 ms frame budget at 25 fps. Loading the action into memory is what the
        reference implementation does when a resource enters its action cache.
        """
        if full and self.full_paths:
            self._images["full"] = self._read_all(self.full_paths, "full", cv2.IMREAD_COLOR)
        if face and self.face_paths:
            self._images["face"] = self._read_all(self.face_paths, "face", cv2.IMREAD_COLOR)
        if mask and self.mask_paths:
            self._images["mask"] = self._read_all(
                self.mask_paths, "mask", cv2.IMREAD_GRAYSCALE
            )
        return self

    def release(self) -> None:
        """Drop the in-memory frames; the Avatar stays usable (falls back to disk)."""
        self._images.clear()

    @property
    def preloaded(self) -> bool:
        return bool(self._images)

    @property
    def loaded_bytes(self) -> int:
        total = 0
        for images in self._images.values():
            for image in images:
                total += int(image.nbytes)
        return total

    @staticmethod
    def _read_all(paths, label: str, flags: int) -> list[np.ndarray]:
        images = []
        for path in paths:
            image = cv2.imread(str(path), flags)
            if image is None:
                raise ValueError(f"unable to read {label}: {path}")
            images.append(image)
        return images

    # --------------------------------------------------------------- loading
    @classmethod
    def load(cls, root: Path | str) -> "Avatar":
        """Build an Avatar, validating the minimum render contract."""
        path = Path(root)
        if not path.is_dir():
            raise FileNotFoundError(f"action resource directory does not exist: {path}")
        if not (path / COORDS_FILE).is_file():
            raise FileNotFoundError(f"not an action resource (no {COORDS_FILE}): {path}")
        avatar = cls(path)
        if not avatar.full_paths:
            raise ValueError(f"no frames under {path / FULL_DIR}")
        if not avatar.face_paths:
            raise ValueError(f"no face crops under {path / FACE_DIR}")
        if len(avatar.full_paths) != len(avatar.coords):
            raise ValueError(
                f"frame count {len(avatar.full_paths)} and coords count "
                f"{len(avatar.coords)} differ in {path}"
            )
        return avatar

    # ------------------------------------------------------------- accessors
    @property
    def frame_count(self) -> int:
        return len(self.full_paths)

    @property
    def latents(self):
        if self._latents is None:
            import torch

            path = self.root / LATENTS_FILE
            self._latents = torch.load(path, map_location="cpu") if path.is_file() else None
        return self._latents

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            path = self.root / METADATA_FILE
            self._metadata = (
                json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            )
        return self._metadata

    def frame(self, index: int) -> np.ndarray:
        return self._read(self.full_paths, index, "full")

    def face(self, index: int) -> np.ndarray:
        return self._read(self.face_paths, index, "face")

    def mask(self, index: int) -> np.ndarray:
        return self._read(self.mask_paths, index, "mask", cv2.IMREAD_GRAYSCALE)

    def _read(self, paths, index, label, flags=cv2.IMREAD_COLOR) -> np.ndarray:
        if not paths:
            raise ValueError(f"no {label} images in {self.root}")
        slot = index % len(paths)
        cached = self._images.get(label)
        if cached:
            return cached[slot % len(cached)]
        path = paths[slot]
        image = cv2.imread(str(path), flags)
        if image is None:
            raise ValueError(f"unable to read {label}: {path}")
        return image

    # ------------------------------------------------------------- integrity
    def problems(self) -> list[str]:
        """Missing or inconsistent artifacts, judged by what this engine reads."""
        assets = assets_for(self.engine)
        issues: list[str] = []
        expected = len(self.full_paths)
        latents = self.latents
        counts = {
            FULL_DIR: len(self.full_paths),
            FACE_DIR: len(self.face_paths),
            MASK_DIR: len(self.mask_paths),
            COORDS_FILE: len(self.coords),
            MASK_COORDS_FILE: len(self.mask_coords),
            LATENTS_FILE: len(latents) if latents is not None else 0,
        }
        for name in assets:
            if not counts[name]:
                issues.append(f"{name} is missing (required by engine {self.engine})")
            elif name not in (FULL_DIR, LATENTS_FILE) and counts[name] != expected:
                issues.append(f"{name} count {counts[name]} != full_imgs count {expected}")

        samples = sorted({0, expected // 2, expected - 1}) if expected else []
        for index in samples:
            if index < len(self.coords):
                y1, y2, x1, x2 = self.coords[index]
                if not (0 <= y1 < y2 and 0 <= x1 < x2):
                    issues.append(f"coords[{index}] is not a valid box: {self.coords[index]}")
        if self.mask_paths and self.mask_coords:
            for index in samples:
                if index >= len(self.mask_coords):
                    continue
                height, width = self.mask(index).shape[:2]
                x1, y1, x2, y2 = self.mask_coords[index]
                if (width, height) != (x2 - x1, y2 - y1):
                    issues.append(
                        f"mask[{index}] is {width}x{height} but mask_coords[{index}] "
                        f"describes {x2 - x1}x{y2 - y1}"
                    )
        if self.full_paths:
            height, width = self.frame(0).shape[:2]
            if height % 2 or width % 2:
                issues.append(f"full_imgs are not even-sized: {width}x{height}")
        return issues

    def describe(self) -> dict:
        latents = self.latents
        frame = self.frame(0) if self.full_paths else None
        face = self.face(0) if self.face_paths else None
        return {
            "root": str(self.root),
            "avatar_key": self.avatar_key,
            "behavior": self.behavior,
            "action": self.action,
            "engine": self.engine,
            "counts": {
                "full_imgs": len(self.full_paths),
                "face_imgs": len(self.face_paths),
                "mask": len(self.mask_paths),
                "coords": len(self.coords),
                "mask_coords": len(self.mask_coords),
            },
            "shapes": {
                "full": list(frame.shape) if frame is not None else None,
                "face": list(face.shape) if face is not None else None,
                "latents": list(latents.shape) if latents is not None else None,
            },
            "coords_order": "(y1, y2, x1, x2)",
            "mask_coords_order": "(x1, y1, x2, y2)",
            "required_assets": list(assets_for(self.engine)),
            "metadata": self.metadata or None,
        }


def iter_actions(root: Path | str, *, max_depth: int = 4) -> list[Avatar]:
    """Every action directory under ``root``, which may be an avatar or the avatar root."""
    base = Path(root)
    if not base.is_dir():
        return []
    found: dict[Path, Avatar] = {}
    for depth in range(1, max_depth + 1):
        for coords_path in base.glob("/".join(["*"] * depth) + f"/{COORDS_FILE}"):
            action_dir = coords_path.parent
            found.setdefault(action_dir, Avatar(action_dir))
    return [found[key] for key in sorted(found, key=str)]


def resolve_avatar(
    target: Path | str,
    *,
    behavior: str | None = None,
    action: str | None = None,
    engine: str | None = None,
) -> Avatar:
    """Resolve one action from an action directory, or narrow down an avatar root."""
    path = Path(target)
    if not path.is_dir():
        raise FileNotFoundError(f"not a directory: {path}")
    if (path / COORDS_FILE).is_file():
        return Avatar.load(path)

    known = iter_actions(path)
    candidates = list(known)
    applied: list[str] = []
    if behavior:
        wanted = behavior.strip().upper()
        applied.append(f"behavior={wanted}")
        candidates = [item for item in candidates if item.behavior.upper() == wanted]
    if action:
        wanted = action.strip().upper()
        applied.append(f"action={wanted}")
        candidates = [item for item in candidates if item.action.upper() == wanted]
    if engine:
        wanted = engine.strip().lower()
        applied.append(f"engine={wanted}")
        candidates = [item for item in candidates if item.engine.lower() == wanted]

    scope = f" (filters: {', '.join(applied)})" if applied else ""
    if not candidates:
        listed = "\n  ".join(str(item.root) for item in known) or "(no action found)"
        raise FileNotFoundError(
            f"no matching action under {path}{scope}; available:\n  {listed}"
        )
    if len(candidates) > 1:
        listed = "\n  ".join(str(item.root) for item in candidates)
        raise ValueError(
            f"{len(candidates)} actions match{scope}; narrow it down with "
            f"--behavior/--action/--engine:\n  {listed}"
        )
    return Avatar.load(candidates[0].root)


def list_avatars(root: Path | str) -> list[str]:
    """Names of ``<index>_<uuid4>`` directories under an avatar root."""
    base = Path(root)
    if not base.is_dir():
        return []
    return sorted(
        child.name
        for child in base.iterdir()
        if child.is_dir() and _AVATAR_DIR_RE.match(child.name)
    )


__all__ = [
    "Avatar",
    "iter_actions",
    "list_avatars",
    "resolve_avatar",
]
