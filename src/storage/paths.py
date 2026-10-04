"""Single source of truth for the avatar resource layout.

avatar/<index>_<uuid4>/<BEHAVIOR>/<ACTION>/<engine>/
    full_imgs/        源帧，偶数尺寸的 BGR PNG
    face_imgs/        output_shape 方形人脸裁剪
    mask/            扩展裁剪区上的融合 mask（尺寸随人脸变化）
    coords.pkl       [(y1, y2, x1, x2)]  人脸框，源图坐标
    mask_coords.pkl  [(x1, y1, x2, y2)]  扩展裁剪框，源图坐标，与 mask/ 配对
    latens.pt        VAE latents [N, 8, 32, 32]
    preprocess.json  输入、模型与参数
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

FACE_DIR = "face_imgs"
FULL_DIR = "full_imgs"
MASK_DIR = "mask"
COORDS_FILE = "coords.pkl"
MASK_COORDS_FILE = "mask_coords.pkl"
LATENTS_FILE = "latens.pt"
METADATA_FILE = "preprocess.json"

IMAGE_DIRS = (FULL_DIR, FACE_DIR, MASK_DIR)

# Which artifacts each engine actually reads at runtime. Preprocessing produces
# only the listed set and Avatar only validates it, so the engine chosen in
# config.toml decides the on-disk contract.
ASSETS_BY_ENGINE: dict[str, tuple[str, ...]] = {
    "musetalkv15": (FULL_DIR, FACE_DIR, MASK_DIR, COORDS_FILE, MASK_COORDS_FILE, LATENTS_FILE),
    "wav2lip": (FULL_DIR, FACE_DIR, COORDS_FILE),
}

# Unknown engines get the smallest set every engine needs.
DEFAULT_ASSETS: tuple[str, ...] = (FULL_DIR, FACE_DIR, COORDS_FILE)


def normalize_engine(engine: str) -> str:
    """Lower-cased engine directory name, stable on case-sensitive filesystems."""
    value = (engine or "").strip().lower()
    if not value:
        raise ValueError("lip_sync_engine must not be empty")
    return value


def assets_for(engine: str) -> tuple[str, ...]:
    """Artifacts ``engine`` reads at runtime; unknown engines get the safe minimum."""
    return ASSETS_BY_ENGINE.get(normalize_engine(engine), DEFAULT_ASSETS)


def frame_name(index: int, width: int = 8) -> str:
    if index < 0:
        raise ValueError("frame index must be >= 0")
    if width < 1:
        raise ValueError("frame_name_width must be >= 1")
    return f"{index:0{width}d}.png"


def resolve_root(root: str | Path, project_root: Path) -> Path:
    """Resolve a configured path against the project root unless it is absolute."""
    path = Path(root).expanduser()
    return path if path.is_absolute() else Path(project_root) / path


def allocate_index(avatar_root: str | Path) -> int:
    """Next free <index> by scanning existing <index>_<uuid> directories."""
    root = Path(avatar_root)
    highest = -1
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            head, separator, _ = child.name.partition("_")
            if separator and head.isdigit():
                highest = max(highest, int(head))
    return highest + 1


@dataclass(frozen=True)
class ActionPaths:
    """Resolved artifact locations for one <behavior>/<action>/<engine>."""

    root: Path
    frame_name_width: int = 8

    @property
    def full_dir(self) -> Path:
        return self.root / FULL_DIR

    @property
    def face_dir(self) -> Path:
        return self.root / FACE_DIR

    @property
    def mask_dir(self) -> Path:
        return self.root / MASK_DIR

    @property
    def coords(self) -> Path:
        return self.root / COORDS_FILE

    @property
    def mask_coords(self) -> Path:
        return self.root / MASK_COORDS_FILE

    @property
    def latents(self) -> Path:
        return self.root / LATENTS_FILE

    @property
    def metadata(self) -> Path:
        return self.root / METADATA_FILE

    def frame(self, directory: Path, index: int) -> Path:
        return Path(directory) / frame_name(index, self.frame_name_width)

    def create(self) -> "ActionPaths":
        """Create only the image directories this engine actually uses."""
        self.root.mkdir(parents=True, exist_ok=True)
        assets = assets_for(self.root.name)
        for directory in (self.full_dir, self.face_dir, self.mask_dir):
            if directory.name in assets:
                directory.mkdir(parents=True, exist_ok=True)
        return self

    def clear_outputs(self) -> None:
        """Remove every generated artifact; not only *.png."""
        for directory in (self.full_dir, self.face_dir, self.mask_dir):
            if not directory.is_dir():
                continue
            for child in directory.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        for path in (self.coords, self.mask_coords, self.latents, self.metadata):
            path.unlink(missing_ok=True)


def build_action_dir(
    avatar_root: str | Path,
    index: int,
    avatar_id: str,
    behavior: str,
    action: str,
    engine: str,
    *,
    frame_name_width: int = 8,
    create: bool = True,
) -> ActionPaths:
    """Build (and by default create) one action directory."""
    if index < 0:
        raise ValueError("avatar index must be >= 0")
    avatar_id = str(avatar_id).strip()
    if not avatar_id:
        raise ValueError("avatar_id must not be empty")
    behavior = (behavior or "").strip().upper()
    action = (action or "").strip().upper()
    if not behavior or not action:
        raise ValueError("behavior and action must not be empty")

    root = (
        Path(avatar_root)
        / f"{index}_{avatar_id}"
        / behavior
        / action
        / normalize_engine(engine)
    )
    paths = ActionPaths(root=root, frame_name_width=frame_name_width)
    return paths.create() if create else paths
