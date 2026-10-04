"""Avatar resource catalogue.

目录约定由 :mod:`src.storage` 统一提供::

    avatar/<index>_<uuid4>/<BEHAVIOR>/<ACTION>/<engine>/

本模块只负责"数字人"这一层的定位与创建，不重复实现预处理。
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Literal

import numpy as np

from src.domain.state import parse_behavior_action
from src.preprocessing import default_preprocessor
from src.storage import allocate_index, build_action_dir
from src.utils import find_project_root
from src.utils.files import remove_path, save_image


def avatar_root() -> Path:
    """Resource root; single source of truth is [paths].avatar_root in config.toml."""
    preprocessor = default_preprocessor()
    root = preprocessor.resolve_path(
        (preprocessor.config.get("paths", {}) or {}).get("avatar_root", "avatar")
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_all_avatars(show=True):
    avatars = sorted(p.name for p in avatar_root().glob("*/"))
    if show:
        print(avatars)
    return avatars


def delete_avatar(index, avatar_id, hard=False):
    """Remove the whole <index>_<uuid4> resource root."""
    root = Path(avatar_root()) / f"{index}_{avatar_id}"
    if not root.is_dir():
        return False
    if hard:
        remove_path(root)
    return True


def __delete_all_avatars():
    """调试用"""
    remove_path(avatar_root())


class Avatar:
    """
    使用 avatar_id 作为唯一索引以及资源根目录
    目录名：<index>_<avatar_id>
    avatar_id 创建时自动生成 uuid4

    以 <Behavior>.<Action> 作为数字人动作资产键值
    """

    def __init__(self):
        self.avatar_id = None
        self.name = None
        self.lip_sync_engine: Literal["wav2lip", "musetalkv15"] = "musetalkv15"
        self.tts_engine: Literal["volcengine"] = "volcengine"
        self.resource = None

    @classmethod
    def new(cls, avatar_id: str, name: str, source: str):
        instance = cls()
        instance.avatar_id = avatar_id
        instance.name = name
        instance.resource = source
        return instance

    @staticmethod
    async def create_from_image(
        name: str | None = None, action_key: str | None = None, image: np.ndarray | None = None
    ) -> "Avatar":
        behavior, action_name = parse_behavior_action(action_key or "IDLE.DEFAULT")
        preprocessor = default_preprocessor()
        engine = (
            preprocessor.config.get("digitalhuman", {}) or {}
        ).get("lip_sync_engine") or "musetalkv15"

        root = avatar_root()
        index = allocate_index(root)
        avatar_id = str(uuid.uuid4())
        paths = build_action_dir(
            root,
            index,
            avatar_id,
            behavior.value,
            action_name,
            engine,
            frame_name_width=preprocessor.frame_name_width,
        )
        source_path = save_image(image, paths.root / "source.jpg")
        try:
            await asyncio.to_thread(preprocessor.preprocess_image, source_path, paths.root)
        except Exception:
            delete_avatar(index, avatar_id, hard=True)
            raise
        return Avatar.new(f"{index}_{avatar_id}", name, str(paths.root))
