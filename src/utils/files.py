from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def imgs2memory(imgs_path: str | Path) -> list[np.ndarray]:
    path = Path(imgs_path)
    imgs: list[np.ndarray] = []
    if path.is_file():
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unable to read image: {path}")
        return [frame]
    for img_path in tqdm(sorted(path.iterdir())):
        if not img_path.is_file():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise ValueError(f"Unable to read image: {img_path}")
        imgs.append(frame)
    return imgs


def save_image(image: np.ndarray | Image.Image, path: str | Path) -> Path:
    """Write a BGR ndarray or a PIL image to ``path``, creating parents."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(target)
        return target
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("image is empty")
    if not cv2.imwrite(str(target), array):
        raise OSError(f"Unable to write image: {target}")
    return target


def remove_path(path: str | Path, *, missing_ok: bool = True) -> None:
    target = Path(path)
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=missing_ok)
    elif target.exists() or not missing_ok:
        target.unlink(missing_ok=missing_ok)


def clear_dir(path: str | Path) -> None:
    target = Path(path)
    if not target.is_dir():
        return
    for child in target.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def video2image(video_path, save_path, ext="jpg", max_cut_frame=10000000, ignore_st_size=True):
    """使用 cv 保存图像默认为 BGR"""
    video_path = Path(video_path)
    save_path = Path(save_path)
    if video_path.stat().st_size >= 50 * 1024 * 1024 and not ignore_st_size:
        raise RuntimeWarning("Too large video.")
    if not video_path.exists():
        raise FileNotFoundError(f"Video {video_path} not found.")
    cap = cv2.VideoCapture(str(video_path))
    try:
        count = 0
        save_path.mkdir(parents=True, exist_ok=True)
        while True:
            if count > max_cut_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f"{save_path}/{count:08d}.{ext}", frame)
            count += 1
    finally:
        cap.release()


def copy_stream_to_file(stream: BinaryIO, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as target:
        shutil.copyfileobj(stream, target)
    return path


__all__ = [
    "imgs2memory",
    "save_image",
    "remove_path",
    "clear_dir",
    "video2image",
    "copy_stream_to_file",
]
