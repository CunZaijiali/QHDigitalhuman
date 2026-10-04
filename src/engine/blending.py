"""MuseTalk-style blending of a generated face back onto its source frame.

``get_image_blending`` is ported verbatim from the reference ``utils/blending.py``:
it pastes the generated ROI through the precomputed jaw mask, using the matching
crop box instead of a hard rectangle. ``blend_generated_region`` is the generic
feathered fallback MuseTalk uses when an action has no ``mask/`` artifacts.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

_mouth_mask_cache: dict[tuple[int, int], np.ndarray] = {}


def get_image_blending(image, face, face_box, mask_array, crop_box):
    """Paste ``face`` into ``image`` through ``mask_array``.

    ``face_box`` is ``(x, y, x, y)`` in source pixels and ``crop_box`` is the
    ``(x, y, x, y)`` expanded crop that ``mask_array`` was built for.

    The reference implementation does this through PIL on the *whole* frame and
    measured 12.8 ms/frame on a 768x1344 asset — the single largest CPU cost in
    the render loop. This version is the same arithmetic (``dst*(1-a) + src*a``
    with a 0-255 alpha) restricted to the crop rectangle, in BGR throughout.
    Crops that run past the frame edge are handled explicitly, replacing PIL's
    silent black padding.
    """
    x, y, x1, y1 = (int(value) for value in face_box)
    x_s, y_s, x_e, y_e = (int(value) for value in crop_box)
    if x_e <= x_s or y_e <= y_s:
        raise ValueError(f"invalid crop box: {crop_box}")

    height, width = image.shape[:2]
    dx1, dy1 = max(0, x_s), max(0, y_s)
    dx2, dy2 = min(width, x_e), min(height, y_e)
    if dx2 <= dx1 or dy2 <= dy1:
        return image.copy()  # the crop lies entirely outside the frame

    result = image.copy()
    dest = result[dy1:dy2, dx1:dx2]

    # Offset of the frame-visible part inside the crop / mask coordinate space.
    ox1, oy1 = dx1 - x_s, dy1 - y_s

    # 1) the crop's own pixels, with the generated face pasted in.
    base = dest.copy()
    fx1, fy1 = x - x_s - ox1, y - y_s - oy1
    fx2, fy2 = x1 - x_s - ox1, y1 - y_s - oy1
    ix1, iy1 = max(0, fx1), max(0, fy1)
    ix2, iy2 = min(dest.shape[1], fx2), min(dest.shape[0], fy2)
    if ix2 > ix1 and iy2 > iy1:
        base[iy1:iy2, ix1:ix2] = face[
            iy1 - fy1 : iy2 - fy1, ix1 - fx1 : ix2 - fx1
        ]

    # 2) blend that back through the mask.
    mask = np.asarray(mask_array)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    ox2, oy2 = ox1 + dest.shape[1], oy1 + dest.shape[0]
    mask_part = mask[oy1:oy2, ox1:ox2]
    if mask_part.shape[:2] != dest.shape[:2]:
        mask_part = cv2.resize(
            mask_part, (dest.shape[1], dest.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    alpha = (mask_part.astype(np.float32) / 255.0)[..., None]
    dest[:] = (
        base.astype(np.float32) * alpha + dest.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    return result


def mouth_blend_mask(height: int, width: int) -> np.ndarray:
    """Feathered lower-face mask, cached per shape."""
    key = (height, width)
    cached = _mouth_mask_cache.get(key)
    if cached is not None:
        return cached

    mask = np.zeros((height, width), dtype=np.float32)
    mask[int(height * 0.50):, :] = 1.0
    blur = max(3, int(min(height, width) * 0.012))
    if blur % 2 == 0:
        blur += 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    maximum = float(mask.max())
    if maximum > 0:
        mask /= maximum
    result = mask[..., None]
    _mouth_mask_cache[key] = result
    return result


def blend_generated_region(base_region, generated_region) -> np.ndarray:
    """Blend generated lips into a source face without a hard rectangular edge."""
    generated = np.asarray(generated_region)
    base = np.asarray(base_region)
    if generated.shape != base.shape:
        raise ValueError(
            f"blend regions must have the same shape, got {generated.shape} and {base.shape}"
        )
    mask = mouth_blend_mask(generated.shape[0], generated.shape[1])
    return np.clip(
        generated.astype(np.float32) * mask + base.astype(np.float32) * (1.0 - mask),
        0,
        255,
    ).astype(np.uint8)


__all__ = ["blend_generated_region", "get_image_blending", "mouth_blend_mask"]
