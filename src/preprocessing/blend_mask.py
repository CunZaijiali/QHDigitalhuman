"""MuseTalk blending mask, ported from the reference ``blending.py``.

``build()`` returns the mask over the *expanded* crop region plus that crop box
in source-frame pixels, so the pair can be pasted back onto the source frame.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

FaceBox = tuple[int, int, int, int]


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    seg_image = fp(image, mode=mode)
    if seg_image is None:
        return None
    return seg_image.resize(image.size)


def get_image_prepare_material(
    image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw", blur_ratio=0.1
):
    """Return (mask over the expanded crop, crop_box as (x, y, x, y)).

    ``face_box`` is ``(x, y, x, y)`` in source-frame pixels.
    """
    body = Image.fromarray(image[:, :, ::-1])

    x, y, x1, y1 = face_box
    crop_box, _ = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    mask_image = Image.new("L", ori_shape, 0)
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    # keep upper_boundary_ratio of the talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new("L", ori_shape, 0)
    modified_mask_image.paste(
        mask_image.crop((0, top_boundary, width, height)), (0, top_boundary)
    )

    blur_kernel_size = int(float(blur_ratio) * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(
        np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0
    )
    return mask_array, crop_box


class FaceBlendMasker:
    """Holds one FaceParsing instance; rebuilding it per frame is the old bug."""

    def __init__(
        self,
        parser,
        *,
        mode: str = "jaw",
        expand: float = 1.5,
        upper_boundary_ratio: float = 0.5,
        blur_ratio: float = 0.1,
    ):
        self._parser = parser
        self.mode = mode
        self.expand = float(expand)
        self.upper_boundary_ratio = float(upper_boundary_ratio)
        self.blur_ratio = float(blur_ratio)

    def build(
        self, frame: np.ndarray, box: FaceBox
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Return (mask, crop_box) where crop_box is (x1, y1, x2, y2) in source pixels."""
        y1, y2, x1, x2 = (int(value) for value in box)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid face box: {box}")
        mask, crop_box = get_image_prepare_material(
            frame,
            (x1, y1, x2, y2),
            upper_boundary_ratio=self.upper_boundary_ratio,
            expand=self.expand,
            fp=self._parser,
            mode=self.mode,
            blur_ratio=self.blur_ratio,
        )
        x_s, y_s, x_e, y_e = crop_box
        return mask, (x_s, y_s, x_e, y_e)
