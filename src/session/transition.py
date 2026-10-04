"""Smoothstep dissolve between the frame we were showing and the next action.

Ported from the reference ``BaseReal._apply_action_transition``: the fade length
is ``round(action_transition_seconds * fps)`` frames and the progress curve is
``p*p*(3 - 2*p)`` (smoothstep), which avoids the brightness step a linear
dissolve shows at both ends.
"""

from __future__ import annotations

import cv2
import numpy as np


class ActionTransition:
    """Holds the fade state for one switch; ``apply`` consumes one frame of it."""

    def __init__(self, seconds: float = 0.2, fps: int = 25):
        self.seconds = max(0.0, float(seconds))
        self.fps = max(1, int(fps))
        self.total = 0
        self.remaining = 0
        self.source: np.ndarray | None = None

    @property
    def frames(self) -> int:
        return 0 if self.seconds <= 0 else max(1, round(self.seconds * self.fps))

    @property
    def active(self) -> bool:
        return self.remaining > 0 and self.source is not None

    def arm(self, last_frame: np.ndarray | None) -> None:
        """Start a fade from ``last_frame``; a missing frame means no fade."""
        total = self.frames
        if total <= 0 or last_frame is None:
            self.source = None
            self.remaining = 0
            self.total = 0
            return
        self.source = np.asarray(last_frame).copy()
        self.total = total
        self.remaining = total

    def cancel(self) -> None:
        self.source = None
        self.remaining = 0
        self.total = 0

    def apply(self, target_frame: np.ndarray) -> np.ndarray:
        target = np.asarray(target_frame)
        if not self.active:
            return target
        source = self.source
        if source.shape != target.shape:
            source = cv2.resize(
                source, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        progress = 1.0 - (self.remaining - 1) / max(1, self.total)
        progress = progress * progress * (3.0 - 2.0 * progress)
        blended = (
            source.astype(np.float32) * (1.0 - progress) + target.astype(np.float32) * progress
        ).clip(0, 255).astype(np.uint8)
        self.remaining -= 1
        if self.remaining <= 0:
            self.cancel()
        return blended


__all__ = ["ActionTransition"]
