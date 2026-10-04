"""Adapter: loaded Wav2Lip model -> the engine's ``infer`` contract.

Ported from the reference ``LipReal.inference``: the model takes six channels
(the lower half of the face zeroed out, then the reference face) plus an
``(B, 1, 80, 16)`` mel window.
"""

from __future__ import annotations

import numpy as np
import torch

MEL_BINS = 80
MEL_STEPS = 16


class Wav2LipAdapter:
    """Stateless: one call is one batch of mel windows + faces -> BGR faces."""

    name = "wav2lip"

    def __init__(self, model, device: str | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device

    @torch.inference_mode()
    def infer(self, mel_batch: np.ndarray, images: np.ndarray) -> np.ndarray:
        mel = np.asarray(mel_batch, dtype=np.float32)
        if mel.ndim != 3 or mel.shape[1:] != (MEL_BINS, MEL_STEPS):
            raise ValueError(
                f"mel batch must have shape [batch, {MEL_BINS}, {MEL_STEPS}], got {mel.shape}"
            )
        # The engine hands over faces scaled to [0, 1]; Wav2Lip wants byte-range BGR.
        faces = np.clip(np.asarray(images, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
        if faces.ndim != 4 or faces.shape[-1] != 3:
            raise ValueError(f"face batch must have shape [batch, H, W, 3], got {faces.shape}")

        batch, height = faces.shape[0], faces.shape[1]
        occluded = faces.copy()
        occluded[:, height // 2 :, :] = 0
        pair = np.concatenate([occluded, faces], axis=3) / 255.0
        mel = mel.reshape(batch, mel.shape[1], mel.shape[2], 1)

        image_tensor = torch.from_numpy(np.transpose(pair, (0, 3, 1, 2))).float().to(self.device)
        mel_tensor = torch.from_numpy(np.transpose(mel, (0, 3, 1, 2))).float().to(self.device)
        output = self.model(mel_tensor, image_tensor)
        output = output.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)


__all__ = ["Wav2LipAdapter"]
