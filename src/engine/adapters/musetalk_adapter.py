"""Adapter: loaded MuseTalk models -> the engine's ``infer`` contract.

Ported from the reference ``MuseReal._infer``: UNet forward over the cached
latents with the positional-encoded Whisper prompts, then VAE decode.
"""

from __future__ import annotations

import numpy as np
import torch

MODEL_FACE_SIZE = 256
# The VAE decoder is the render bottleneck, not the UNet. Measured on an 8 GB
# RTX 4060 (batch of latents -> decoded frames): B=1 costs 40 ms/frame, B=16
# costs 54 ms/frame, and B>=32 falls off a cliff (~2 fps) as VRAM runs out.
# Decoding in slices keeps the small-batch efficiency without the cliff; the VAE
# decoder is per-sample (GroupNorm), so slicing is numerically equivalent.
VAE_DECODE_SLICE = 1


class MuseTalkAdapter:
    """Stateless: one call is one batch of latents -> one batch of BGR faces."""

    name = "musetalk"

    def __init__(self, models):
        self.vae = models.vae
        self.unet = models.unet
        self.device = models.unet.device
        self.weight_dtype = next(models.unet.model.parameters()).dtype
        self.positional_encoding = models.positional_encoding.to(
            self.device, dtype=self.weight_dtype
        ).eval()
        self._timesteps = torch.zeros(1, device=self.device, dtype=torch.long)

    @torch.inference_mode()
    def infer(self, faces: np.ndarray, features: np.ndarray, latents: np.ndarray) -> np.ndarray:
        latent_batch = torch.as_tensor(
            np.asarray(latents), device=self.device, dtype=self.weight_dtype
        )
        if latent_batch.ndim != 4 or latent_batch.shape[1] != 8:
            raise ValueError(
                f"latents must have shape [batch, 8, 32, 32], got {tuple(latent_batch.shape)}"
            )
        prompts = self.positional_encoding(
            torch.as_tensor(np.asarray(features), device=self.device, dtype=self.weight_dtype)
        )
        batch = latent_batch.shape[0]
        timesteps = self._timesteps.expand(batch)
        output = self.unet.model(latent_batch, timesteps, encoder_hidden_states=prompts).sample
        output = output / self.vae.scaling_factor

        slices = output.split(VAE_DECODE_SLICE) if VAE_DECODE_SLICE > 0 else (output,)
        decoded = torch.cat(
            [self.vae.vae.decode(part.to(self.vae.vae.dtype)).sample for part in slices], dim=0
        )
        if decoded.ndim != 4 or tuple(decoded.shape[1:]) != (3, MODEL_FACE_SIZE, MODEL_FACE_SIZE):
            raise RuntimeError(f"MuseTalk VAE returned unexpected shape: {tuple(decoded.shape)}")
        decoded = ((decoded / 2) + 0.5).clamp(0, 1)
        decoded = (
            (decoded.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
        )
        return decoded[..., ::-1]


__all__ = ["MuseTalkAdapter"]
