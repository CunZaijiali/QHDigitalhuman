"""Offline audio feature extraction used by the lip-sync pipelines."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np


def load_audio(audio: str | Path | np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Read mono float32 audio and resample it when a WAV has another rate."""
    if isinstance(audio, np.ndarray):
        samples = audio
        source_rate = sample_rate
    else:
        with wave.open(str(audio), "rb") as stream:
            source_rate = stream.getframerate()
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            raw = stream.readframes(stream.getnframes())
        if width == 1:
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
        elif width == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
        elif width == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648
        else:
            raise ValueError(f"Unsupported WAV sample width: {width}")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate != sample_rate and samples.size:
        target_size = round(samples.size * sample_rate / source_rate)
        samples = np.interp(
            np.linspace(0, samples.size - 1, target_size),
            np.arange(samples.size),
            samples,
        ).astype(np.float32)
    return np.clip(np.nan_to_num(samples), -1.0, 1.0)


def prompt_stride(fps: int) -> int:
    """Whisper encoder steps per video frame.

    Whisper emits one step per 20 ms, so frame ``i`` (audio time ``i / fps``)
    starts at step ``i * stride`` with ``stride = 1 / (fps * 0.02)``: 2 at 25 fps,
    5 at 10 fps. This must track ``fps`` -- a hardcoded 2 silently made the lips
    move at 40% speed when the pipeline ran at 10 fps.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    return max(1, round(1.0 / (fps * 0.02)))


class MuseTalkASR:
    """Convert audio to MuseTalk Whisper prompt batches."""

    def __init__(
        self,
        whisper: Any,
        feature_extractor: Any,
        sample_rate: int = 16000,
        fps: int = 25,
    ):
        self.whisper = whisper
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.fps = fps

    def extract(self, audio: str | Path | np.ndarray, frame_count: int | None = None) -> np.ndarray:
        import torch

        samples = load_audio(audio, self.sample_rate)
        count = frame_count or max(1, round(samples.size / self.sample_rate * self.fps))
        inputs = self.feature_extractor(
            samples,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=False,
        ).input_features
        device = next(self.whisper.parameters()).device
        dtype = next(self.whisper.parameters()).dtype
        if device.type == "cpu" and dtype == torch.float16:
            self.whisper.float()
            dtype = torch.float32
        with torch.inference_mode():
            hidden_states = self.whisper.encoder(
                inputs.to(device=device, dtype=dtype), output_hidden_states=True
            ).hidden_states
            hidden = torch.stack(hidden_states, dim=2)[0]
        prompt_size = 10
        stride = prompt_stride(self.fps)
        # The UNet was trained on a *symmetric* window: MuseTalk's training code
        # pads the encoder output with ``2 * audio_padding_length_left`` zero steps
        # on the left before slicing ``[frame * stride, +10)``, so frame n sees
        # original steps ``[2n - 4, 2n + 6)``. Slicing without that pad (which is
        # what this used to do) hands the model a window that is entirely future-
        # biased, and it answers with a barely-moving mouth.
        pad = 2 * stride
        hidden = torch.cat(
            [
                hidden.new_zeros((pad, hidden.shape[1], hidden.shape[2])),
                hidden,
                hidden.new_zeros((pad, hidden.shape[1], hidden.shape[2])),
            ],
            dim=0,
        )
        prompts = []
        for index in range(count):
            start = index * stride
            clip = hidden[start : start + prompt_size]
            if clip.shape[0] < prompt_size:
                clip = torch.nn.functional.pad(clip, (0, 0, 0, 0, 0, prompt_size - clip.shape[0]))
            prompts.append(clip.reshape(prompt_size * 5, 384))
        return torch.stack(prompts).cpu().numpy()


class Wav2LipASR:
    """Convert audio to the 80x16 Mel windows consumed by Wav2Lip."""

    def __init__(self, sample_rate: int = 16000, fps: int = 25):
        self.sample_rate = sample_rate
        self.fps = fps

    def extract(self, audio: str | Path | np.ndarray, frame_count: int | None = None) -> np.ndarray:
        samples = load_audio(audio, self.sample_rate)
        mel = _melspectrogram(samples, self.sample_rate)
        count = frame_count or max(1, round(samples.size / self.sample_rate * self.fps))
        multiplier = 80.0 * 2 / self.fps
        chunks = []
        for index in range(count):
            start = int(index * multiplier)
            chunk = mel[:, start : start + 16]
            if chunk.shape[1] < 16:
                if chunk.shape[1] == 0:
                    chunk = np.zeros((mel.shape[0], 16), dtype=np.float32)
                else:
                    chunk = np.pad(chunk, ((0, 0), (0, 16 - chunk.shape[1])), mode="edge")
            chunks.append(chunk)
        return np.asarray(chunks, dtype=np.float32)


def _melspectrogram(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    n_fft, hop, win, mel_count = 800, 200, 800, 80
    samples = np.concatenate(([samples[0] if samples.size else 0], samples[1:] - 0.97 * samples[:-1]))
    pad = win // 2
    samples = np.pad(samples, (pad, pad), mode="reflect")
    frames = 1 + max(0, (len(samples) - win) // hop)
    window = np.hanning(win).astype(np.float32)
    spectrum = np.abs(np.stack([
        np.fft.rfft(samples[index * hop : index * hop + win] * window, n=n_fft)
        for index in range(frames)
    ], axis=1))
    mel_filter = _mel_filterbank(mel_count, n_fft, sample_rate, 55, 7600)
    values = 20 * np.log10(np.maximum(np.exp(-100 / 20 * np.log(10)), mel_filter @ spectrum)) - 20
    return np.clip(8 * ((values + 100) / 100) - 4, -4, 4).astype(np.float32)


def _mel_filterbank(count: int, n_fft: int, sample_rate: int, minimum: int, maximum: int) -> np.ndarray:
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    points = np.floor(
        (n_fft + 1) * mel_to_hz(np.linspace(hz_to_mel(minimum), hz_to_mel(maximum), count + 2)) / sample_rate
    ).astype(int)
    filters = np.zeros((count, n_fft // 2 + 1), dtype=np.float32)
    for index in range(count):
        left, center, right = points[index : index + 3]
        if center > left:
            filters[index, left:center] = np.arange(left, center) / (center - left)
        if right > center:
            filters[index, center:right] = np.arange(right - center, 0, -1) / (right - center)
    return filters


__all__ = ["MuseTalkASR", "Wav2LipASR", "load_audio", "prompt_stride"]
