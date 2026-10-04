"""Encode rendered frames to a proper MP4 with FFmpeg.

``render`` used to write video-only MPEG-4 Part 2 through ``cv2.VideoWriter``:
the result was **silent** and both larger and less compatible than H.264.

Here the frames are piped straight into FFmpeg as raw BGR, so there is exactly
one encode. FFmpeg is optional: if it cannot be found the caller falls back to
the old writer and the render still succeeds (just without sound).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

#: Where to look for ffmpeg, in order.
FFMPEG_ENV = "FFMPEG_ROOT"


def find_ffmpeg() -> str | None:
    """Locate an ``ffmpeg`` executable.

    Order: ``$FFMPEG_ROOT/bin/ffmpeg(.exe)``, then ``PATH``. The env var wins
    because this project already documents it for building the RTMP extension.
    """
    root = os.environ.get(FFMPEG_ENV, "").strip()
    if root:
        base = Path(root)
        candidates = [base / "bin" / "ffmpeg", base / "ffmpeg"]
        if sys.platform == "win32":
            candidates = [path.with_suffix(".exe") for path in candidates] + candidates
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return shutil.which("ffmpeg")


def build_encode_command(
    ffmpeg: str,
    output: Path,
    audio: Path,
    *,
    width: int,
    height: int,
    fps: int,
    crf: int = 18,
    preset: str = "medium",
    audio_bitrate: str = "128k",
) -> list[str]:
    """The FFmpeg argv used to turn raw BGR frames on stdin + a WAV into an MP4."""
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        # raw frames arrive on stdin, one BGR24 frame after another
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        # audio and video are the same length by construction; never run past it
        "-shortest",
        # let players start before the whole file is downloaded
        "-movflags",
        "+faststart",
        str(output),
    ]


def _even(frame: np.ndarray) -> np.ndarray:
    """H.264 with yuv420p needs even width and height."""
    height, width = frame.shape[:2]
    return np.ascontiguousarray(frame[: height - height % 2, : width - width % 2])


def encode_with_ffmpeg(
    frames: Iterable[np.ndarray],
    output: str | Path,
    audio_path: str | Path,
    *,
    fps: int,
    crf: int = 18,
    preset: str = "medium",
    ffmpeg: str | None = None,
) -> Path:
    """Pipe ``frames`` into FFmpeg together with ``audio_path``; return the output.

    Raises ``RuntimeError`` on failure so the caller can fall back to cv2.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found")
    audio = Path(audio_path)
    if not audio.is_file():
        raise RuntimeError(f"audio file not found: {audio}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Encode to a sibling temp file, then move it into place: a failure halfway
    # through must not leave a broken file where the real output belongs.
    staging = output.with_name(output.stem + ".encoding.mp4")

    iterator = iter(frames)
    try:
        first = _even(next(iterator))
    except StopIteration as exc:
        raise RuntimeError("no frames were rendered") from exc
    height, width = first.shape[:2]

    command = build_encode_command(
        ffmpeg, staging, audio, width=width, height=height, fps=fps, crf=crf, preset=preset
    )
    logger.info(
        "encoding %dx%d @%dfps with ffmpeg (crf=%d preset=%s)", width, height, fps, crf, preset
    )

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    written = 0
    try:
        assert process.stdin is not None
        for frame in _chain(first, iterator):
            process.stdin.write(frame.tobytes())
            written += 1
        process.stdin.close()
    except BrokenPipeError as exc:
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        process.wait()
        staging.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg stopped early: {stderr.strip()[:400]}") from exc
    except Exception:
        process.kill()
        process.wait()
        staging.unlink(missing_ok=True)
        raise

    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    code = process.wait()
    if code != 0 or not staging.is_file():
        staging.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg exited with {code}: {stderr.strip()[:400]}")

    staging.replace(output)
    logger.info("wrote %s (%d frames, H.264 + AAC)", output, written)
    return output


def _chain(first: np.ndarray, rest: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    yield first
    for frame in rest:
        yield _even(frame)


__all__ = ["FFMPEG_ENV", "build_encode_command", "encode_with_ffmpeg", "find_ffmpeg"]
