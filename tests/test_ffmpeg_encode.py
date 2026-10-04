"""The FFmpeg-based render path: real MP4s with sound."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.utils.ffmpeg import (
    FFMPEG_ENV,
    build_encode_command,
    encode_with_ffmpeg,
    find_ffmpeg,
)


class FindFFmpegTest(unittest.TestCase):
    def test_prefers_ffmpeg_root_over_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "bin" / "ffmpeg.exe"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"")
            with patch.dict(os.environ, {FFMPEG_ENV: tmp}):
                self.assertEqual(find_ffmpeg(), str(binary))

    def test_returns_none_when_nothing_is_found(self):
        with patch.dict(os.environ, {FFMPEG_ENV: ""}):
            with patch("shutil.which", return_value=None):
                self.assertIsNone(find_ffmpeg())

    def test_falls_back_to_path(self):
        with patch.dict(os.environ, {FFMPEG_ENV: ""}):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                self.assertEqual(find_ffmpeg(), "/usr/bin/ffmpeg")


class BuildCommandTest(unittest.TestCase):
    def test_carries_resolution_audio_and_quality(self):
        command = build_encode_command(
            "ffmpeg",
            Path("out.mp4"),
            Path("a.wav"),
            width=768,
            height=1344,
            fps=25,
            crf=18,
            preset="medium",
        )
        joined = " ".join(command)
        # raw BGR frames on stdin at the right geometry
        self.assertIn("-f rawvideo", joined)
        self.assertIn("-pix_fmt bgr24", joined)
        self.assertIn("-s 768x1344", joined)
        self.assertIn("-r 25", joined)
        # audio from the wav, video from stdin
        self.assertIn("-map 0:v:0 -map 1:a:0", joined)
        self.assertIn("-c:v libx264", joined)
        self.assertIn("-crf 18", joined)
        self.assertIn("-c:a aac", joined)
        self.assertIn("-shortest", joined)

    def test_output_is_last_argument(self):
        command = build_encode_command(
            "ffmpeg", Path("o.mp4"), Path("a.wav"), width=2, height=2, fps=25
        )
        self.assertEqual(command[-1], "o.mp4")


class _StubProcess:
    """Pretends to be ffmpeg: consumes stdin and creates the staging file."""

    def __init__(self, output, fail=False):
        self.output = output
        self.fail = fail
        self.stdin = _StubStdin(output, fail)
        self.stderr = _StubStderr(b"boom" if fail else b"")
        self.returncode = None

    def wait(self, timeout=None):
        self.returncode = 1 if self.fail else 0
        return self.returncode

    def kill(self):
        return None


class _StubStderr:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload


class _StubStdin:
    def __init__(self, output, fail):
        self.output = output
        self.fail = fail
        self.frames = 0

    def write(self, data):
        if self.fail:
            raise BrokenPipeError("ffmpeg went away")
        self.frames += 1

    def close(self):
        if not self.fail:
            self.output.write_bytes(b"stub-mp4")


class EncodeTest(unittest.TestCase):
    def test_writes_the_staging_file_then_moves_it(self):
        out = Path(tempfile.mkdtemp()) / "clip.mp4"
        audio = out.with_name("a.wav")
        audio.write_bytes(b"wav")

        def fake_popen(command, **kwargs):
            staging = Path(command[-1])
            return _StubProcess(staging)

        frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)]
        with patch("subprocess.Popen", side_effect=fake_popen):
            result = encode_with_ffmpeg(frames, out, audio, fps=25)
        self.assertEqual(result, out)
        self.assertTrue(out.is_file())
        # no leftover staging file
        self.assertEqual(list(out.parent.glob("*encoding*")), [])

    def test_raises_when_ffmpeg_is_missing(self):
        with patch("src.utils.ffmpeg.find_ffmpeg", return_value=None):
            with self.assertRaises(RuntimeError):
                encode_with_ffmpeg(
                    [np.zeros((4, 4, 3), np.uint8)], "o.mp4", "a.wav", fps=25
                )

    def test_raises_when_the_audio_does_not_exist(self):
        with self.assertRaises(RuntimeError):
            encode_with_ffmpeg(
                [np.zeros((4, 4, 3), np.uint8)], "o.mp4", "missing.wav", fps=25, ffmpeg="ffmpeg"
            )

    def test_raises_when_no_frames_arrive(self):
        out = Path(tempfile.mkdtemp()) / "x.mp4"
        audio = out.with_name("a.wav")
        audio.write_bytes(b"wav")
        with self.assertRaises(RuntimeError):
            encode_with_ffmpeg([], out, audio, fps=25, ffmpeg="ffmpeg")

    def test_a_broken_pipe_is_reported_and_leaves_no_partial_file(self):
        out = Path(tempfile.mkdtemp()) / "clip.mp4"
        audio = out.with_name("a.wav")
        audio.write_bytes(b"wav")

        def fake_popen(command, **kwargs):
            staging = Path(command[-1])
            staging.write_bytes(b"partial")
            return _StubProcess(staging, fail=True)

        with patch("subprocess.Popen", side_effect=fake_popen):
            with self.assertRaises(RuntimeError) as ctx:
                encode_with_ffmpeg(
                    [np.zeros((4, 4, 3), np.uint8)], out, audio, fps=25
                )
        self.assertIn("stopped early", str(ctx.exception))
        self.assertFalse(out.exists())
        self.assertEqual(list(out.parent.glob("*encoding*")), [])

    def test_odd_sized_frames_are_cropped_to_even(self):
        out = Path(tempfile.mkdtemp()) / "clip.mp4"
        audio = out.with_name("a.wav")
        audio.write_bytes(b"wav")
        seen = {}

        def fake_popen(command, **kwargs):
            seen["geometry"] = command[command.index("-s") + 1]
            return _StubProcess(Path(command[-1]))

        with patch("subprocess.Popen", side_effect=fake_popen):
            encode_with_ffmpeg(
                [np.zeros((5, 7, 3), np.uint8)], out, audio, fps=25
            )
        # yuv420p needs even dimensions
        self.assertEqual(seen["geometry"], "6x4")


if __name__ == "__main__":
    unittest.main()
