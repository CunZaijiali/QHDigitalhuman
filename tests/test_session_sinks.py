import unittest

import numpy as np

from src.session.session import _audio_packets
from src.session.sinks import CompositeSink, VirtualCameraSink


class AudioPacketTest(unittest.TestCase):
    def test_pads_a_tail_that_does_not_divide_evenly(self):
        # 1001 samples -> 4 packets of 320 (1280), i.e. a silent tail of 279.
        # Regression: TTS output is an arbitrary length and reshape(-1, 320) raises.
        packets = _audio_packets(np.arange(1001, dtype=np.float32), 320)
        self.assertEqual(packets.shape, (4, 320))
        self.assertTrue(np.all(packets[-1][279:] == 0))

    def test_exact_multiple_is_untouched(self):
        packets = _audio_packets(np.arange(640, dtype=np.float32), 320)
        self.assertEqual(packets.shape, (2, 320))
        self.assertTrue(np.all(packets[1] == np.arange(320, 640, dtype=np.float32)))

    def test_shorter_than_one_packet_is_padded_to_a_full_packet(self):
        # The sink contract is a fixed 20 ms packet, so a short tail is padded.
        packets = _audio_packets(np.arange(100, dtype=np.float32), 320)
        self.assertEqual(packets.shape, (1, 320))
        self.assertTrue(np.all(packets[0][100:] == 0))


class _FakeCamera:
    def __init__(self):
        self.frames = []

    def send(self, frame):
        self.frames.append(frame)

    def close(self):
        return None


class VirtualCameraSinkTest(unittest.TestCase):
    def test_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            VirtualCameraSink(backend="magic")

    def test_prepare_resizes_and_converts_bgr_to_rgb(self):
        sink = VirtualCameraSink(backend="obs")
        sink.size = (4, 4)
        frame = np.zeros((8, 8, 3), np.uint8)
        frame[..., 0] = 10   # blue in BGR
        frame[..., 2] = 200  # red in BGR
        out = sink._prepare(frame)
        self.assertEqual(out.shape, (4, 4, 3))
        self.assertTrue(out.flags["C_CONTIGUOUS"])
        self.assertEqual(int(out[0, 0, 0]), 200)  # red first in RGB
        self.assertEqual(int(out[0, 0, 2]), 10)
        self.assertEqual(sink.video_resized, 1)

    def test_pushing_to_a_closed_sink_raises_instead_of_using_none(self):
        # Regression: stop() nulls the camera while the producer thread may still
        # be sending, which surfaced as "NoneType has no attribute send".
        sink = VirtualCameraSink(backend="obs")
        sink._closed = True
        with self.assertRaises(RuntimeError):
            sink.push_video_frame(np.zeros((4, 4, 3), np.uint8))

    def test_video_only_sink_accepts_no_audio(self):
        self.assertFalse(VirtualCameraSink(backend="obs").accepts_audio)


class CompositeSinkTest(unittest.TestCase):
    def test_audio_only_reaches_sinks_that_accept_it(self):
        class AudioSink:
            name = "audio"
            accepts_audio = True

            def __init__(self):
                self.received = 0

            def start(self):
                return None

            def push_video_frame(self, frame):
                return 0

            def push_audio_frame(self, samples):
                self.received += 1

            def wait_ready(self, timeout=None):
                return True

            @property
            def stopped(self):
                return False

            def stats(self):
                return {"sink": self.name}

            def stop(self):
                return None

        audio = AudioSink()
        camera = VirtualCameraSink(backend="obs")
        composite = CompositeSink([audio, camera])
        composite.push_audio_frame(np.zeros(320, np.float32))
        self.assertEqual(audio.received, 1)
        self.assertTrue(composite.accepts_audio)

    def test_a_failing_sink_is_dropped_not_fatal(self):
        class Broken:
            name = "broken"
            accepts_audio = False

            def start(self):
                return None

            def push_video_frame(self, frame):
                raise RuntimeError("device gone")

            def push_audio_frame(self, samples):
                return None

            def wait_ready(self, timeout=None):
                return True

            @property
            def stopped(self):
                return False

            def stats(self):
                return {"sink": self.name}

            def stop(self):
                return None

        class Working(Broken):
            name = "working"

            def __init__(self):
                self.count = 0

            def push_video_frame(self, frame):
                self.count += 1
                return self.count

        working = Working()
        composite = CompositeSink([Broken(), working])
        for _ in range(3):
            composite.push_video_frame(np.zeros((4, 4, 3), np.uint8))
        self.assertEqual(working.count, 3)
        self.assertEqual(composite.stats()["failed_sinks"], ["broken"])


if __name__ == "__main__":
    unittest.main()
