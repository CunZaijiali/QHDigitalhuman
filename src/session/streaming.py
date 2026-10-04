"""Push rendered frames and PCM to an RTMP server.

Ported from the reference ``domain/streaming.py::RTMPOutput`` with the
server-side plumbing removed: no ``IMetaInfo`` / ``IPlayer`` / aiortc, and no
local MP4 recorder. What remains is the part that matters:

* a bounded, drop-oldest video queue (a slow network must never block rendering),
* a lossless audio buffer that reports backpressure instead of dropping samples,
* two pacing threads — video at ``fps``, audio at one 20 ms packet per tick.

The extension wants **RGB24**, so frames arrive here as BGR and are converted
before being handed to ``stream_frame``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

AUDIO_PACKET_HZ = 50  # the extension consumes one packet per 20 ms
AUDIO_PACKET_SECONDS = 1 / AUDIO_PACKET_HZ


class RTMPOutput:
    """Push frames and audio to an RTMP endpoint over the vendored C++ streamer."""

    name = "rtmp"
    accepts_audio = True

    """Bounded frame cache plus a ``rtmp_streaming`` publisher."""

    def __init__(
        self,
        push_url: str,
        *,
        fps: int = 25,
        bitrate: int = 1_600_000,
        sample_rate: int = 16000,
        profile: str = "main",
        # x264 tuning, from [digitalhuman]. Empty means "leave it to FFmpeg".
        # The upstream vendor hardcoded ultrafast+zerolatency, which measured
        # 44.0 dB PSNR where medium + rc-lookahead=4:bframes=1 reaches 48.8 dB at
        # the same bitrate (and costs no more than ~8 ms/frame here).
        x264_preset: str = "",
        x264_tune: str = "",
        x264_params: str = "",
        # The producer renders in batches, so a whole batch can arrive within a
        # few milliseconds while the sink takes one frame per 1/fps. A queue
        # smaller than the batch would drop most of every batch; drop-oldest is
        # meant for network stalls, not for batched production.
        video_queue_size: int = 32,
        audio_capacity_packets: int = 250,
    ):
        if not push_url:
            raise ValueError("push_url is required")
        self.push_url = push_url
        self.fps = int(fps)
        self.bitrate = int(bitrate)
        self.sample_rate = int(sample_rate)
        self.profile = profile
        self.x264_preset = x264_preset or ""
        self.x264_tune = x264_tune or ""
        self.x264_params = x264_params or ""
        self.packet_size = max(1, self.sample_rate // AUDIO_PACKET_HZ)
        self._max_audio_samples = max(1, int(audio_capacity_packets)) * self.packet_size

        self._video_queue: queue.Queue = queue.Queue(maxsize=max(1, int(video_queue_size)))
        self._audio_pending = np.zeros(0, dtype=np.float32)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._audio_thread: threading.Thread | None = None
        self._streamer = None
        self._streamer_ready = threading.Event()
        self._error: str | None = None
        self._lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._audio_lock = threading.Lock()
        self.video_enqueued = self.video_streamed = self.video_dropped = 0
        self.video_repeated = 0
        #: How many output ticks each produced frame is currently held for.
        self.video_hold = 1
        self.audio_enqueued = self.audio_streamed = self.audio_backpressure_events = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ensure_even_frame(frame) -> np.ndarray:
        height, width = frame.shape[:2]
        even_height = height - (height % 2)
        even_width = width - (width % 2)
        if even_height < 2 or even_width < 2:
            raise ValueError(f"RTMP frame is too small: {width}x{height}")
        return np.ascontiguousarray(frame[:even_height, :even_width])

    @staticmethod
    def _put_latest(target: queue.Queue, item) -> int:
        """Drop the oldest entry until ``item`` fits; returns how many were dropped."""
        dropped = 0
        while True:
            try:
                target.put_nowait(item)
                return dropped
            except queue.Full:
                try:
                    target.get_nowait()
                    dropped += 1
                except queue.Empty:
                    pass

    # -------------------------------------------------------------------- public
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rtmp-video", daemon=True)
        self._audio_thread = threading.Thread(
            target=self._audio_loop, name="rtmp-audio", daemon=True
        )
        self._thread.start()
        self._audio_thread.start()

    def push_video_frame(self, frame) -> None:
        """BGR frame; the newest frames win when the queue is saturated."""
        if not isinstance(frame, np.ndarray) or self._stop.is_set():
            return
        normalized = self._ensure_even_frame(frame)
        dropped = self._put_latest(self._video_queue, normalized.copy())
        with self._lock:
            self.video_enqueued += 1
            self.video_dropped += dropped

    def push_audio_frame(self, samples) -> None:
        """Mono float32 PCM; samples are never dropped, overflow is reported."""
        if self._stop.is_set():
            return
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        with self._audio_lock:
            self._audio_pending = np.concatenate((self._audio_pending, block))
            overflow = self._audio_pending.size > self._max_audio_samples
        with self._lock:
            self.audio_enqueued += 1
            if overflow:
                self.audio_backpressure_events += 1

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until the first frame has opened the stream (or time out)."""
        return self._streamer_ready.wait(timeout)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "push_url": self.push_url,
                "fps": self.fps,
                "bitrate": self.bitrate,
                "video_enqueued": self.video_enqueued,
                "video_streamed": self.video_streamed,
                "video_dropped": self.video_dropped,
                "video_repeated": self.video_repeated,
                "video_hold": self.video_hold,
                "video_queue": self._video_queue.qsize(),
                "audio_enqueued": self.audio_enqueued,
                "audio_streamed": self.audio_streamed,
                "audio_pending_samples": int(self._audio_pending.size),
                "audio_backpressure_events": self.audio_backpressure_events,
                "worker_alive": bool(
                    self._thread and self._thread.is_alive()
                    and self._audio_thread and self._audio_thread.is_alive()
                ),
                "error": self._error,
            }

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=5)
        self._streamer_ready.clear()
        self._streamer = None
        with self._audio_lock:
            self._audio_pending = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------- workers
    def _init_streamer(self, frame) -> None:
        from rtmp_streaming import Streamer, StreamerConfig

        frame = self._ensure_even_frame(frame)
        height, width = frame.shape[:2]
        config = StreamerConfig()
        config.source_width = config.stream_width = width
        config.source_height = config.stream_height = height
        config.stream_fps = self.fps
        config.stream_bitrate = self.bitrate
        config.stream_profile = self.profile
        config.stream_preset = self.x264_preset
        config.stream_tune = self.x264_tune
        config.stream_x264_params = self.x264_params
        config.audio_channel = 1
        config.sample_rate = self.sample_rate
        config.stream_server = self.push_url
        streamer = Streamer()
        result = streamer.init(config)
        if result not in (None, 0):
            raise RuntimeError(f"rtmp_streaming init failed with code {result}")
        self._streamer = streamer
        self._streamer_ready.set()
        logger.info("RTMP output opened: %s (%dx%d @%dfps)", self.push_url, width, height, self.fps)

    def _run(self) -> None:
        """Paced video loop.

        The cadence is the wall clock, not the producer: a frame is sent every
        ``1/fps``, and when the producer has nothing new the newest frame is
        re-sent. Holding a frame longer than that would stretch the *content*
        against the audio clock, so the split between "new frame" and "repeat" is
        left entirely to the producer's arrival rate.
        """
        try:
            interval = 1 / self.fps
            last: np.ndarray | None = None
            previous_sent: np.ndarray | None = None
            next_frame = time.perf_counter()
            while not self._stop.is_set():
                try:
                    last = self._video_queue.get_nowait()
                except queue.Empty:
                    pass
                if last is None:
                    time.sleep(min(interval, 0.02))
                    next_frame = time.perf_counter()
                    continue

                if self._streamer is None:
                    self._init_streamer(last)
                frame = self._ensure_even_frame(last)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._stream_lock:
                    self._streamer.stream_frame(rgb)
                with self._lock:
                    self.video_streamed += 1
                    if last is previous_sent:
                        self.video_repeated += 1
                previous_sent = last

                next_frame += interval
                delay = next_frame - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -interval:
                    next_frame = time.perf_counter()
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._stop.set()
            logger.error("RTMP video worker failed: %s", self._error)

    def _drain_audio(self) -> None:
        if self._streamer is None:
            return
        with self._audio_lock:
            if self._audio_pending.size < self.packet_size:
                return
            packet = np.ascontiguousarray(
                self._audio_pending[: self.packet_size], dtype=np.float32
            )
            self._audio_pending = self._audio_pending[self.packet_size :]
        with self._stream_lock:
            self._streamer.stream_frame_audio(packet)
        with self._lock:
            self.audio_streamed += 1

    def _audio_loop(self) -> None:
        try:
            next_packet = time.perf_counter()
            while not self._stop.is_set():
                if self._streamer_ready.wait(0.05):
                    self._drain_audio()
                next_packet += AUDIO_PACKET_SECONDS
                delay = next_packet - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -AUDIO_PACKET_SECONDS:
                    next_packet = time.perf_counter()
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._stop.set()
            logger.error("RTMP audio worker failed: %s", self._error)


__all__ = ["RTMPOutput", "AUDIO_PACKET_HZ", "AUDIO_PACKET_SECONDS"]
