"""Output sinks.

A *sink* is anything that consumes rendered frames and audio. The protocol is
exactly the contract :class:`src.session.streaming.RTMPOutput` already had; this
module only names it, adds a DirectShow virtual camera, and lets one render feed
several destinations at once.

Fan-out matters because rendering is the expensive part (MuseTalk runs well
below real time here). Rendering once and copying the frame into N sinks keeps
the frame rate; rendering once per sink would halve it per extra destination.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSink(Protocol):
    """One destination for rendered frames."""

    name: str
    #: False for destinations that are video-only (e.g. a webcam device).
    accepts_audio: bool

    def start(self) -> None: ...
    def push_video_frame(self, frame: np.ndarray) -> int: ...
    def push_audio_frame(self, samples: np.ndarray) -> None: ...
    def wait_ready(self, timeout: float | None = None) -> bool: ...
    @property
    def stopped(self) -> bool: ...
    def stats(self) -> dict[str, Any]: ...
    def stop(self) -> None: ...


class VirtualCameraSink:
    """Publish frames to a DirectShow virtual camera.

    ``pyvirtualcam`` drives an already-installed driver through shared memory, so
    OBS itself does **not** need to be running. On this machine both registered
    devices work: ``obs`` -> "OBS Virtual Camera", ``unitycapture`` -> "VTubeStudioCam".

    Video only: conference apps will see the lips move but hear nothing unless a
    virtual audio device is wired up separately.
    """

    name = "virtualcam"
    accepts_audio = False

    def __init__(
        self,
        *,
        fps: int = 25,
        backend: str = "obs",
        size: tuple[int, int] | None = None,
        device: str | None = None,
        print_fps: bool = False,
        video_queue_size: int = 24,
    ):
        if backend not in {"obs", "unitycapture"}:
            raise ValueError("virtual camera backend must be 'obs' or 'unitycapture'")
        self.fps = fps
        self.backend = backend
        self.size = size
        self.device = device
        self.print_fps = print_fps
        self.video_streamed = 0
        self.video_resized = 0
        self.video_repeated = 0
        self.video_enqueued = 0
        self.video_dropped = 0
        self.video_hold = 1
        self.error: str | None = None
        self._camera = None
        self._closed = False
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, video_queue_size))
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ sink api
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="virtualcam", daemon=True
            )
            self._thread.start()
        if self.size is not None:
            with self._lock:
                self._open_locked(self.size[0], self.size[1])

    def push_video_frame(self, frame: np.ndarray) -> int:
        """Enqueue for the delivery thread; never blocks and never paces here.

        Delivery has to be decoupled from production: a rendered batch arrives in
        one burst, and sending it straight through would freeze the camera for the
        rest of the batch (the producer's own thread is busy rendering).
        """
        if self.stopped:
            raise RuntimeError("virtual camera sink is already closed")
        frame = np.asarray(frame)
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop-oldest
                self._queue.put_nowait(frame)
                self.video_dropped += 1
            except queue.Empty:
                pass
        self.video_enqueued += 1
        return self.video_enqueued

    def _run(self) -> None:
        """Deliver at exactly ``fps``: a new frame when one is queued, otherwise
        the newest one again. Content stays on the audio clock because the
        producer only hands over frames the audio has actually reached."""
        interval = 1 / self.fps
        last: np.ndarray | None = None
        previous_sent: np.ndarray | None = None
        next_tick = time.perf_counter()
        while not self._closed:
            try:
                last = self._queue.get_nowait()
            except queue.Empty:
                pass
            if last is None:
                time.sleep(min(interval, 0.02))
                next_tick = time.perf_counter()
                continue
            try:
                self._deliver(last)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                logger.error("virtual camera delivery failed: %s", exc)
                self._closed = True
                break
            with self._lock:
                self.video_streamed += 1
                if last is previous_sent:
                    self.video_repeated += 1
            previous_sent = last

            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -interval:
                next_tick = time.perf_counter()

    def _deliver(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._camera is None:
                # Portraits are common here (768x1344); only the width/height matter.
                height, width = frame.shape[:2]
                self._open_locked(width - width % 2, height - height % 2)
            self._camera.send(self._prepare(frame))

    def push_audio_frame(self, samples: np.ndarray) -> None:
        """No-op: a DirectShow video device carries no audio."""
        return None

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout)

    @property
    def stopped(self) -> bool:
        return self._closed or self.error is not None

    def stats(self) -> dict[str, Any]:
        return {
            "sink": self.name,
            "device": self.device,
            "backend": self.backend,
            "fps": self.fps,
            "video_streamed": self.video_streamed,
            "video_enqueued": self.video_enqueued,
            "video_dropped": self.video_dropped,
            "video_repeated": self.video_repeated,
            "video_hold": self.video_hold,
            "video_resized": self.video_resized,
            "error": self.error,
        }

    def stop(self) -> None:
        with self._lock:
            self._closed = True
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2)
        with self._lock:
            camera, self._camera = self._camera, None
        if camera is not None:
            try:
                camera.close()
            except Exception as exc:  # closing a device must never mask the real error
                logger.warning("virtual camera close failed: %s", exc)

    # ------------------------------------------------------------------ internals
    def _open_locked(self, width: int, height: int) -> None:
        """Create the camera. The caller must hold ``self._lock``."""
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=self.fps,
            device=self.device,
            backend=self.backend,
            print_fps=self.print_fps,
        )
        self._camera = camera
        self.device = camera.device
        self.size = (width, height)
        logger.info(
            "virtual camera open: device=%r backend=%r %dx%d @%dfps",
            camera.device,
            camera.backend,
            width,
            height,
            self.fps,
        )
        self._ready.set()

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Exact size, contiguous and RGB48/RGB24 — what pyvirtualcam requires."""
        width, height = self.size
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            self.video_resized += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)


class LocalAudioSink:
    """Play the clip through the default output device.

    A DirectShow virtual camera carries video only, so meeting apps see the lips
    move in silence. This sink makes the voice audible on the machine itself
    (stdlib ``winsound``, no new dependency); routing it into a *virtual
    microphone* still needs a system audio device such as VB-Cable.

    Playback starts on the first audio packet, which is the moment the audio feed
    starts, so it stays in step with the video.
    """

    name = "speaker"
    accepts_audio = True

    def __init__(self, wav_path, *, volume_percent: int | None = None):
        self.wav_path = Path(wav_path)
        self.volume_percent = volume_percent
        self.started = False
        self.error: str | None = None

    def start(self) -> None:
        if not self.wav_path.is_file():
            raise FileNotFoundError(f"cannot play audio, file not found: {self.wav_path}")

    def push_video_frame(self, frame: np.ndarray) -> int:
        return 0

    def push_audio_frame(self, samples: np.ndarray) -> None:
        if self.started:
            return
        self.started = True
        try:
            import winsound

            if self.volume_percent is not None:
                winsound.PlaySound(
                    str(self.wav_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
                )
            else:
                winsound.PlaySound(
                    str(self.wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC
                )
            logger.info("playing audio locally: %s", self.wav_path)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("local audio playback failed: %s", self.error)

    def wait_ready(self, timeout: float | None = None) -> bool:
        return True

    @property
    def stopped(self) -> bool:
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "sink": self.name,
            "playing": self.started,
            "path": str(self.wav_path),
            "error": self.error,
        }

    def stop(self) -> None:
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


class CompositeSink:
    """Fan one render out to several sinks, in order."""

    name = "composite"

    def __init__(self, sinks: Sequence[FrameSink]):
        sinks = list(sinks)
        if not sinks:
            raise ValueError("a composite needs at least one sink")
        self.sinks = sinks
        self.accepts_audio = any(sink.accepts_audio for sink in sinks)
        self._failed: set[str] = set()

    def start(self) -> None:
        for sink in self.sinks:
            sink.start()

    def push_video_frame(self, frame: np.ndarray) -> int:
        count = 0
        for sink in self.sinks:
            if sink.name in self._failed or sink.stopped:
                continue
            try:
                count = sink.push_video_frame(frame)
            except Exception as exc:
                # A failing camera must not take the stream down with it.
                self._failed.add(sink.name)
                logger.error("%s sink failed, dropping it: %s", sink.name, exc)
        return count

    def push_audio_frame(self, samples: np.ndarray) -> None:
        for sink in self.sinks:
            if sink.accepts_audio and sink.name not in self._failed:
                sink.push_audio_frame(samples)

    def wait_ready(self, timeout: float | None = None) -> bool:
        return all(sink.wait_ready(timeout) for sink in self.sinks)

    @property
    def stopped(self) -> bool:
        return all(sink.stopped for sink in self.sinks)

    def stats(self) -> dict[str, Any]:
        # The first sink is the primary one, so its counters keep their old names
        # and the CLI output stays stable; the rest are reported under by_sink.
        merged = dict(self.sinks[0].stats())
        merged["sinks"] = [sink.name for sink in self.sinks]
        merged["by_sink"] = {sink.name: sink.stats() for sink in self.sinks}
        if self._failed:
            merged["failed_sinks"] = sorted(self._failed)
        return merged

    def stop(self) -> None:
        for sink in self.sinks:
            try:
                sink.stop()
            except Exception as exc:
                logger.warning("%s sink stop failed: %s", sink.name, exc)


def create_sink(
    name: str,
    *,
    url: str | None,
    fps: int,
    sample_rate: int,
    config: dict,
    audio_path: str | Path | None = None,
):
    """Build one sink by name; ``config`` supplies the ``[session]`` defaults."""
    from src.session.streaming import RTMPOutput

    section = dict(config.get("session", {}) or {})
    normalized = (name or "").strip().lower()

    if normalized in {"rtmp", "push"}:
        if not url:
            raise ValueError("the rtmp sink needs a url")
        # The bitrate lives in [digitalhuman]; not forwarding it silently pinned
        # every stream to RTMPOutput's 1.6 Mbps default, which is 0.06 bit/pixel
        # at 768x1344 and blurs the mouth away.
        digitalhuman = dict(config.get("digitalhuman", {}) or {})
        return RTMPOutput(
            url,
            fps=fps,
            bitrate=int(digitalhuman.get("bitrate", 1_600_000)),
            sample_rate=sample_rate,
            x264_preset=str(digitalhuman.get("x264_preset", "") or ""),
            x264_tune=str(digitalhuman.get("x264_tune", "") or ""),
            x264_params=str(digitalhuman.get("x264_params", "") or ""),
            video_queue_size=max(1, int(section.get("video_queue_size", 32))),
        )
    if normalized in {"virtualcam", "virtual-camera", "camera"}:
        size = section.get("virtual_camera_size")
        return VirtualCameraSink(
            fps=fps,
            backend=str(section.get("virtual_camera_backend", "obs")),
            size=tuple(int(v) for v in size) if size else None,
            device=section.get("virtual_camera_device") or None,
            video_queue_size=max(1, int(section.get("video_queue_size", 32))),
        )
    if normalized in {"speaker", "audio", "local"}:
        if not audio_path:
            raise ValueError("the speaker sink needs an audio path")
        return LocalAudioSink(audio_path)
    raise ValueError(f"unknown sink: {name!r} (expected rtmp, virtualcam or speaker)")


def create_sinks(
    names: Sequence[str],
    *,
    url: str | None,
    fps: int,
    sample_rate: int,
    config: dict,
    audio_path: str | Path | None = None,
) -> FrameSink:
    """Build every requested sink, collapsing a single one to itself."""
    sinks = [
        create_sink(
            name,
            url=url,
            fps=fps,
            sample_rate=sample_rate,
            config=config,
            audio_path=audio_path,
        )
        for name in names
    ]
    return sinks[0] if len(sinks) == 1 else CompositeSink(sinks)


__all__ = [
    "CompositeSink",
    "FrameSink",
    "LocalAudioSink",
    "VirtualCameraSink",
    "create_sink",
    "create_sinks",
]
