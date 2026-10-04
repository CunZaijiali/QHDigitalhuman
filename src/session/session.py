"""Drive one avatar action -- or a whole behaviour -- into an output sink.

This is the session layer: it owns **timing** (what to send when), while
:class:`src.domain.avatar.Avatar` owns **data** and :mod:`src.engine` owns the
**algorithm**. Nothing here knows about HTTP, RTMP internals, or asyncio.

Two things happen here that the engine deliberately does not know about:

* **action rotation** — after a full ping-pong round the next action of the
  behaviour is chosen by filename weight, and the frames are cross-faded with a
  smoothstep dissolve (:mod:`src.session.transition`),
* **resource caching** — actions are loaded through
  :class:`src.domain.actions.ActionCache`, which decodes their frames into memory
  (removing the per-frame PNG cost) and keeps the most recent few resident.

Rendering and pushing are interleaved: frames come from ``engine.iter_frames()``
one inference batch at a time, so the first frame reaches the network after one
batch instead of after the whole clip.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from src.domain.actions import ActionCache, ActionRef, list_actions, weighted_choice
from src.domain.avatar import Avatar
from src.domain.models.loaders import load_models
from src.engine import create_engine
from src.engine.adapters import create_adapter
from src.engine.asr import load_audio
from src.engine.pipeline import LipSyncInput
from src.session.sinks import FrameSink, create_sinks
from src.session.transition import ActionTransition
from src.utils import find_project_root, get_config

logger = logging.getLogger(__name__)


@dataclass
class StreamResult:
    """What one streaming run produced."""

    url: str
    action: str
    frames: int
    seconds: float
    first_frame_seconds: float
    render_fps: float
    stats: dict = field(default_factory=dict)

    @property
    def realtime_ratio(self) -> float:
        """1.0 means the stream kept up with real time; <1 means it lagged."""
        target = self.stats.get("fps") or 25
        if not self.frames or not target:
            return 0.0
        return self.render_fps / float(target)


def _audio_packets(samples: np.ndarray, packet: int) -> np.ndarray:
    """Split audio into fixed 20 ms packets, padding the tail.

    TTS output has an arbitrary sample count, so the last packet is usually
    partial; ``reshape(-1, packet)`` would raise without the pad.
    """
    remainder = samples.size % packet
    feed = np.pad(samples, (0, packet - remainder)) if remainder else samples
    if feed.size < packet:
        return feed.reshape(1, -1)
    return feed.reshape(-1, packet)


def _session_settings(config: dict) -> dict:
    section = dict(config.get("session", {}) or {})
    return {
        "cache_size": int(section.get("action_cache_size", 3)),
        "strategy": str(section.get("action_loading_strategy", "lazy")),
        "preload": bool(section.get("preload_frames", True)),
        "transition_seconds": float(section.get("action_transition_seconds", 0.2)),
        "rotate_after_cycles": max(1, int(section.get("rotate_after_cycles", 1))),
        # Small batches keep the hand-over smooth: a 16-frame batch arrives in one
        # burst, so frames 1..15 are already stale by the time they are shown.
        "batch": max(1, int(section.get("render_batch_size", 4))),
    }


def _build_feature_extractor(engine: str, models, sample_rate: int, fps: int):
    normalized = engine.strip().lower().replace("_", "")
    if normalized == "wav2lip":
        from src.engine.asr import Wav2LipASR

        return Wav2LipASR(sample_rate=sample_rate, fps=fps)
    if normalized in {"musetalk", "musetalkv15"}:
        if models is None:
            raise ValueError("MuseTalk feature extraction needs the loaded Whisper models")
        from src.engine.asr import MuseTalkASR

        return MuseTalkASR(
            models.whisper, models.feature_extractor, sample_rate=sample_rate, fps=fps
        )
    raise ValueError(f"unsupported engine: {engine}")


def stream_action(
    audio_path: str | Path,
    url: str | None = None,
    *,
    avatar: Avatar | None = None,
    behavior_dir: str | Path | None = None,
    engine: str,
    fps: int = 25,
    sample_rate: int = 16000,
    duration: float | None = None,
    device: str | None = None,
    flush_seconds: float = 2.5,
    seed: int | None = None,
    sink: FrameSink | None = None,
    sink_names: Sequence[str] | None = None,
    config: dict | None = None,
    project_root: str | Path | None = None,
) -> StreamResult:
    """Render and publish in real time.

    Pass ``behavior_dir`` to rotate between the actions of a behaviour by weight,
    or ``avatar`` to pin a single action. ``sink`` takes a ready-made destination;
    otherwise ``sink_names`` (default ``("rtmp",)``) is used to build one.
    """
    if avatar is None and behavior_dir is None:
        raise ValueError("stream_action needs either an avatar or a behavior_dir")

    config = config or get_config()
    project_root = Path(project_root or find_project_root())
    settings = _session_settings(config)

    models = load_models(engine, config, project_root, device=device)
    adapter = create_adapter(engine, models)
    extractor = _build_feature_extractor(engine, models, sample_rate, fps)

    samples = load_audio(audio_path, sample_rate)
    if duration is not None:
        samples = samples[: int(duration * sample_rate)]
    frame_count = max(1, round(samples.size / sample_rate * fps))
    features = extractor.extract(audio_path, frame_count)

    cache = ActionCache(
        limit=settings["cache_size"],
        strategy=settings["strategy"],
        preload=settings["preload"],
    )
    refs: list[ActionRef] = []
    if behavior_dir is not None:
        refs = list_actions(behavior_dir, engine=engine)
        if not refs:
            raise ValueError(f"no usable action under {behavior_dir} for engine {engine}")
        cache.set_total(len(refs))
        logger.info(
            "rotation over %d action(s): %s",
            len(refs),
            ", ".join(f"{ref.action}(w={ref.weight:g})" for ref in refs),
        )
    rng = random.Random(seed)
    transition = ActionTransition(settings["transition_seconds"], fps)
    cycles = settings["rotate_after_cycles"]

    # Decoding an action's frames takes ~2s; do it before the clock starts so it
    # is neither a first-frame delay nor part of the measured render throughput.
    if avatar is not None and settings["preload"] and not avatar.preloaded:
        avatar.preload()

    sink = sink or create_sinks(
        sink_names or ("rtmp",),
        url=url,
        fps=fps,
        sample_rate=sample_rate,
        config=config,
        audio_path=audio_path,
    )
    render_started = time.perf_counter()
    sink.start()

    # Three clocks, deliberately separate:
    #   * the producer renders and pushes at *no faster* than fps, so a fast GPU
    #     never overruns the bounded queue,
    #   * the sink's video loop emits at exactly fps, re-sending the newest frame
    #     when the producer is late, which keeps the video timeline on the wall clock,
    #   * the audio below is fed at exactly one 20 ms packet per 20 ms, so audio is
    #     the master clock and a slow render shows up as stutter rather than desync.
    pace_start: float | None = None
    first_frame_at: float | None = None
    last_frame_at: float | None = None
    produced = 0
    skipped = 0
    lag_max = 0.0
    lag_sum = 0.0
    lag_count = 0
    load_seconds = 0.0
    rotations: list[str] = []

    def pick() -> ActionRef:
        if not refs:
            assert avatar is not None
            return ActionRef(
                behavior=avatar.behavior,
                action=avatar.action,
                weight=1.0,
                dir=avatar.root.parent,
                engine=avatar.engine,
                root=avatar.root,
            )
        return weighted_choice(refs, rng)

    def produce_frames() -> None:
        nonlocal pace_start, first_frame_at, last_frame_at, produced, load_seconds, skipped
        nonlocal lag_max, lag_sum, lag_count
        global_index = 0
        action_start = 0
        action_span = 0
        current_avatar: Avatar | None = None if refs else avatar
        last_frame: np.ndarray | None = None

        while global_index < frame_count and not sink.stopped:
            action_progress = global_index - action_start
            if current_avatar is None or action_progress >= action_span:
                if not refs:
                    # Single pinned action: render the whole clip in one go, no rotation.
                    current_avatar = avatar
                    if action_span <= 0:
                        action_start = 0
                        action_span = frame_count
                else:
                    ref = pick()
                    # Warm the likely next candidate exactly like the reference's
                    # lazy strategy does, so the switch does not stall on disk.
                    if settings["strategy"] == "lazy" and len(refs) > 1:
                        warm_started = time.perf_counter()
                        cache.get(weighted_choice(refs, rng).root)
                        load_seconds += time.perf_counter() - warm_started
                    load_started = time.perf_counter()
                    current_avatar = cache.get(ref.root)
                    load_seconds += time.perf_counter() - load_started
                    if produced > 0:
                        transition.arm(last_frame)
                        rotations.append(f"{ref.action}@{global_index}")
                        logger.info(
                            "action switch -> %s at frame %d", ref.action, global_index
                        )
                    action_start = global_index
                    # One complete ping-pong round of the action's own timeline,
                    # as in the reference (``index >= len(images) * 2``). Counted in
                    # *timeline* frames so the rotation period stays the same number
                    # of seconds no matter how many frames we manage to render.
                    action_span = current_avatar.frame_count * 2 * cycles

            # Follow the audio clock. Rendering is slower than real time, and a
            # frame's lips belong to audio time index/fps; stretching frames to
            # fill the gap would drift further behind every second. Skipping the
            # frames we were too slow to produce keeps the content in sync and
            # costs only smoothness.
            if pace_start is not None:
                wall_index = int((time.perf_counter() - pace_start) * fps)
                if wall_index > global_index:
                    skipped += wall_index - global_index
                    global_index = wall_index
                    if global_index >= frame_count:
                        break

            action_progress = global_index - action_start
            chunk = max(
                1,
                min(
                    action_span - action_progress,
                    settings["batch"],
                    frame_count - global_index,
                ),
            )
            request = LipSyncInput(
                resource_dir=current_avatar.root,
                audio_features=features,
                fps=fps,
                sample_rate=sample_rate,
                frame_count=chunk,
                frame_offset=global_index,
                # The action's own frames follow the *timeline*, not the number of
                # frames we managed to render: a skipped frame must advance the
                # pose too, otherwise the motion plays in slow motion whenever the
                # renderer cannot keep up.
                asset_offset=global_index - action_start,
                batch_size=settings["batch"],
            )
            pipeline = create_engine(engine, model=adapter)
            for frame in pipeline.iter_frames(request):
                now = time.perf_counter()
                if pace_start is None:
                    pace_start = now
                    first_frame_at = now
                    logger.info("first frame rendered after %.2fs", now - render_started)
                last_frame_at = now
                if pace_start is not None:
                    lag = (now - pace_start) - global_index / fps
                    lag_max = max(lag_max, lag)
                    lag_sum += lag
                    lag_count += 1
                frame = transition.apply(frame)
                last_frame = frame
                try:
                    sink.push_video_frame(frame)
                except RuntimeError as exc:
                    # The sink closed under us (shutdown); stop producing quietly.
                    logger.debug("producer stopping: %s", exc)
                    break

                produced += 1
                global_index += 1

                # Only ever *wait*; falling behind is handled by the skip above.
                target = pace_start + global_index / fps
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                if global_index >= frame_count:
                    break
                if global_index - action_start >= action_span:
                    break  # this action has played its round; pick the next one

    producer = threading.Thread(target=produce_frames, name="render-producer", daemon=True)
    try:
        producer.start()
        if not sink.wait_ready(timeout=60.0):
            raise RuntimeError("RTMP output never opened: no frame was produced within 60s")

        # 20 ms of audio per packet is the protocol contract, not a sink property:
        # the sink-agnostic audio clock is what keeps a long render in step.
        packet = max(1, sample_rate // 50)
        packets = _audio_packets(samples, packet)
        feed_started = time.perf_counter()
        for index, block in enumerate(packets):
            sink.push_audio_frame(np.ascontiguousarray(block, dtype=np.float32))
            target = feed_started + (index + 1) * (packet / sample_rate)
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        # Let the video/audio threads drain the queues and the AAC encoder flush.
        time.sleep(max(0.0, flush_seconds))
        stats = sink.stats()
    finally:
        sink.stop()
        producer.join(timeout=2)

    finished = time.perf_counter()
    # Render throughput excludes action loading: decoding an action's frames into
    # memory costs ~2s and would otherwise dominate the average.
    render_seconds = (last_frame_at or finished) - render_started
    active_seconds = max(1e-6, render_seconds - load_seconds)
    render_fps = produced / active_seconds
    stats["rendered_frames"] = produced
    stats["render_seconds"] = round(render_seconds, 3)
    stats["load_seconds"] = round(load_seconds, 3)
    stats["active_render_seconds"] = round(active_seconds, 3)
    stats["render_fps"] = round(render_fps, 2)
    stats["first_frame_seconds"] = round((first_frame_at or finished) - render_started, 3)
    stats["audio_seconds"] = round(samples.size / sample_rate, 3)
    stats["wall_seconds"] = round(finished - render_started, 3)
    stats["rotations"] = rotations
    stats["cache"] = cache.stats()
    stats["sink"] = sink.name
    stats["skipped_frames"] = skipped
    stats["render_batch"] = settings["batch"]
    # Content lag: how far behind the audio clock the frame that was just handed
    # over is. Zero means the lips on screen belong to the audio being heard.
    stats["sync_lag_seconds_max"] = round(lag_max, 2)
    stats["sync_lag_seconds_avg"] = round(lag_sum / lag_count, 2) if lag_count else 0.0

    logger.info(
        "stream finished: rendered %d/%d frames (repeated %d, dropped %d), "
        "audio %s packets, first frame %.2fs, render %.2f fps (target %d), "
        "%d action switch(es)",
        produced,
        frame_count,
        stats.get("video_repeated", 0),
        stats.get("video_dropped", 0),
        stats.get("audio_streamed", 0),
        stats["first_frame_seconds"],
        render_fps,
        fps,
        len(rotations),
    )
    return StreamResult(
        url=url or f"{sink.name}",
        action=str(behavior_dir or (avatar.root if avatar else "")),
        frames=produced,
        seconds=finished - render_started,
        first_frame_seconds=stats["first_frame_seconds"],
        render_fps=render_fps,
        stats=stats,
    )


__all__ = ["StreamResult", "stream_action"]
